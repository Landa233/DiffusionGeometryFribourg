"""Circular coordinates from Hodge eigenforms.

This module implements the TDA pipeline prototyped in ``figures/tda.ipynb``:
low-energy 1-forms are converted to advection-diffusion operators, their first
complex eigenfunction gives an ``R^2`` coordinate plane, and candidates are
ranked by how well ``x dy - y dx`` recovers the source 1-form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CircularCoordinateCandidate:
    """Diagnostics for one 1-form candidate."""

    index: int
    hodge_eigenvalue: float
    form: Any
    exact_ratio: float
    coclosed_ratio: float
    passed_hodge_filter: bool
    flow_eigenvalue: complex
    coordinate_functions: tuple[Any, Any]
    coordinate_values: np.ndarray
    angle: np.ndarray
    rotation_form: Any
    fit_scale: float
    reconstruction_error: float
    similarity: float


@dataclass(frozen=True)
class CircularCoordinateResult:
    """Selected circular coordinate and the ranked candidate list."""

    coordinate_functions: tuple[Any, Any]
    coordinate_values: np.ndarray
    angle: np.ndarray
    form: Any
    candidate: CircularCoordinateCandidate
    candidates: tuple[CircularCoordinateCandidate, ...]


def _safe_norm(tensor) -> float:
    value = tensor.norm()
    return float(np.real_if_close(value))


def _weighted_normalize_complex(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total_weight = np.sum(weights)
    centered = values - np.sum(weights * values) / total_weight
    scale = np.sqrt(np.sum(weights * np.abs(centered) ** 2) / total_weight)
    if not np.isfinite(scale) or scale <= 0:
        return centered
    return centered / scale


def _hodge_scores(form, norm_floor: float) -> tuple[float, float, bool]:
    norm = _safe_norm(form)
    if norm <= norm_floor:
        return np.inf, 0.0, False

    exact_potential, coexact_potential, harmonic_part = form.hodge_decomposition()
    exact_part = exact_potential.d()
    if coexact_potential is None:
        coclosed_part = harmonic_part
    else:
        coclosed_part = coexact_potential.codifferential() + harmonic_part

    exact_ratio = _safe_norm(exact_part) / norm
    coclosed_ratio = _safe_norm(coclosed_part) / norm
    finite = np.isfinite(exact_ratio) and np.isfinite(coclosed_ratio)
    return exact_ratio, coclosed_ratio, finite


def _first_circular_eigenfunction(form, epsilon: float, imag_tol: float):
    dg = form.dg
    operator = form.sharp().operator - epsilon * dg.laplacian(0)
    evals, functions = operator.spectrum()
    evals = np.asarray(evals)

    positive = np.flatnonzero(evals.imag > imag_tol)
    if positive.size == 0:
        nonreal = np.flatnonzero(np.abs(evals.imag) > imag_tol)
        if nonreal.size == 0:
            return None, None
        idx = nonreal[0]
    else:
        idx = positive[0]

    function = functions[idx]
    if evals[idx].imag < 0:
        function = function.real - 1j * function.imag
        eigenvalue = np.conjugate(evals[idx])
    else:
        eigenvalue = evals[idx]
    return eigenvalue, function


def _coordinate_rotation_form(dg, coordinate_values: np.ndarray):
    x_values = coordinate_values[:, 0]
    y_values = coordinate_values[:, 1]
    x = dg.function(x_values)
    y = dg.function(y_values)
    rotation_form = y.d() * x - x.d() * y
    return x, y, rotation_form


def _score_coordinate_form(alpha, rotation_form, norm_floor: float):
    alpha_norm = _safe_norm(alpha)
    beta_norm = _safe_norm(rotation_form)
    if alpha_norm <= norm_floor or beta_norm <= norm_floor:
        return 0.0, np.inf, 0.0

    inner = alpha.dg.inner(alpha, rotation_form)
    inner = float(np.real_if_close(inner))
    beta_inner = beta_norm * beta_norm
    fit_scale = inner / beta_inner if beta_inner > norm_floor else 0.0
    residual = alpha - fit_scale * rotation_form
    reconstruction_error = _safe_norm(residual) / alpha_norm
    similarity = abs(inner) / (alpha_norm * beta_norm)
    return fit_scale, reconstruction_error, similarity


def circular_coordinates(
    dg,
    *,
    epsilon: float = 1.0,
    k: int = 20,
    max_exact_ratio: float = 0.8,
    min_coclosed_ratio: float = 0.5,
    imag_tol: float = 1e-8,
    norm_floor: float = 1e-10,
) -> CircularCoordinateResult:
    """Compute a circular coordinate from Hodge 1-form eigenvectors.

    Parameters
    ----------
    dg:
        A ``DiffusionGeometry`` instance.
    epsilon:
        Diffusion strength in ``alpha.sharp() - epsilon * Delta``.
    k:
        Number of Hodge 1-form eigenvectors to inspect.
    max_exact_ratio:
        Candidate 1-forms with a larger exact Hodge component are deprioritised.
    min_coclosed_ratio:
        Candidate 1-forms with a smaller coexact-plus-harmonic component are
        deprioritised.
    imag_tol:
        Minimum imaginary part used to identify circular eigenfunctions.
    norm_floor:
        Numerical floor for norms used in ratios and least-squares scoring.

    Returns
    -------
    CircularCoordinateResult
        The selected coordinate as two diffusion-basis functions, pointwise
        ``(x, y)`` values, angles, the selected 1-form, and candidate diagnostics.
    """

    if k <= 0:
        raise ValueError("k must be positive.")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    hodge_evals, hodge_forms = dg.laplacian(1).spectrum()
    candidates: list[CircularCoordinateCandidate] = []

    for index in range(min(k, len(hodge_evals))):
        form = hodge_forms[index].real
        form_norm = _safe_norm(form)
        if form_norm <= norm_floor:
            continue
        alpha = form / form_norm

        exact_ratio, coclosed_ratio, finite_hodge = _hodge_scores(alpha, norm_floor)
        passed_hodge_filter = (
            finite_hodge
            and exact_ratio <= max_exact_ratio
            and coclosed_ratio >= min_coclosed_ratio
        )

        flow_eigenvalue, coordinate_function = _first_circular_eigenfunction(
            alpha, epsilon, imag_tol
        )
        if coordinate_function is None:
            continue

        complex_values = coordinate_function.to_ambient()
        complex_values = _weighted_normalize_complex(complex_values, dg.measure)
        coordinate_values = np.column_stack(
            (complex_values.real, complex_values.imag)
        )
        x, y, rotation_form = _coordinate_rotation_form(dg, coordinate_values)
        fit_scale, reconstruction_error, similarity = _score_coordinate_form(
            alpha, rotation_form, norm_floor
        )

        angle = np.mod(
            np.arctan2(coordinate_values[:, 1], coordinate_values[:, 0]), 2 * np.pi
        )
        candidates.append(
            CircularCoordinateCandidate(
                index=index,
                hodge_eigenvalue=float(np.real_if_close(hodge_evals[index])),
                form=alpha,
                exact_ratio=exact_ratio,
                coclosed_ratio=coclosed_ratio,
                passed_hodge_filter=passed_hodge_filter,
                flow_eigenvalue=complex(flow_eigenvalue),
                coordinate_functions=(x, y),
                coordinate_values=coordinate_values,
                angle=angle,
                rotation_form=rotation_form,
                fit_scale=fit_scale,
                reconstruction_error=reconstruction_error,
                similarity=similarity,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No circular candidates were found. Try increasing k or decreasing imag_tol."
        )

    filtered = [candidate for candidate in candidates if candidate.passed_hodge_filter]
    pool = filtered if filtered else candidates
    best = min(pool, key=lambda candidate: candidate.reconstruction_error)
    ranked = tuple(
        sorted(candidates, key=lambda candidate: candidate.reconstruction_error)
    )

    return CircularCoordinateResult(
        coordinate_functions=best.coordinate_functions,
        coordinate_values=best.coordinate_values,
        angle=best.angle,
        form=best.form,
        candidate=best,
        candidates=ranked,
    )


hodge_circular_coordinates = circular_coordinates
