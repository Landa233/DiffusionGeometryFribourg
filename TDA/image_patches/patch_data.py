"""Utilities for natural-image 3x3 DCT patch experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from diffusion_geometry.core import knn_graph, markov_chain


@dataclass(frozen=True)
class CutDiagnostics:
    edges_before: int
    edges_after: int
    fraction_removed: float
    components_before: int
    components_after: int
    min_degree_after: int
    median_degree_after: float
    max_degree_after: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _numeric_2d_variables(mat: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        key: value
        for key, value in mat.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 2
        and np.issubdtype(value.dtype, np.number)
    }


def load_patch_dct(
    mat_file: Union[str, Path], variable: Optional[str] = None
) -> np.ndarray:
    """Load a 3x3 natural-image patch array in DCT coordinates.

    The classic files store eight non-DC DCT coefficients, sometimes as
    ``(n, 8)`` and sometimes as ``(8, n)``. This function returns ``(n, 8)``.
    """

    mat = loadmat(mat_file)
    variables = _numeric_2d_variables(mat)
    if variable is None:
        if not variables:
            raise ValueError(f"No non-private 2D numeric arrays found in {mat_file}.")
        variable = max(variables, key=lambda key: variables[key].size)
        print(f"Selected Matlab variable {variable!r} from {mat_file}.")
    if variable not in variables:
        available = ", ".join(sorted(variables)) or "<none>"
        raise KeyError(f"Variable {variable!r} not found. Available 2D numeric arrays: {available}")

    x = np.asarray(variables[variable], dtype=float)
    if x.shape[1] == 8:
        return np.ascontiguousarray(x)
    if x.shape[0] == 8:
        return np.ascontiguousarray(x.T)
    raise ValueError(
        f"Expected a DCT array with one axis of length 8, got shape {x.shape}."
    )


def normalise_rows(X: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Normalise rows to unit norm and drop near-zero rows."""

    x = np.asarray(X, dtype=float)
    norms = np.linalg.norm(x, axis=1)
    keep = norms > tol
    if not np.all(keep):
        print(f"Dropping {np.count_nonzero(~keep)} near-zero DCT rows.")
    return x[keep] / norms[keep, None]


def density_core(X: np.ndarray, k: int = 15, percent: float = 30.0) -> np.ndarray:
    """Compute the density core ``X(k, p)`` by kth-neighbour radius."""

    x = np.asarray(X, dtype=float)
    if not 0 < percent <= 100:
        raise ValueError("percent must lie in (0, 100].")
    if k < 1 or k >= len(x):
        raise ValueError("k must satisfy 1 <= k < len(X).")

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x)
    distances, _ = nbrs.kneighbors(x)
    kth_radius = distances[:, k]
    cutoff = np.percentile(kth_radius, percent)
    keep = kth_radius <= cutoff
    print(
        f"Density core X({k}, {percent:g}) keeps {np.count_nonzero(keep)} "
        f"of {len(x)} rows."
    )
    return x[keep]


def maxmin_landmarks(X: np.ndarray, n_points: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point/max-min landmark selection."""

    x = np.asarray(X, dtype=float)
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if n_points >= len(x):
        return x.copy()

    rng = np.random.default_rng(seed)
    indices = np.empty(n_points, dtype=int)
    indices[0] = int(rng.integers(len(x)))
    min_sq_dist = np.sum((x - x[indices[0]]) ** 2, axis=1)

    for pos in range(1, n_points):
        indices[pos] = int(np.argmax(min_sq_dist))
        next_sq_dist = np.sum((x - x[indices[pos]]) ** 2, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, next_sq_dist)
    return x[indices]


def build_knn_diffusion_kernel(
    X: np.ndarray,
    *,
    knn_kernel: int = 32,
    knn_bandwidth: int = 8,
    bandwidth: Optional[float] = None,
    c: float = 0.0,
    bandwidth_variability: float = -0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the same kNN Markov kernel used by ``DiffusionGeometry``."""

    nbr_distances, nbr_indices = knn_graph(np.asarray(X, dtype=float), knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances=nbr_distances,
        nbr_indices=nbr_indices,
        c=c,
        bandwidth_variability=bandwidth_variability,
        knn_bandwidth=knn_bandwidth,
        bandwidth=bandwidth,
    )
    return nbr_indices, kernel, bandwidths


def _component_count(nbr_indices: np.ndarray, kernel: np.ndarray) -> int:
    n, k = nbr_indices.shape
    rows = np.repeat(np.arange(n), k)
    mask = kernel.ravel() > 0
    graph = coo_matrix(
        (np.ones(np.count_nonzero(mask)), (rows[mask], nbr_indices.ravel()[mask])),
        shape=(n, n),
    )
    graph = graph.maximum(graph.T)
    return int(connected_components(graph, directed=False, return_labels=False))


def cut_kernel_by_angle(
    nbr_indices: np.ndarray,
    kernel: np.ndarray,
    theta: np.ndarray,
    *,
    cut_angle: float = 0.0,
    threshold: float = np.pi,
) -> Tuple[np.ndarray, CutDiagnostics]:
    """Sever kNN kernel edges that cross an angular seam.

    Angles are shifted so the cut lies at ``cut_angle`` and reduced to
    ``[0, 2*pi)``. Edges with linear angular difference larger than
    ``threshold`` are removed and remaining rows are renormalised.
    """

    nbr_indices = np.asarray(nbr_indices)
    kernel = np.asarray(kernel, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if nbr_indices.shape != kernel.shape:
        raise ValueError("nbr_indices and kernel must have the same shape.")
    if theta.shape != (kernel.shape[0],):
        raise ValueError("theta must have shape (n_samples,).")

    n, k = kernel.shape
    phi = np.mod(theta - cut_angle, 2.0 * np.pi)
    diff = np.abs(phi[:, None] - phi[nbr_indices])
    crosses_cut = diff > threshold
    crosses_cut[nbr_indices == np.arange(n)[:, None]] = False

    edges_before = int(np.count_nonzero(kernel > 0))
    components_before = _component_count(nbr_indices, kernel)

    cut_kernel = kernel.copy()
    cut_kernel[crosses_cut] = 0.0
    row_sums = cut_kernel.sum(axis=1)
    zero_rows = row_sums <= 0
    if np.any(zero_rows):
        for row in np.flatnonzero(zero_rows):
            self_positions = np.flatnonzero(nbr_indices[row] == row)
            if self_positions.size == 0:
                raise RuntimeError(f"Cut removed all edges from row {row}; no self-loop found.")
            cut_kernel[row, self_positions[0]] = 1.0
        row_sums = cut_kernel.sum(axis=1)
    cut_kernel /= row_sums[:, None]

    edges_after = int(np.count_nonzero(cut_kernel > 0))
    degrees_after = np.count_nonzero(cut_kernel > 0, axis=1)
    diagnostics = CutDiagnostics(
        edges_before=edges_before,
        edges_after=edges_after,
        fraction_removed=(
            (edges_before - edges_after) / edges_before if edges_before else 0.0
        ),
        components_before=components_before,
        components_after=_component_count(nbr_indices, cut_kernel),
        min_degree_after=int(np.min(degrees_after)),
        median_degree_after=float(np.median(degrees_after)),
        max_degree_after=int(np.max(degrees_after)),
    )
    return cut_kernel, diagnostics
