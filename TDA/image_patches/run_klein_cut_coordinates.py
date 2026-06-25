"""Hodge/advection-diffusion circular coordinates for image-patch Klein data."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_geometry import DiffusionGeometry
from methods.circular_coordinates import circular_coordinates
from TDA.synthetic.run_circular_coordinates import monomial_function_basis
from TDA.image_patches.patch_data import (
    build_knn_diffusion_kernel,
    cut_kernel_by_angle,
    density_core,
    load_patch_dct,
    maxmin_landmarks,
    normalise_rows,
)


def _json_float(value: Any) -> Optional[float]:
    value = float(np.real_if_close(value))
    return value if np.isfinite(value) else None


def candidate_summary(candidate) -> Dict[str, Any]:
    return {
        "index": int(candidate.index),
        "hodge_eigenvalue": _json_float(candidate.hodge_eigenvalue),
        "exact_ratio": _json_float(candidate.exact_ratio),
        "coclosed_ratio": _json_float(candidate.coclosed_ratio),
        "passed_hodge_filter": bool(candidate.passed_hodge_filter),
        "flow_eigenvalue_real": _json_float(candidate.flow_eigenvalue.real),
        "flow_eigenvalue_imag": _json_float(candidate.flow_eigenvalue.imag),
        "fit_scale": _json_float(candidate.fit_scale),
        "reconstruction_error": _json_float(candidate.reconstruction_error),
        "similarity": _json_float(candidate.similarity),
    }


def select_coordinate(result):
    """Select a stable candidate, preferring the Hodge filter."""

    pool = [c for c in result.candidates if c.passed_hodge_filter] or list(result.candidates)
    return min(pool, key=lambda c: (c.reconstruction_error, -c.similarity, c.index))


def compute_ph_signature(
    X_landmarks: np.ndarray,
    coeffs: Tuple[int, ...] = (2, 3),
    maxdim: int = 2,
) -> Dict[str, Any]:
    """Compute lightweight persistent-homology summaries with ripser if installed."""

    try:
        from ripser import ripser
    except ImportError as exc:
        raise RuntimeError("Install the optional 'tda' dependencies to use --ph.") from exc

    summaries: Dict[str, Any] = {}
    for coeff in coeffs:
        diagrams = ripser(X_landmarks, coeff=coeff, maxdim=maxdim)["dgms"]
        coeff_summary: Dict[str, Any] = {}
        for dim, diagram in enumerate(diagrams):
            finite = diagram[np.isfinite(diagram[:, 1])]
            if finite.size:
                lifetimes = finite[:, 1] - finite[:, 0]
                order = np.argsort(lifetimes)[::-1]
                longest = [
                    {
                        "birth": float(finite[idx, 0]),
                        "death": float(finite[idx, 1]),
                        "lifetime": float(lifetimes[idx]),
                    }
                    for idx in order[:10]
                ]
            else:
                longest = []
            infinite_count = int(np.count_nonzero(~np.isfinite(diagram[:, 1])))
            coeff_summary[f"H{dim}"] = {
                "longest_finite_intervals": longest,
                "infinite_count": infinite_count,
            }
        summaries[f"F_{coeff}"] = coeff_summary
    return summaries


def save_ph_diagrams(
    X_landmarks: np.ndarray,
    coeffs: Tuple[int, ...],
    maxdim: int,
    output_path: Path,
) -> None:
    try:
        from ripser import ripser
    except ImportError:
        return
    arrays = {}
    for coeff in coeffs:
        diagrams = ripser(X_landmarks, coeff=coeff, maxdim=maxdim)["dgms"]
        for dim, diagram in enumerate(diagrams):
            arrays[f"F{coeff}_H{dim}"] = diagram
    np.savez_compressed(output_path, **arrays)


def plot_outputs(
    output_stem: Path,
    landmarks: np.ndarray,
    theta_base: np.ndarray,
    theta_fibre: np.ndarray,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_stem.parent / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG plots.")
        return

    plots = [
        ("base_angle_on_first_two_dct_coords", theta_base, "base angle"),
        ("fibre_angle_on_first_two_dct_coords", theta_fibre, "fibre angle"),
    ]
    for suffix, values, title in plots:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        scatter = ax.scatter(landmarks[:, 0], landmarks[:, 1], c=values, s=8, cmap="hsv")
        ax.set_xlabel("DCT coordinate 1")
        ax.set_ylabel("DCT coordinate 2")
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, fraction=0.046)
        fig.savefig(output_stem.with_name(f"{output_stem.name}_{suffix}.png"), dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(theta_base, theta_fibre, c=theta_base, s=8, cmap="hsv")
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("base angle")
    ax.set_ylabel("fibre angle after cut")
    ax.set_title("cut-and-unwrap coordinates")
    fig.savefig(output_stem.with_name(f"{output_stem.name}_angle_square.png"), dpi=180)
    plt.close(fig)


def _maybe_default_mat_file() -> Path:
    preferred = Path("TDA/image_patches/data/nk15c30Dct.mat")
    return preferred


def run(args: argparse.Namespace) -> None:
    x = load_patch_dct(args.mat_file, args.variable)
    if not args.no_normalise:
        x = normalise_rows(x)
    if args.core_k is not None:
        x = density_core(x, k=args.core_k, percent=args.core_percent)

    landmarks = maxmin_landmarks(x, args.n_points, args.seed)
    print(f"Using {len(landmarks)} landmarks in R^{landmarks.shape[1]}.")

    ph_summary = None
    if args.ph:
        ph_summary = compute_ph_signature(landmarks, coeffs=tuple(args.coeff), maxdim=2)
        print("Persistent-homology signature is suggestive, not a hard pass/fail:")
        print(json.dumps(ph_summary, indent=2))

    dg_kwargs = {
        "n_function_basis": args.n_function_basis,
        "n_coefficients": args.n_coefficients,
        "knn_kernel": args.knn_kernel,
        "knn_bandwidth": args.knn_bandwidth,
        "regularisation_method": args.regularisation_method,
        "rcond": args.rcond,
    }
    function_basis = None
    if args.function_basis == "monomial":
        function_basis = monomial_function_basis(landmarks, args.monomial_degree)
        if function_basis.shape[1] < args.n_function_basis:
            print(
                f"Monomial basis has {function_basis.shape[1]} functions; "
                f"using that instead of --n-function-basis={args.n_function_basis}."
            )
        dg_kwargs["function_basis"] = function_basis
        dg_kwargs["n_function_basis"] = function_basis.shape[1]
    coord_kwargs = {
        "epsilon": args.epsilon,
        "k": args.k,
        "max_exact_ratio": args.max_exact_ratio,
        "min_coclosed_ratio": args.min_coclosed_ratio,
        "imag_tol": args.imag_tol,
    }

    dg = DiffusionGeometry.from_point_cloud(landmarks, **dg_kwargs)
    base_result = circular_coordinates(dg, **coord_kwargs)
    base_candidate = select_coordinate(base_result)
    theta_base = base_candidate.angle
    print(f"Base coordinate candidate: {candidate_summary(base_candidate)}")

    nbr_indices, kernel, bandwidths = build_knn_diffusion_kernel(
        landmarks,
        knn_kernel=args.knn_kernel,
        knn_bandwidth=args.knn_bandwidth,
    )
    if args.no_cut:
        cut_kernel = kernel
        cut_diagnostics = None
        print("--no-cut supplied; recomputing second coordinate on the original kernel.")
    else:
        cut_kernel, cut_diagnostics = cut_kernel_by_angle(
            nbr_indices,
            kernel,
            theta_base,
            cut_angle=args.cut_angle,
            threshold=args.cut_threshold,
        )
        print(f"Cut diagnostics: {cut_diagnostics.to_dict()}")

    cut_dg = DiffusionGeometry.from_knn_kernel(
        nbr_indices=nbr_indices,
        kernel=cut_kernel,
        bandwidths=bandwidths,
        immersion_coords=landmarks,
        data_matrix=landmarks,
        n_function_basis=dg_kwargs["n_function_basis"],
        n_coefficients=args.n_coefficients,
        function_basis=function_basis,
        regularisation_method=args.regularisation_method,
        rcond=args.rcond,
    )
    fibre_result = circular_coordinates(cut_dg, **coord_kwargs)
    fibre_candidate = select_coordinate(fibre_result)
    theta_fibre = fibre_candidate.angle
    print(f"Fibre coordinate candidate: {candidate_summary(fibre_candidate)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name or Path(args.mat_file).stem
    output_stem = args.output_dir / name

    np.savez_compressed(
        output_stem.with_suffix(".npz"),
        landmarks=landmarks,
        theta_base=theta_base,
        theta_fibre=theta_fibre,
        selected_candidate_indices=np.array([base_candidate.index, fibre_candidate.index]),
        selected_hodge_eigenvalues=np.array(
            [base_candidate.hodge_eigenvalue, fibre_candidate.hodge_eigenvalue]
        ),
        selected_reconstruction_errors=np.array(
            [base_candidate.reconstruction_error, fibre_candidate.reconstruction_error]
        ),
        selected_similarities=np.array([base_candidate.similarity, fibre_candidate.similarity]),
    )
    if args.ph:
        save_ph_diagrams(landmarks, tuple(args.coeff), 2, output_stem.with_name(f"{name}_ph_diagrams.npz"))

    args_json = vars(args).copy()
    args_json["mat_file"] = str(args.mat_file)
    args_json["output_dir"] = str(args.output_dir)
    metadata = {
        "args": args_json,
        "base_candidate": candidate_summary(base_candidate),
        "fibre_candidate": candidate_summary(fibre_candidate),
        "base_candidates": [candidate_summary(c) for c in base_result.candidates],
        "fibre_candidates": [candidate_summary(c) for c in fibre_result.candidates],
        "ph_summary": ph_summary,
        "cut_diagnostics": cut_diagnostics.to_dict() if cut_diagnostics else None,
    }
    output_stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.no_plots:
        print("--no-plots supplied; skipping PNG plots.")
    else:
        plot_outputs(output_stem, landmarks, theta_base, theta_fibre)
    print(f"Saved outputs under {output_stem.parent} with stem {output_stem.name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-file", type=Path, default=_maybe_default_mat_file())
    parser.add_argument("--variable")
    parser.add_argument("--core-k", type=int)
    parser.add_argument("--core-percent", type=float, default=30.0)
    parser.add_argument("--n-points", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--ph", action="store_true")
    parser.add_argument("--coeff", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--n-function-basis", type=int, default=80)
    parser.add_argument("--n-coefficients", type=int, default=40)
    parser.add_argument(
        "--function-basis",
        choices=["diffusion", "monomial"],
        default="diffusion",
    )
    parser.add_argument("--monomial-degree", type=int, default=1)
    parser.add_argument("--knn-kernel", type=int, default=80)
    parser.add_argument("--knn-bandwidth", type=int, default=24)
    parser.add_argument("--regularisation-method", default="diffusion")
    parser.add_argument("--rcond", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--max-exact-ratio", type=float, default=0.8)
    parser.add_argument("--min-coclosed-ratio", type=float, default=0.5)
    parser.add_argument("--imag-tol", type=float, default=1e-8)
    parser.add_argument("--cut-angle", type=float, default=0.0)
    parser.add_argument("--cut-threshold", type=float, default=math.pi)
    parser.add_argument("--no-cut", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("TDA/image_patches/output"))
    parser.add_argument("--output-name")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
