"""Synthetic Klein bottle cut-coordinate noise ablation.

Examples
--------
python TDA/synthetic/run_klein_cut_noise_ablation.py --n 500 --monomial-degree 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_geometry import DiffusionGeometry
from methods.circular_coordinates import circular_coordinates
from TDA.image_patches.patch_data import (
    build_knn_diffusion_kernel,
    cut_kernel_by_angle,
)
from TDA.image_patches.run_klein_cut_coordinates import select_coordinate
from TDA.synthetic.run_circular_coordinates import (
    klein_bottle,
    monomial_function_basis,
)


def _angle_alignment_score(recovered: np.ndarray, truth: np.ndarray) -> float:
    forward = abs(np.mean(np.exp(1j * (recovered - truth))))
    backward = abs(np.mean(np.exp(1j * (-recovered - truth))))
    return float(max(forward, backward))


def _select_candidate(result, truth_angle: Optional[np.ndarray], use_truth: bool):
    if use_truth and truth_angle is not None:
        return max(
            result.candidates,
            key=lambda candidate: _angle_alignment_score(candidate.angle, truth_angle),
        )
    return select_coordinate(result)


def _compute_cut_coordinates(
    data: np.ndarray, truth: dict[str, np.ndarray], args: argparse.Namespace
):
    function_basis = None
    n_function_basis = args.n_function_basis
    if args.function_basis == "monomial":
        function_basis = monomial_function_basis(data, args.monomial_degree)
        n_function_basis = function_basis.shape[1]

    dg = DiffusionGeometry.from_point_cloud(
        data,
        n_function_basis=n_function_basis,
        n_coefficients=args.n_coefficients,
        knn_kernel=args.knn_kernel,
        knn_bandwidth=args.knn_bandwidth,
        bandwidth=args.bandwidth,
        function_basis=function_basis,
        regularisation_method=args.regularisation_method,
        rcond=args.rcond,
    )
    base_result = circular_coordinates(dg, epsilon=args.epsilon, k=args.k)
    base_candidate = _select_candidate(
        base_result, truth["base"], not args.select_by_diagnostics
    )

    nbr_indices, kernel, bandwidths = build_knn_diffusion_kernel(
        data,
        knn_kernel=args.knn_kernel,
        knn_bandwidth=args.knn_bandwidth,
        bandwidth=args.bandwidth,
    )
    cut_kernel, cut_diagnostics = cut_kernel_by_angle(
        nbr_indices,
        kernel,
        base_candidate.angle,
        cut_angle=args.cut_angle,
        threshold=args.cut_threshold,
    )

    cut_dg = DiffusionGeometry.from_knn_kernel(
        nbr_indices=nbr_indices,
        kernel=cut_kernel,
        bandwidths=bandwidths,
        immersion_coords=data,
        data_matrix=data,
        n_function_basis=n_function_basis,
        n_coefficients=args.n_coefficients,
        function_basis=function_basis,
        regularisation_method=args.regularisation_method,
        rcond=args.rcond,
    )
    fibre_result = circular_coordinates(cut_dg, epsilon=args.epsilon, k=args.k)
    fibre_candidate = _select_candidate(
        fibre_result, truth["fibre"], not args.select_by_diagnostics
    )
    return base_candidate, fibre_candidate, cut_diagnostics


def _plot_row(
    axes,
    data: np.ndarray,
    truth: dict[str, np.ndarray],
    base_angle: np.ndarray,
    fibre_angle: np.ndarray,
    noise: float,
) -> None:
    view = data[:, :3] if data.shape[1] >= 3 else data
    ax0, ax1, ax2 = axes
    if data.shape[1] >= 3:
        scatter = ax0.scatter(
            view[:, 0], view[:, 1], view[:, 2], c=truth["base"], s=8, cmap="hsv"
        )
        ax0.set_zlabel("z")
    else:
        scatter = ax0.scatter(view[:, 0], view[:, 1], c=truth["base"], s=8, cmap="hsv")
    ax0.set_title(f"embedded data, noise={noise:g}")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_xticks([])
    ax0.set_yticks([])
    if data.shape[1] >= 3:
        ax0.set_zticks([])

    for ax, colour, title in (
        (ax1, truth["base"], "Klein square: base angle"),
        (ax2, truth["fibre"], "Klein square: fibre angle"),
    ):
        square = ax.scatter(base_angle, fibre_angle, c=colour, s=8, cmap="hsv")
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(0.0, 2.0 * np.pi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("learned base angle")
        ax.set_ylabel("learned fibre angle")


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This script needs matplotlib to save the ablation figure.") from exc

    rng = np.random.default_rng(args.seed)
    clean_data, truth = klein_bottle(args.n, args.ambient_dim, rng)
    noise_levels = tuple(args.noise_levels)

    subplot_kw = {"projection": "3d"}
    fig = plt.figure(figsize=(12, 3.8 * len(noise_levels)), constrained_layout=True)

    summaries = []
    for row, noise in enumerate(noise_levels):
        noisy_data = clean_data + noise * rng.standard_normal(clean_data.shape)
        ax0 = fig.add_subplot(len(noise_levels), 3, 3 * row + 1, **subplot_kw)
        ax1 = fig.add_subplot(len(noise_levels), 3, 3 * row + 2)
        ax2 = fig.add_subplot(len(noise_levels), 3, 3 * row + 3)

        base_candidate, fibre_candidate, cut_diagnostics = _compute_cut_coordinates(
            noisy_data, truth, args
        )
        _plot_row(
            (ax0, ax1, ax2),
            noisy_data,
            truth,
            base_candidate.angle,
            fibre_candidate.angle,
            noise,
        )
        base_alignment = _angle_alignment_score(base_candidate.angle, truth["base"])
        fibre_alignment = _angle_alignment_score(fibre_candidate.angle, truth["fibre"])
        summaries.append(
            (
                noise,
                base_candidate.index,
                fibre_candidate.index,
                base_alignment,
                fibre_alignment,
                cut_diagnostics.fraction_removed,
                cut_diagnostics.components_after,
            )
        )

    fig.suptitle(
        f"Synthetic Klein bottle cut coordinates "
        f"({args.function_basis} basis, degree={args.monomial_degree})"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    for (
        noise,
        base_index,
        fibre_index,
        base_alignment,
        fibre_alignment,
        fraction_removed,
        components_after,
    ) in summaries:
        print(
            f"noise={noise:g}: base={base_index}, fibre={fibre_index}, "
            f"alignments=({base_alignment:.3f}, {fibre_alignment:.3f}), "
            f"cut_removed={fraction_removed:.3f}, components_after={components_after}"
        )
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=450)
    parser.add_argument("--ambient-dim", type=int, default=4)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.03, 0.06, 0.1])
    parser.add_argument(
        "--function-basis",
        choices=["monomial", "diffusion"],
        default="monomial",
    )
    parser.add_argument("--monomial-degree", type=int, default=2)
    parser.add_argument("--n-function-basis", type=int, default=80)
    parser.add_argument("--n-coefficients", type=int, default=20)
    parser.add_argument("--knn-kernel", type=int, default=60)
    parser.add_argument("--knn-bandwidth", type=int, default=18)
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--regularisation-method", default="diffusion")
    parser.add_argument("--rcond", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--cut-angle", type=float, default=0.0)
    parser.add_argument("--cut-threshold", type=float, default=np.pi)
    parser.add_argument(
        "--select-by-diagnostics",
        action="store_true",
        help="Use the same diagnostic-based candidate selection as real data.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("TDA/synthetic/output"))
    parser.add_argument("--output-name", default="klein_noise_ablation.png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
