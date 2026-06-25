"""Torus noise ablation for synthetic circular coordinates.

Examples
--------
python TDA/synthetic/run_torus_noise_ablation.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from opt_einsum import contract

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_geometry import DiffusionGeometry
from methods.circular_coordinates import circular_coordinates
from TDA.synthetic.run_circular_coordinates import (
    _angle_alignment_score,
    monomial_function_basis,
)


def torus_from_parameters(
    n: int, ambient_dim: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if ambient_dim < 3:
        raise ValueError("ambient_dim must be at least 3 for the torus.")

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    major_radius = 1.5
    minor_radius = 1.0
    torus = np.column_stack(
        (
            (major_radius + minor_radius * np.cos(phi)) * np.cos(theta),
            (major_radius + minor_radius * np.cos(phi)) * np.sin(theta),
            minor_radius * np.sin(phi),
        )
    )
    q, _ = np.linalg.qr(rng.normal(size=(ambient_dim, 3)))
    return contract("ij,kj->ik", torus, q), {"theta": theta, "phi": phi}


def best_coordinate_pair(result, truth: dict[str, np.ndarray]):
    labels = ("theta", "phi")
    best_pair = None
    best_score = -np.inf

    for i, first in enumerate(result.candidates):
        for second in result.candidates[i + 1 :]:
            direct = sum(
                _angle_alignment_score(candidate.angle, truth[label])
                for candidate, label in zip((first, second), labels)
            )
            swapped = sum(
                _angle_alignment_score(candidate.angle, truth[label])
                for candidate, label in zip((second, first), labels)
            )
            pair = (first, second) if direct >= swapped else (second, first)
            score = max(direct, swapped) / 2.0
            if score > best_score:
                best_pair = pair
                best_score = score

    if best_pair is None:
        raise RuntimeError("Need at least two circular-coordinate candidates.")
    return best_pair


def compute_coordinates(data: np.ndarray, truth: dict[str, np.ndarray], args):
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
        function_basis=function_basis,
        regularisation_method=args.regularisation_method,
        rcond=args.rcond,
    )
    result = circular_coordinates(dg, epsilon=args.epsilon, k=args.k)
    return best_coordinate_pair(result, truth)


def plot_row(axes, data, truth, theta_coordinate, phi_coordinate, noise):
    ax0, ax1, ax2 = axes

    if data.shape[1] >= 3:
        ax0.scatter(
            data[:, 0], data[:, 1], data[:, 2], c=truth["theta"], s=8, cmap="hsv"
        )
        ax0.set_zlabel("z")
        ax0.set_zticks([])
    else:
        ax0.scatter(data[:, 0], data[:, 1], c=truth["theta"], s=8, cmap="hsv")
    ax0.set_title(f"torus data, noise={noise:g}")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_xticks([])
    ax0.set_yticks([])

    for ax, colour, title in (
        (ax1, truth["theta"], "square coloured by theta"),
        (ax2, truth["phi"], "square coloured by phi"),
    ):
        ax.scatter(theta_coordinate, phi_coordinate, c=colour, s=8, cmap="hsv")
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(0.0, 2.0 * np.pi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("learned theta coordinate")
        ax.set_ylabel("learned phi coordinate")
        ax.set_title(title)


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This script needs matplotlib to save the ablation figure.") from exc

    rng = np.random.default_rng(args.seed)
    clean_data, truth = torus_from_parameters(args.n, args.ambient_dim, rng)
    noise_levels = tuple(args.noise_levels)

    fig = plt.figure(figsize=(11, 3.6 * len(noise_levels)), constrained_layout=True)
    summaries = []
    for row, noise in enumerate(noise_levels):
        noisy_data = clean_data + noise * rng.standard_normal(clean_data.shape)
        ax0 = fig.add_subplot(len(noise_levels), 3, 3 * row + 1, projection="3d")
        ax1 = fig.add_subplot(len(noise_levels), 3, 3 * row + 2)
        ax2 = fig.add_subplot(len(noise_levels), 3, 3 * row + 3)

        theta_candidate, phi_candidate = compute_coordinates(noisy_data, truth, args)
        plot_row(
            (ax0, ax1, ax2),
            noisy_data,
            truth,
            theta_candidate.angle,
            phi_candidate.angle,
            noise,
        )
        summaries.append(
            (
                noise,
                theta_candidate.index,
                phi_candidate.index,
                _angle_alignment_score(theta_candidate.angle, truth["theta"]),
                _angle_alignment_score(phi_candidate.angle, truth["phi"]),
            )
        )

    fig.suptitle("Synthetic torus noise ablation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    for noise, theta_index, phi_index, theta_alignment, phi_alignment in summaries:
        print(
            f"noise={noise:g}: theta_candidate={theta_index}, "
            f"phi_candidate={phi_index}, "
            f"alignments=({theta_alignment:.3f}, {phi_alignment:.3f})"
        )
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=450)
    parser.add_argument("--ambient-dim", type=int, default=8)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15]
    )
    parser.add_argument(
        "--function-basis",
        choices=["monomial", "diffusion"],
        default="monomial",
    )
    parser.add_argument("--monomial-degree", type=int, default=2)
    parser.add_argument("--n-function-basis", type=int, default=80)
    parser.add_argument("--n-coefficients", type=int, default=40)
    parser.add_argument("--knn-kernel", type=int, default=80)
    parser.add_argument("--knn-bandwidth", type=int, default=24)
    parser.add_argument("--regularisation-method", default="diffusion")
    parser.add_argument("--rcond", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("TDA/synthetic/output"))
    parser.add_argument("--output-name", default="torus_noise_ablation.png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
