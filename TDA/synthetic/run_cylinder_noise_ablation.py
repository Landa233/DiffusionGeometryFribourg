"""Cylinder noise ablation for synthetic circular coordinates.

Examples
--------
python TDA/synthetic/run_cylinder_noise_ablation.py
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


def cylinder_from_torus_parameters(
    n: int, ambient_dim: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if ambient_dim < 3:
        raise ValueError("ambient_dim must be at least 3 for the cylinder.")

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    height = 3.0 * phi / (2.0 * np.pi) - 1.5
    cylinder = np.column_stack((np.cos(theta), np.sin(theta), height))
    q, _ = np.linalg.qr(rng.normal(size=(ambient_dim, 3)))
    return contract("ij,kj->ik", cylinder, q), {"theta": theta, "phi": phi}


def scale_to_angle(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    value_range = np.ptp(values)
    if value_range <= 1e-12:
        return np.zeros_like(values)
    return 2.0 * np.pi * (values - np.min(values)) / value_range


def best_phi_coordinate(dg, phi: np.ndarray, max_search: int) -> tuple[np.ndarray, int, float]:
    hodge_evals, hodge_forms = dg.laplacian(1).spectrum()
    best_coordinate = None
    best_index = -1
    best_score = -np.inf

    for index in range(min(max_search, len(hodge_evals))):
        form = hodge_forms[index].real
        form_norm = form.norm()
        if form_norm <= 1e-10:
            continue
        potential, _, _ = (form / form_norm).hodge_decomposition()
        coordinate = scale_to_angle(potential.to_ambient())
        score = _angle_alignment_score(coordinate, phi)
        if score > best_score:
            best_coordinate = coordinate
            best_index = index
            best_score = score

    if best_coordinate is None:
        raise RuntimeError("No nonzero Hodge 1-eigenform found for phi coordinate.")
    return best_coordinate, best_index, best_score


def best_theta_candidate(result, theta: np.ndarray):
    return max(
        result.candidates,
        key=lambda candidate: _angle_alignment_score(candidate.angle, theta),
    )


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
    theta_candidate = best_theta_candidate(result, truth["theta"])
    phi_coordinate, phi_index, phi_alignment = best_phi_coordinate(
        dg, truth["phi"], args.exact_search
    )
    return theta_candidate, phi_coordinate, phi_index, phi_alignment


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
    ax0.set_title(f"cylinder data, noise={noise:g}")
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
        ax.set_xlabel("learned circular coordinate")
        ax.set_ylabel("learned phi coordinate")
        ax.set_title(title)


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This script needs matplotlib to save the ablation figure.") from exc

    rng = np.random.default_rng(args.seed)
    clean_data, truth = cylinder_from_torus_parameters(args.n, args.ambient_dim, rng)
    noise_levels = tuple(args.noise_levels)

    fig = plt.figure(figsize=(11, 3.6 * len(noise_levels)), constrained_layout=True)
    summaries = []
    for row, noise in enumerate(noise_levels):
        noisy_data = clean_data + noise * rng.standard_normal(clean_data.shape)
        ax0 = fig.add_subplot(len(noise_levels), 3, 3 * row + 1, projection="3d")
        ax1 = fig.add_subplot(len(noise_levels), 3, 3 * row + 2)
        ax2 = fig.add_subplot(len(noise_levels), 3, 3 * row + 3)

        theta_candidate, phi_coordinate, phi_index, phi_alignment = compute_coordinates(
            noisy_data, truth, args
        )
        plot_row(
            (ax0, ax1, ax2),
            noisy_data,
            truth,
            theta_candidate.angle,
            phi_coordinate,
            noise,
        )
        summaries.append(
            (
                noise,
                theta_candidate.index,
                phi_index,
                _angle_alignment_score(theta_candidate.angle, truth["theta"]),
                phi_alignment,
            )
        )

    fig.suptitle("Synthetic cylinder noise ablation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    for noise, theta_index, phi_index, theta_alignment, phi_alignment in summaries:
        print(
            f"noise={noise:g}: theta_candidate={theta_index}, "
            f"phi_exact={phi_index}, "
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
    parser.add_argument("--exact-search", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("TDA/synthetic/output"))
    parser.add_argument("--output-name", default="cylinder_noise_ablation.png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
