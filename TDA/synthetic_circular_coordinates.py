"""Run synthetic checks for the Hodge circular-coordinate pipeline.

Examples
--------
python TDA/synthetic_circular_coordinates.py --n 400 --ambient-dim 8
"""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import os
from pathlib import Path
import sys

import numpy as np
from opt_einsum import contract

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_geometry import DiffusionGeometry
from methods.circular_coordinates import circular_coordinates


def monomial_function_basis(data: np.ndarray, degree: int = 1) -> np.ndarray:
    if degree < 0:
        raise ValueError("monomial degree must be non-negative.")

    coords = np.asarray(data, dtype=float)
    coord_scale = np.std(coords, axis=0)
    coord_scale[coord_scale == 0.0] = 1.0
    coords = (coords - np.mean(coords, axis=0)) / coord_scale

    columns = [np.ones(coords.shape[0])]
    for total_degree in range(1, degree + 1):
        for indices in combinations_with_replacement(range(coords.shape[1]), total_degree):
            column = np.prod(coords[:, indices], axis=1)
            column = column - np.mean(column)
            column_norm = np.linalg.norm(column)
            if column_norm > 1e-12:
                column = column / column_norm * np.sqrt(coords.shape[0])
            columns.append(column)
    return np.column_stack(columns)


def _embed_high_dim(data: np.ndarray, ambient_dim: int, rng: np.random.Generator):
    if ambient_dim < data.shape[1]:
        raise ValueError("ambient_dim must be at least the base data dimension.")
    q, _ = np.linalg.qr(rng.normal(size=(ambient_dim, data.shape[1])))
    return contract("ij,kj->ik", data, q)


def torus(n: int, ambient_dim: int, rng: np.random.Generator):
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    major_radius = 1.5
    minor_radius = 1.0
    data = np.column_stack(
        (
            (major_radius + minor_radius * np.cos(v)) * np.cos(u),
            (major_radius + minor_radius * np.cos(v)) * np.sin(u),
            minor_radius * np.sin(v),
        )
    )
    return _embed_high_dim(data, ambient_dim, rng), {"major": u, "minor": v}


def cylinder(n: int, ambient_dim: int, rng: np.random.Generator):
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    height = rng.uniform(-1.5, 1.5, size=n)
    radius = 1.0
    data = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), height))
    return _embed_high_dim(data, ambient_dim, rng), {"circle": theta}


def _angle_alignment_score(recovered: np.ndarray, truth: np.ndarray) -> float:
    forward = abs(np.mean(np.exp(1j * (recovered - truth))))
    backward = abs(np.mean(np.exp(1j * (-recovered - truth))))
    return float(max(forward, backward))


def _best_aligned_pair(result, truth_angles: dict[str, np.ndarray]):
    if len(result.candidates) < 2:
        raise RuntimeError("Need at least two circular candidates for angle-square plots.")

    labels = list(truth_angles)
    best_pair = None
    best_score = -np.inf
    for i, first in enumerate(result.candidates):
        for second in result.candidates[i + 1 :]:
            if len(labels) == 1:
                first_score = _angle_alignment_score(
                    first.angle, truth_angles[labels[0]]
                )
                second_score = _angle_alignment_score(
                    second.angle, truth_angles[labels[0]]
                )
                pair = (first, second) if first_score >= second_score else (second, first)
                score = max(first_score, second_score)
            else:
                direct = sum(
                    _angle_alignment_score(candidate.angle, truth_angles[label])
                    for candidate, label in zip((first, second), labels)
                )
                swapped = sum(
                    _angle_alignment_score(candidate.angle, truth_angles[label])
                    for candidate, label in zip((second, first), labels)
                )
                pair = (first, second) if direct >= swapped else (second, first)
                score = max(direct, swapped) / len(labels)
            if score > best_score:
                best_score = score
                best_pair = pair
    return best_pair


def _first_exact_potential(dg, k: int = 50, min_exact_ratio: float = 0.5):
    hodge_evals, hodge_forms = dg.laplacian(1).spectrum()
    fallback = None

    for index in range(min(k, len(hodge_evals))):
        form = hodge_forms[index].real
        form_norm = form.norm()
        if form_norm <= 1e-10:
            continue

        exact_potential, _, _ = (form / form_norm).hodge_decomposition()
        exact_part = exact_potential.d()
        exact_ratio = exact_part.norm() / (form / form_norm).norm()
        if fallback is None:
            fallback = exact_potential
        if exact_ratio >= min_exact_ratio:
            return exact_potential, index, exact_ratio

    if fallback is None:
        raise RuntimeError("No nonzero Hodge 1-eigenform found for exact potential.")
    return fallback, 0, np.nan


def _plot_example(
    name: str,
    data: np.ndarray,
    truth_angles: dict[str, np.ndarray],
    result,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This script needs matplotlib for plotting.") from exc

    first, second = _best_aligned_pair(result, truth_angles)
    top = result.candidates[: min(8, len(result.candidates))]

    n_truth = len(truth_angles)
    fig, axes = plt.subplots(
        1, n_truth + 2, figsize=(4.2 * (n_truth + 2), 4), constrained_layout=True
    )

    scatter = axes[0].scatter(
        data[:, 0], data[:, 1], c=first.angle, s=10, cmap="hsv"
    )
    axes[0].set_title(f"{name}: first coordinate")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(scatter, ax=axes[0], fraction=0.046)

    if name == "cylinder":
        exact_potential, exact_index, exact_ratio = _first_exact_potential(
            result.form.dg
        )
        y_values = exact_potential.to_ambient()
        y_range = np.ptp(y_values)
        if y_range > 0:
            y_values = (y_values - np.min(y_values)) / y_range
        for ax, (label, truth) in zip(axes[1 : 1 + n_truth], truth_angles.items()):
            square = ax.scatter(first.angle, y_values, c=truth, s=10, cmap="hsv")
            ax.set_title(f"cylinder: true {label}")
            ax.set_xlim(0.0, 2.0 * np.pi)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("circular coordinate angle")
            ax.set_ylabel("first exact potential")
            fig.colorbar(square, ax=ax, fraction=0.046)
    else:
        exact_index = None
        exact_ratio = None
        for ax, (label, truth) in zip(axes[1 : 1 + n_truth], truth_angles.items()):
            square = ax.scatter(first.angle, second.angle, c=truth, s=10, cmap="hsv")
            ax.set_title(f"angle square: true {label}")
            ax.set_xlim(0.0, 2.0 * np.pi)
            ax.set_ylim(0.0, 2.0 * np.pi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("coordinate 1 angle")
            ax.set_ylabel("coordinate 2 angle")
            fig.colorbar(square, ax=ax, fraction=0.046)

    labels = [str(candidate.index) for candidate in top]
    errors = [candidate.reconstruction_error for candidate in top]
    colors = [
        "#2878b5" if candidate.passed_hodge_filter else "#9a9a9a"
        for candidate in top
    ]
    score_ax = axes[-1]
    score_ax.bar(labels, errors, color=colors)
    score_ax.set_title("1-form reconstruction error")
    score_ax.set_xlabel("Hodge eigenform index")
    score_ax.set_ylabel("relative norm")

    if exact_index is None:
        alignments = {
            label: (
                _angle_alignment_score(first.angle, truth),
                _angle_alignment_score(second.angle, truth),
            )
            for label, truth in truth_angles.items()
        }
        alignment_text = ", ".join(
            f"{label}=({score_1:.2f},{score_2:.2f})"
            for label, (score_1, score_2) in alignments.items()
        )
        aligned_score = np.mean(
            [scores[index] for index, scores in enumerate(alignments.values())]
        )
        title = (
            f"candidates=({first.index}, {second.index}), "
            f"aligned error={1.0 - aligned_score:.3f}, {alignment_text}"
        )
        summary = f"candidates=({first.index}, {second.index})"
    else:
        alignments = {
            label: (_angle_alignment_score(first.angle, truth),)
            for label, truth in truth_angles.items()
        }
        alignment_text = ", ".join(
            f"{label}={scores[0]:.2f}" for label, scores in alignments.items()
        )
        aligned_score = next(iter(alignments.values()))[0]
        title = (
            f"candidate={first.index}, exact={exact_index} "
            f"(ratio={exact_ratio:.2f}), aligned error={1.0 - aligned_score:.3f}, "
            f"{alignment_text}"
        )
        summary = f"candidate={first.index}, exact={exact_index}"
    aligned_error = 1.0 - aligned_score
    fig.suptitle(title)

    path = output_dir / f"{name}_circular_coordinate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, aligned_error, alignments, summary


def run(args):
    rng = np.random.default_rng(args.seed)
    examples = {
        "torus": torus(args.n, args.ambient_dim, rng),
        "cylinder": cylinder(args.n, args.ambient_dim, rng),
    }

    summaries = []
    for name, (data, truth_angles) in examples.items():
        function_basis = None
        if args.function_basis == "monomial":
            function_basis = monomial_function_basis(data, args.monomial_degree)

        dg = DiffusionGeometry.from_point_cloud(
            data,
            n_function_basis=args.n_function_basis,
            n_coefficients=args.n_coefficients,
            knn_kernel=args.knn_kernel,
            knn_bandwidth=args.knn_bandwidth,
            function_basis=function_basis,
        )
        result = circular_coordinates(dg, epsilon=args.epsilon, k=args.k)
        path, aligned_error, alignments, summary = _plot_example(
            name, data, truth_angles, result, args.output_dir
        )
        summaries.append((name, path, aligned_error, alignments, summary))

    for name, path, aligned_error, alignments, summary in summaries:
        alignment_text = ", ".join(
            f"{label}={','.join(f'{score:.3f}' for score in scores)}"
            for label, scores in alignments.items()
        )
        print(
            f"{name}: saved {path} | {summary} | "
            f"aligned_error={aligned_error:.3f} | "
            f"basis={args.function_basis} | alignments {alignment_text}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--ambient-dim", type=int, default=8)
    parser.add_argument("--n-function-basis", type=int, default=80)
    parser.add_argument("--n-coefficients", type=int, default=40)
    parser.add_argument("--knn-kernel", type=int, default=80)
    parser.add_argument("--knn-bandwidth", type=int, default=24)
    parser.add_argument(
        "--function-basis",
        choices=["monomial", "diffusion"],
        default="monomial",
    )
    parser.add_argument("--monomial-degree", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("TDA/output"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
