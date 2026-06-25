"""Run synthetic checks for the Hodge circular-coordinate pipeline.

Examples
--------
python TDA/synthetic_circular_coordinates.py --n 400 --ambient-dim 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_geometry import DiffusionGeometry
from methods.circular_coordinates import circular_coordinates


def _embed_high_dim(data: np.ndarray, ambient_dim: int, rng: np.random.Generator):
    if ambient_dim < data.shape[1]:
        raise ValueError("ambient_dim must be at least the base data dimension.")
    q, _ = np.linalg.qr(rng.normal(size=(ambient_dim, data.shape[1])))
    return data @ q.T


def torus(n: int, ambient_dim: int, rng: np.random.Generator):
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    major_radius = 2.0
    minor_radius = 0.7
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
    z_recovered = np.exp(1j * recovered)
    z_truth = np.exp(1j * truth)
    return float(abs(np.mean(z_recovered * np.conjugate(z_truth))))


def _angle_pair_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = np.exp(1j * first)
    second_centered = np.exp(1j * second)
    return float(abs(np.mean(first_centered * np.conjugate(second_centered))))


def _two_best_candidates(result, max_pair_search: int = 12):
    if len(result.candidates) < 2:
        raise RuntimeError("Need at least two circular candidates for angle-square plots.")

    pool = result.candidates[: min(max_pair_search, len(result.candidates))]
    best_pair = None
    best_score = np.inf
    for i, first in enumerate(pool):
        for second in pool[i + 1 :]:
            correlation = _angle_pair_correlation(first.angle, second.angle)
            score = (
                first.reconstruction_error
                + second.reconstruction_error
                + correlation
            )
            if score < best_score:
                best_score = score
                best_pair = (first, second)
    return best_pair


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

    pca = PCA(n_components=2).fit_transform(data)
    first, second = _two_best_candidates(result)
    top = result.candidates[: min(8, len(result.candidates))]

    n_truth = len(truth_angles)
    fig, axes = plt.subplots(
        1, n_truth + 2, figsize=(4.2 * (n_truth + 2), 4), constrained_layout=True
    )

    scatter = axes[0].scatter(
        pca[:, 0], pca[:, 1], c=first.angle, s=10, cmap="hsv"
    )
    axes[0].set_title(f"{name}: first coordinate")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(scatter, ax=axes[0], fraction=0.046)

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
    fig.suptitle(
        f"candidates=({first.index}, {second.index}), alignments {alignment_text}"
    )

    path = output_dir / f"{name}_circular_coordinate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, alignments, (first, second)


def run(args):
    rng = np.random.default_rng(args.seed)
    examples = {
        "torus": torus(args.n, args.ambient_dim, rng),
        "cylinder": cylinder(args.n, args.ambient_dim, rng),
    }

    summaries = []
    for name, (data, truth_angles) in examples.items():
        dg = DiffusionGeometry.from_point_cloud(
            data,
            n_function_basis=args.n_function_basis,
            n_coefficients=args.n_coefficients,
            knn_kernel=args.knn_kernel,
            knn_bandwidth=args.knn_bandwidth,
        )
        result = circular_coordinates(dg, epsilon=args.epsilon, k=args.k)
        path, alignments, candidates = _plot_example(
            name, data, truth_angles, result, args.output_dir
        )
        summaries.append((name, path, alignments, candidates))

    for name, path, alignments, candidates in summaries:
        alignment_text = ", ".join(
            f"{label}=({score_1:.3f},{score_2:.3f})"
            for label, (score_1, score_2) in alignments.items()
        )
        print(
            f"{name}: saved {path} | candidates=({candidates[0].index}, "
            f"{candidates[1].index}) | alignments {alignment_text}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--ambient-dim", type=int, default=8)
    parser.add_argument("--n-function-basis", type=int, default=40)
    parser.add_argument("--n-coefficients", type=int, default=20)
    parser.add_argument("--knn-kernel", type=int, default=40)
    parser.add_argument("--knn-bandwidth", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("TDA/output"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
