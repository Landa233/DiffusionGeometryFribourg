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
    return _embed_high_dim(data, ambient_dim, rng), u


def cylinder(n: int, ambient_dim: int, rng: np.random.Generator):
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    height = rng.uniform(-1.5, 1.5, size=n)
    radius = 1.0
    data = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), height))
    return _embed_high_dim(data, ambient_dim, rng), theta


def _angle_alignment_score(recovered: np.ndarray, truth: np.ndarray) -> float:
    z_recovered = np.exp(1j * recovered)
    z_truth = np.exp(1j * truth)
    return float(abs(np.mean(z_recovered * np.conjugate(z_truth))))


def _plot_example(
    name: str, data: np.ndarray, truth: np.ndarray, result, output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This script needs matplotlib for plotting.") from exc

    pca = PCA(n_components=2).fit_transform(data)
    score = _angle_alignment_score(result.angle, truth)
    best = result.candidate
    top = result.candidates[: min(8, len(result.candidates))]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    scatter = axes[0].scatter(
        pca[:, 0], pca[:, 1], c=result.angle, s=10, cmap="hsv"
    )
    axes[0].set_title(f"{name}: recovered angle")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(scatter, ax=axes[0], fraction=0.046)

    axes[1].scatter(
        result.coordinate_values[:, 0],
        result.coordinate_values[:, 1],
        c=truth,
        s=10,
        cmap="hsv",
    )
    axes[1].set_title("selected R2 coordinate")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    labels = [str(candidate.index) for candidate in top]
    errors = [candidate.reconstruction_error for candidate in top]
    colors = [
        "#2878b5" if candidate.passed_hodge_filter else "#9a9a9a"
        for candidate in top
    ]
    axes[2].bar(labels, errors, color=colors)
    axes[2].set_title("1-form reconstruction error")
    axes[2].set_xlabel("Hodge eigenform index")
    axes[2].set_ylabel("relative norm")

    fig.suptitle(
        f"alignment={score:.3f}, best={best.index}, "
        f"exact={best.exact_ratio:.2f}, coclosed={best.coclosed_ratio:.2f}"
    )

    path = output_dir / f"{name}_circular_coordinate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, score


def run(args):
    rng = np.random.default_rng(args.seed)
    examples = {
        "torus": torus(args.n, args.ambient_dim, rng),
        "cylinder": cylinder(args.n, args.ambient_dim, rng),
    }

    summaries = []
    for name, (data, truth) in examples.items():
        dg = DiffusionGeometry.from_point_cloud(
            data,
            n_function_basis=args.n_function_basis,
            n_coefficients=args.n_coefficients,
            knn_kernel=args.knn_kernel,
            knn_bandwidth=args.knn_bandwidth,
        )
        result = circular_coordinates(dg, epsilon=args.epsilon, k=args.k)
        path, score = _plot_example(name, data, truth, result, args.output_dir)
        summaries.append((name, path, score, result.candidate))

    for name, path, score, candidate in summaries:
        print(
            f"{name}: saved {path} | alignment={score:.3f} | "
            f"candidate={candidate.index} | error={candidate.reconstruction_error:.3f}"
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
