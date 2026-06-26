"""Gardner-style persistent-cohomology circular coordinates on a filled cylinder.

This script starts from a randomly sampled cylinder, progressively adds uniform
interior points through the middle, and decodes a circular coordinate from the
most persistent H1 cocycle using
``TDA.gardner2022_persistent_homology.get_coords_compat``.

Example
-------
.venv/bin/python TDA/perfect_circle_circular_coordinates.py
"""

from __future__ import annotations
from TDA.gardner2022_persistent_homology import get_coords_compat
from methods.circular_coordinates import circular_coordinates
from diffusion_geometry import DiffusionGeometry

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from ripser import ripser
from scipy.sparse.linalg import lsmr
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTLIER_FRACTIONS = (0.0, 0.5, 1.0, 2.0)


def make_cylinder(
    n_points: int,
    radius: float,
    height: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_points < 8:
        raise ValueError("n_points must be at least 8.")

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    z = rng.uniform(-height / 2.0, height / 2.0, size=n_points)
    points = np.column_stack(
        (radius * np.cos(theta), radius * np.sin(theta), z))
    return points, theta, z


def add_middle_points(
    points: np.ndarray,
    fraction: float,
    radius: float,
    height: float,
    middle_radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction < 0.0:
        raise ValueError("middle-point fraction must be non-negative.")
    if not 0.0 < middle_radius <= 1.0:
        raise ValueError("middle_radius must be in (0, 1].")

    n_middle = int(round(points.shape[0] * fraction))
    inlier_mask = np.ones(points.shape[0] + n_middle, dtype=bool)
    if n_middle == 0:
        return points.copy(), inlier_mask

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_middle)
    radial = radius * middle_radius * \
        np.sqrt(rng.uniform(0.0, 1.0, size=n_middle))
    z = rng.uniform(-height / 2.0, height / 2.0, size=n_middle)
    middle = np.column_stack(
        (
            radial * np.cos(theta),
            radial * np.sin(theta),
            z,
        )
    )
    inlier_mask[points.shape[0]:] = False
    return np.vstack((points, middle)), inlier_mask


def choose_landmarks(
    points: np.ndarray,
    n_landmarks: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_landmarks = min(n_landmarks, points.shape[0])
    indices = rng.choice(points.shape[0], size=n_landmarks, replace=False)
    return points[indices], indices


def gardner_circular_coordinate(
    points: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, object]:
    landmarks, landmark_indices = choose_landmarks(
        points, args.n_landmarks, rng)
    persistence = ripser(
        landmarks,
        maxdim=1,
        coeff=args.coeff,
        do_cocycles=True,
    )

    diagrams = persistence["dgms"]
    cocycles = persistence["cocycles"][1]
    h1_diagram = diagrams[1]
    if h1_diagram.size == 0 or not cocycles:
        raise RuntimeError("ripser found no H1 cocycles.")

    births = h1_diagram[:, 0]
    deaths = h1_diagram[:, 1].copy()
    deaths[np.isinf(deaths)] = 0.0
    lifetimes = deaths - births
    order = np.argsort(lifetimes)
    selected = int(order[-1])
    threshold_source = int(order[-2]) if len(order) > 1 else selected
    threshold = (
        births[threshold_source]
        + (deaths[threshold_source] -
           births[threshold_source]) * args.dec_thresh
    )

    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    utils = SimpleNamespace(np=np, lsmr=lsmr)
    landmark_coords, decoded_vertices = get_coords_compat(
        utils,
        cocycles[selected].copy(),
        threshold,
        len(landmarks),
        persistence["dperm2all"],
        args.coeff,
    )

    landmark_angle = np.empty(len(landmarks))
    landmark_distances = persistence["dperm2all"]
    nearest_decoded = np.argmin(
        landmark_distances[:, decoded_vertices], axis=1)
    landmark_angle[:] = landmark_coords[nearest_decoded] * 2.0 * np.pi
    landmark_angle[decoded_vertices] = landmark_coords * 2.0 * np.pi

    nearest_landmark = np.argmin(cdist(points, landmarks), axis=1)
    point_angle = landmark_angle[nearest_landmark]
    return {
        "angle": point_angle,
        "diagram": h1_diagram,
        "selected": selected,
        "birth": float(births[selected]),
        "death": float(deaths[selected]),
        "lifetime": float(lifetimes[selected]),
        "threshold": float(threshold),
        "n_h1": len(h1_diagram),
        "landmark_indices": landmark_indices,
    }


def diffusion_geometry_circular_coordinate(
    points: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    dg = DiffusionGeometry.from_point_cloud(
        points,
        n_function_basis=args.dg_n_function_basis,
        n_coefficients=args.dg_n_coefficients,
        knn_kernel=args.dg_knn_kernel,
        knn_bandwidth=args.dg_knn_bandwidth,
    )
    result = circular_coordinates(
        dg,
        epsilon=args.dg_epsilon,
        k=args.dg_k,
        max_exact_ratio=args.dg_max_exact_ratio,
        min_coclosed_ratio=args.dg_min_coclosed_ratio,
        imag_tol=args.dg_imag_tol,
    )
    return {
        "angle": result.angle,
        "candidate_index": result.candidate.index,
        "reconstruction_error": float(result.candidate.reconstruction_error),
        "similarity": float(result.candidate.similarity),
    }


def circular_alignment(recovered: np.ndarray, truth: np.ndarray) -> tuple[float, int, float]:
    options = []
    for orientation in (1, -1):
        shifted = orientation * recovered
        phase = np.angle(np.mean(np.exp(1j * (truth - shifted))))
        residual = np.angle(np.exp(1j * (shifted + phase - truth)))
        score = abs(np.mean(np.exp(1j * residual)))
        options.append((float(score), orientation, float(phase)))
    return max(options, key=lambda item: item[0])


def summarize(
    result: dict[str, object],
    truth_theta: np.ndarray,
    height_values: np.ndarray,
    inlier_mask: np.ndarray,
) -> dict[str, float]:
    inlier_angles = result["angle"][inlier_mask]
    score, orientation, phase = circular_alignment(inlier_angles, truth_theta)
    aligned = orientation * inlier_angles + phase
    mean_error = float(
        np.mean(np.abs(np.angle(np.exp(1j * (aligned - truth_theta))))))
    height_correlation = float(
        abs(np.corrcoef(np.cos(inlier_angles), height_values)[0, 1])
    )
    return {
        "score": score,
        "orientation": orientation,
        "phase": phase,
        "mean_error": mean_error,
        "height_correlation": height_correlation,
        "n_outliers": int(np.count_nonzero(~inlier_mask)),
    }


def zoom_cylinder_axes(ax, radius: float = 1.1, half_height: float = 1.1) -> None:
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-half_height, half_height)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.5))
    except AttributeError:
        pass


def plot_runs(runs: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(
        output_path.parent / ".matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Install matplotlib or run with --no-plot.") from exc

    clean_points = runs[0]["clean_points"]
    truth_theta = runs[0]["truth_theta"]
    radius_limit = 1.08 * \
        float(np.max(np.linalg.norm(clean_points[:, :2], axis=1)))
    half_height = 1.08 * float(np.max(np.abs(clean_points[:, 2])))

    fig = plt.figure(figsize=(15.0, 11.0))
    grid = fig.add_gridspec(
        len(runs),
        5,
        left=0.01,
        right=0.995,
        bottom=0.03,
        top=0.91,
        wspace=0.05,
        hspace=0.33,
    )
    column_titles = (
        "Ground Truth\n 1600 points",
        "XY Projection",
        "Persistence Diagram",
        "Persistent Coordinate",
        "Diffusion Geometry",
    )
    for index, title in enumerate(column_titles):
        fig.text(
            0.105 + index * 0.197,
            0.985,
            title,
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

    cell_font_size = 16

    for row, run in enumerate(runs):
        points = run["points"]
        angle = run["result"]["angle"]
        dg_angle = run["dg_result"]["angle"]
        inlier_mask = run["inlier_mask"]
        outlier_mask = ~inlier_mask
        summary = run["summary"]
        dg_summary = run["dg_summary"]
        fraction = run["middle_fraction"]

        ax_clean = fig.add_subplot(grid[row, 0], projection="3d")
        ax_clean.scatter(
            points[inlier_mask, 0],
            points[inlier_mask, 1],
            points[inlier_mask, 2],
            c=truth_theta,
            s=7,
            cmap="hsv",
        )
        if np.any(outlier_mask):
            ax_clean.scatter(
                points[outlier_mask, 0],
                points[outlier_mask, 1],
                points[outlier_mask, 2],
                c="#202020",
                marker="x",
                s=10,
                linewidths=0.7,
            )
        # ax_clean.set_title(
        #     f"cylinder={clean_points.shape[0]}, middle added={int(np.count_nonzero(outlier_mask))}"
        # )
        zoom_cylinder_axes(ax_clean, radius_limit, half_height)
        ax_clean.set_xticks([])
        ax_clean.set_yticks([])
        ax_clean.set_zticks([])

        ax_xy = fig.add_subplot(grid[row, 1])
        ax_xy.scatter(
            points[inlier_mask, 0],
            points[inlier_mask, 1],
            c="#5f87c7",
            s=5,
            alpha=0.55,
            label="surface",
        )
        if np.any(outlier_mask):
            ax_xy.scatter(
                points[outlier_mask, 0],
                points[outlier_mask, 1],
                c="#202020",
                s=5,
                alpha=0.55,
                label="middle",
            )
        ax_xy.set_title(
            f"+{int(np.count_nonzero(outlier_mask))} middle points",
            fontsize=cell_font_size,
        )
        ax_xy.set_xlim(-radius_limit, radius_limit)
        ax_xy.set_ylim(-radius_limit, radius_limit)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.set_xticks([])
        ax_xy.set_yticks([])

        ax_points = fig.add_subplot(grid[row, 3], projection="3d")
        ax_points.scatter(
            points[inlier_mask, 0],
            points[inlier_mask, 1],
            points[inlier_mask, 2],
            c=angle[inlier_mask],
            s=7,
            cmap="hsv",
        )
        if np.any(outlier_mask):
            ax_points.scatter(
                points[outlier_mask, 0],
                points[outlier_mask, 1],
                points[outlier_mask, 2],
                c="#202020",
                marker="x",
                s=10,
                linewidths=0.7,
            )
        ax_points.set_title(
            f"alignment={summary['score']:.3f}",
            fontsize=cell_font_size,
        )
        zoom_cylinder_axes(ax_points, radius_limit, half_height)
        ax_points.set_xticks([])
        ax_points.set_yticks([])
        ax_points.set_zticks([])

        ax_dgm = fig.add_subplot(grid[row, 2])
        diagram = run["result"]["diagram"]
        selected = run["result"]["selected"]
        finite = diagram[np.isfinite(diagram[:, 1])]
        if finite.size:
            ax_dgm.scatter(finite[:, 0], finite[:, 1], s=12, c="#707070")
        ax_dgm.scatter(
            diagram[selected, 0],
            diagram[selected, 1],
            s=42,
            c="#d62728",
            label="decoded H1",
        )
        max_value = float(np.nanmax(diagram[np.isfinite(diagram)]))
        ax_dgm.plot([0.0, max_value], [0.0, max_value],
                    c="#b0b0b0", linewidth=1)
        ax_dgm.set_xlim(0.0, max_value)
        ax_dgm.set_ylim(0.0, max_value)
        ax_dgm.set_aspect("equal", adjustable="box")
        ax_dgm.set_xlabel("birth")
        ax_dgm.set_ylabel("death")

        ax_dg = fig.add_subplot(grid[row, 4], projection="3d")
        ax_dg.scatter(
            points[inlier_mask, 0],
            points[inlier_mask, 1],
            points[inlier_mask, 2],
            c=dg_angle[inlier_mask],
            s=7,
            cmap="hsv",
        )
        if np.any(outlier_mask):
            ax_dg.scatter(
                points[outlier_mask, 0],
                points[outlier_mask, 1],
                points[outlier_mask, 2],
                c="#202020",
                marker="x",
                s=10,
                linewidths=0.7,
            )
        ax_dg.set_title(
            f"alignment={dg_summary['score']:.3f}, ",
            fontsize=cell_font_size,
        )
        zoom_cylinder_axes(ax_dg, radius_limit, half_height)
        ax_dg.set_xticks([])
        ax_dg.set_yticks([])
        ax_dg.set_zticks([])

    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    clean_points, truth_theta, height_values = make_cylinder(
        args.n_points, args.radius, args.height, rng
    )
    middle_fractions = tuple(args.outlier_fractions)
    if len(middle_fractions) != 4:
        raise ValueError(
            "--outlier-fractions must contain exactly four values.")

    print(f"clean cylinder points: {args.n_points}")
    print(f"landmarks per ripser run: {min(args.n_landmarks, args.n_points)}")
    runs = []
    for fraction in middle_fractions:
        points, inlier_mask = add_middle_points(
            clean_points, fraction, args.radius, args.height, args.middle_radius, rng
        )
        result = gardner_circular_coordinate(points, args, rng)
        summary = summarize(result, truth_theta, height_values, inlier_mask)
        dg_result = diffusion_geometry_circular_coordinate(points, args)
        dg_summary = summarize(dg_result, truth_theta,
                               height_values, inlier_mask)
        runs.append(
            {
                "clean_points": clean_points,
                "truth_theta": truth_theta,
                "points": points,
                "inlier_mask": inlier_mask,
                "middle_fraction": fraction,
                "result": result,
                "summary": summary,
                "dg_result": dg_result,
                "dg_summary": dg_summary,
            }
        )
        print(
            f"middle points={fraction:.0%} ({summary['n_outliers']} points) | "
            f"H1 classes={result['n_h1']} | "
            f"decoded lifetime={result['lifetime']:.6f} | "
            f"PH alignment={summary['score']:.6f} | "
            f"PH error={summary['mean_error']:.6f} rad | "
            f"DG alignment={dg_summary['score']:.6f} | "
            f"DG error={dg_summary['mean_error']:.6f} rad"
        )

    if not args.no_plot:
        plot_runs(runs, args.output)
        print(f"saved plot: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=1600)
    parser.add_argument("--n-landmarks", type=int, default=500)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument("--coeff", type=int, default=47)
    parser.add_argument("--dec-thresh", type=float, default=0.99)
    parser.add_argument("--dg-n-function-basis", type=int, default=80)
    parser.add_argument("--dg-n-coefficients", type=int, default=40)
    parser.add_argument("--dg-knn-kernel", type=int, default=40)
    parser.add_argument("--dg-knn-bandwidth", type=int, default=16)
    parser.add_argument("--dg-epsilon", type=float, default=1.0)
    parser.add_argument("--dg-k", type=int, default=30)
    parser.add_argument("--dg-max-exact-ratio", type=float, default=0.8)
    parser.add_argument("--dg-min-coclosed-ratio", type=float, default=0.5)
    parser.add_argument("--dg-imag-tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--outlier-fractions",
        type=float,
        nargs=4,
        default=OUTLIER_FRACTIONS,
        help="Four interior middle-point fractions relative to the clean cylinder point count.",
    )
    parser.add_argument(
        "--middle-radius",
        type=float,
        default=0.95,
        help="Interior points are sampled in a solid cylinder with this fraction of the surface radius.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TDA/output/gardner_cylinder_outlier_coordinates.pdf"),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
