"""Run repo-native Hodge circular coordinates on Gardner et al. grid-cell data.

This follows the Gardner et al. preprocessing path up to the denoised PCA
landmarks, then replaces ripser cocycles with ``methods.circular_coordinates``.
The selected circular coordinates are decoded back onto the open-field
trajectory using the same population-vector projection used by the Gardner
notebook, and the physical location is coloured by the two learned angles.

Example
-------
python TDA/gardner2022/run_hodge_circular_coordinates.py --rat R --module 1 --session OF --day day2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_geometry import DiffusionGeometry
from TDA.gardner2022.physical_coordinate_scores import score_decoded_coordinates
from TDA.synthetic.run_circular_coordinates import monomial_function_basis
from methods.circular_coordinates import circular_coordinates
from TDA.gardner2022.run_persistent_homology import (
    DEFAULT_ARCHIVE,
    DEFAULT_DATA_DIR,
    DEFAULT_ORIGINAL_DIR,
    dataset_stem,
    load_original_utils,
    load_spikes,
    load_unsmoothed_spikes,
    plot_physical_coordinates,
    project_coordinates,
    require_archive_checksum,
    working_directory,
)


def preprocess_landmarks(
    utils,
    sspikes: np.ndarray,
    *,
    dim: int,
    active_times: int,
    num_times: int,
    denoise_k: int,
    n_points: int,
    metric: str,
):
    preprocessing = utils.preprocessing

    times_cube = np.arange(0, len(sspikes[:, 0]), num_times)
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]

    dim_red_spikes_move_scaled, _, _ = utils.pca(
        preprocessing.scale(sspikes[movetimes, :]), dim=dim
    )
    indstemp, _, _ = utils.sample_denoising(
        dim_red_spikes_move_scaled, denoise_k, n_points, 1, metric
    )
    landmarks = dim_red_spikes_move_scaled[indstemp, :]
    return landmarks, indstemp, movetimes


def angle_pair_correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(1j * first) * np.conjugate(np.exp(1j * second)))))


def two_best_candidates(result, max_pair_search: int, *, criterion: str = "intrinsic"):
    if len(result.candidates) < 2:
        raise RuntimeError("Need at least two circular-coordinate candidates.")
    if criterion not in {"intrinsic", "reconstruction"}:
        raise ValueError(f"Unknown intrinsic candidate criterion: {criterion}.")

    pool = result.candidates[: min(max_pair_search, len(result.candidates))]
    best_pair = None
    best_score = np.inf
    for i, first in enumerate(pool):
        for second in pool[i + 1 :]:
            correlation = angle_pair_correlation(first.angle, second.angle)
            if criterion == "intrinsic":
                score = (
                    first.dirichlet_energy
                    + second.dirichlet_energy
                    + correlation * np.sqrt(first.dirichlet_energy * second.dirichlet_energy)
                )
            else:
                score = first.reconstruction_error + second.reconstruction_error + correlation
            if score < best_score:
                best_score = score
                best_pair = (first, second)
    return best_pair


def score_decoded_candidates(
    coordsbox: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    times_box: np.ndarray,
    *,
    physical_neighbors: int,
    physical_score_stride: int,
) -> list:
    return score_decoded_coordinates(
        coordsbox,
        xx,
        yy,
        times_box,
        physical_neighbors=physical_neighbors,
        stride=physical_score_stride,
    )


def two_best_physical_candidates(result, scores: list):
    if len(scores) < 2:
        raise RuntimeError("Need at least two scored circular-coordinate candidates.")
    score_by_candidate = {score.coordinate_index: score for score in scores}
    best_pair = None
    best_positions = None
    best_score = np.inf
    for i, first in enumerate(result.candidates):
        if i not in score_by_candidate:
            continue
        first_score = score_by_candidate[i]
        for j, second in enumerate(result.candidates[i + 1 :], start=i + 1):
            if j not in score_by_candidate:
                continue
            second_score = score_by_candidate[j]
            redundancy = angle_pair_correlation(
                first_score.decoded_angle,
                second_score.decoded_angle,
            )
            variance_penalty = (
                max(0.05 - first_score.circular_variance, 0.0)
                + max(0.05 - second_score.circular_variance, 0.0)
            )
            score = (
                first_score.physical_smoothness
                + second_score.physical_smoothness
                + redundancy
                + 10.0 * variance_penalty
            )
            if score < best_score:
                best_score = score
                best_pair = (first, second)
                best_positions = (i, j)
    if best_pair is None:
        raise RuntimeError("No candidate pair could be scored against physical location.")
    return best_pair, best_positions, best_score


def decode_candidate_angles(utils, sspikes, movetimes, indstemp, candidates):
    preprocessing = utils.preprocessing

    angles = np.vstack([candidate.angle for candidate in candidates])
    num_neurons = len(sspikes[0, :])
    num_circ, n_points = angles.shape
    centcosall = np.zeros((num_neurons, num_circ, n_points))
    centsinall = np.zeros((num_neurons, num_circ, n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp], :])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid, :, :] = np.multiply(np.cos(angles), spktemp)
        centsinall[neurid, :, :] = np.multiply(np.sin(angles), spktemp)
    return angles, centcosall, centsinall


def save_hodge_output(
    path: Path,
    result,
    candidates,
    landmarks,
    indstemp,
    movetimes,
    angles,
    *,
    selection_method: str,
    physical_scores: list | None = None,
    selected_physical_score: float | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        landmarks=landmarks,
        indstemp=indstemp,
        movetimes=movetimes,
        selected_angles=angles,
        selected_candidate_indices=np.array([candidate.index for candidate in candidates]),
        selected_hodge_eigenvalues=np.array(
            [candidate.hodge_eigenvalue for candidate in candidates]
        ),
        selected_reconstruction_errors=np.array(
            [candidate.reconstruction_error for candidate in candidates]
        ),
        selected_similarities=np.array([candidate.similarity for candidate in candidates]),
        selected_dirichlet_energies=np.array(
            [candidate.dirichlet_energy for candidate in candidates]
        ),
        selection_method=selection_method,
        selected_physical_score=np.nan
        if selected_physical_score is None
        else selected_physical_score,
        all_candidate_indices=np.array([candidate.index for candidate in result.candidates]),
        all_reconstruction_errors=np.array(
            [candidate.reconstruction_error for candidate in result.candidates]
        ),
        all_similarities=np.array([candidate.similarity for candidate in result.candidates]),
        all_dirichlet_energies=np.array(
            [candidate.dirichlet_energy for candidate in result.candidates]
        ),
        physical_candidate_positions=np.array(
            [] if physical_scores is None else [score.coordinate_index for score in physical_scores]
        ),
        physical_smoothness_scores=np.array(
            [] if physical_scores is None else [score.physical_smoothness for score in physical_scores]
        ),
        physical_circular_variances=np.array(
            [] if physical_scores is None else [score.circular_variance for score in physical_scores]
        ),
        physical_local_energies=np.array(
            [] if physical_scores is None else [score.local_energy for score in physical_scores]
        ),
    )


def append_attempt_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    original_dir = args.original_dir.resolve()

    if args.verify_archive:
        require_archive_checksum(args.archive.resolve())
    if not data_dir.exists():
        raise FileNotFoundError(f"Extracted data directory not found: {data_dir}")

    os.environ.setdefault(
        "MPLCONFIGDIR", str(ROOT / "TDA" / "gardner2022" / "output" / ".matplotlib")
    )
    utils = load_original_utils(original_dir)
    stem = dataset_stem(args.rat, args.module, args.session, args.day)

    with working_directory(original_dir):
        if args.session in ("OF", "WW"):
            sspikes_ph, _, _, _, _ = load_spikes(
                utils, args.rat, args.module, args.day, args.session, data_dir
            )
        else:
            sspikes_ph = load_spikes(
                utils, args.rat, args.module, args.day, args.session, data_dir
            )

        landmarks, indstemp, movetimes = preprocess_landmarks(
            utils,
            sspikes_ph,
            dim=args.dim,
            active_times=args.active_times,
            num_times=args.num_times,
            denoise_k=args.denoise_k,
            n_points=args.n_points,
            metric=args.metric,
        )
        function_basis = None
        if args.function_basis == "monomial":
            function_basis = monomial_function_basis(landmarks, args.monomial_degree)

        immersion_coords = landmarks
        if args.embedding_dim is not None:
            if args.embedding_dim <= 0 or args.embedding_dim > landmarks.shape[1]:
                raise ValueError(
                    f"embedding_dim must be between 1 and {landmarks.shape[1]}."
                )
            immersion_coords = landmarks[:, : args.embedding_dim]
        if args.standardize_embedding:
            coord_scale = np.std(immersion_coords, axis=0)
            coord_scale[coord_scale == 0.0] = 1.0
            immersion_coords = (immersion_coords - np.mean(immersion_coords, axis=0)) / coord_scale

        dg = DiffusionGeometry.from_point_cloud(
            landmarks,
            immersion_coords=immersion_coords,
            function_basis=function_basis,
            n_function_basis=args.n_function_basis,
            n_coefficients=args.n_coefficients,
            knn_kernel=args.knn_kernel,
            knn_bandwidth=args.knn_bandwidth,
            bandwidth=args.bandwidth,
            regularisation_method=args.regularisation_method,
            rcond=args.rcond,
        )
        result = circular_coordinates(
            dg,
            epsilon=args.epsilon,
            k=args.k,
            max_exact_ratio=args.max_exact_ratio,
            min_coclosed_ratio=args.min_coclosed_ratio,
            hodge_up_weight=args.hodge_up_weight,
            imag_tol=args.imag_tol,
        )
        smoothed = load_spikes(
            utils,
            args.rat,
            args.module,
            args.day,
            args.session,
            data_dir,
            smoothing_width=args.sigma,
        )
        raw = load_unsmoothed_spikes(
            utils, args.rat, args.module, args.day, args.session, data_dir
        )
        if args.session in ("OF", "WW"):
            sspikes_decode, xx, yy, _, _ = smoothed
            spikes_decode, _, _, _, _ = raw
        else:
            sspikes_decode = smoothed
            spikes_decode = raw
            xx = yy = None
        physical_scores = []
        physical_pair_score = None
        selected_pool_positions = None
        if args.selection == "physical":
            pool = result.candidates[: min(args.max_pair_search, len(result.candidates))]
            _, pool_centcosall, pool_centsinall = decode_candidate_angles(
                utils, sspikes_ph, movetimes, indstemp, pool
            )
            if args.session == "OF":
                pool_coordsbox, pool_times_box = project_coordinates(
                    utils, sspikes_decode, spikes_decode, pool_centcosall, pool_centsinall
                )
                if xx is None or yy is None:
                    _, xx, yy, _, _ = load_spikes(
                        utils,
                        args.rat,
                        args.module,
                        args.day,
                        "OF",
                        data_dir,
                        smoothing_width=args.sigma,
                    )
            else:
                folder = f"{data_dir}/"
                sspikes_of, xx, yy, _, _ = utils.get_spikes(
                    args.rat,
                    args.module,
                    args.day,
                    "OF",
                    bType="pure",
                    bSmooth=True,
                    bSpeed=True,
                    smoothing_width=args.sigma,
                    folder=folder,
                )
                spikes_of, _, _, _, _ = utils.get_spikes(
                    args.rat,
                    args.module,
                    args.day,
                    "OF",
                    bType="pure",
                    bSmooth=False,
                    bSpeed=True,
                    folder=folder,
                )
                pool_coordsbox, pool_times_box = project_coordinates(
                    utils, sspikes_of, spikes_of, pool_centcosall, pool_centsinall
                )
            physical_scores = score_decoded_candidates(
                pool_coordsbox,
                xx,
                yy,
                pool_times_box,
                physical_neighbors=args.physical_neighbors,
                physical_score_stride=args.physical_score_stride,
            )
            candidates, selected_pool_positions, physical_pair_score = two_best_physical_candidates(
                result, physical_scores
            )
        else:
            candidates = two_best_candidates(
                result, args.max_pair_search, criterion=args.selection
            )

        angles, centcosall, centsinall = decode_candidate_angles(
            utils, sspikes_ph, movetimes, indstemp, candidates
        )
        coords, times = project_coordinates(
            utils, sspikes_decode, spikes_decode, centcosall, centsinall
        )

        if args.session == "OF":
            coordsbox = coords.copy()
            times_box = times.copy()
            if xx is None or yy is None:
                _, xx, yy, _, _ = load_spikes(
                    utils,
                    args.rat,
                    args.module,
                    args.day,
                    "OF",
                    data_dir,
                    smoothing_width=args.sigma,
                )
        else:
            folder = f"{data_dir}/"
            sspikes_of, xx, yy, _, _ = utils.get_spikes(
                args.rat,
                args.module,
                args.day,
                "OF",
                bType="pure",
                bSmooth=True,
                bSpeed=True,
                smoothing_width=args.sigma,
                folder=folder,
            )
            spikes_of, _, _, _, _ = utils.get_spikes(
                args.rat,
                args.module,
                args.day,
                "OF",
                bType="pure",
                bSmooth=False,
                bSpeed=True,
                folder=folder,
            )
            coordsbox, times_box = project_coordinates(
                utils, sspikes_of, spikes_of, centcosall, centsinall
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        hodge_path = output_dir / f"{stem}_hodge_circular_coordinates.npz"
        decoding_path = output_dir / f"{stem}_hodge_decoding.npz"
        figure_path = figure_dir / f"{stem}_hodge_physical_location_by_coordinates.png"

        plot_physical_coordinates(utils, xx, yy, times_box, coordsbox, figure_path)
        selected_physical_scores = score_decoded_candidates(
            coordsbox,
            xx,
            yy,
            times_box,
            physical_neighbors=args.physical_neighbors,
            physical_score_stride=args.physical_score_stride,
        )
        save_hodge_output(
            hodge_path,
            result,
            candidates,
            landmarks,
            indstemp,
            movetimes,
            angles,
            selection_method=args.selection,
            physical_scores=physical_scores,
            selected_physical_score=physical_pair_score,
        )
        np.savez_compressed(
            decoding_path,
            coords=coords,
            coordsbox=coordsbox,
            times=times,
            times_box=times_box,
            centcosall=centcosall,
            centsinall=centsinall,
            selected_physical_smoothness=np.array(
                [score.physical_smoothness for score in selected_physical_scores]
            ),
            selected_physical_circular_variance=np.array(
                [score.circular_variance for score in selected_physical_scores]
            ),
            selected_physical_local_energy=np.array(
                [score.local_energy for score in selected_physical_scores]
            ),
        )

        attempt_log = Path(__file__).with_name("hodge_circular_coordinate_attempts.md")
        selected_text = ", ".join(
            f"{candidate.index} (recon={candidate.reconstruction_error:.4g}, "
            f"dirichlet={candidate.dirichlet_energy:.4g})"
            for candidate in candidates
        )
        physical_text = ", ".join(
            f"coord{score.coordinate_index}: smooth={score.physical_smoothness:.4g}, "
            f"circ_var={score.circular_variance:.4g}"
            for score in selected_physical_scores
        )
        pool_text = ""
        if physical_scores:
            top_scores = sorted(physical_scores, key=lambda item: item.physical_smoothness)[:5]
            pool_text = "\n  - best pool physical scores: " + ", ".join(
                f"{item.coordinate_index}:{item.physical_smoothness:.4g}"
                for item in top_scores
            )
        append_attempt_log(
            attempt_log,
            [
                f"## {stem} Hodge attempt",
                "",
                f"- selection: `{args.selection}`",
                f"- selected candidates: {selected_text}",
                f"- selected decoded physical scores: {physical_text}",
                f"- physical pair score: {physical_pair_score}",
                f"- selected pool positions: {selected_pool_positions}",
                f"- geometry: dim={args.dim}, embedding_dim={args.embedding_dim}, "
                f"function_basis={args.function_basis}, monomial_degree={args.monomial_degree}",
                f"- kernel: knn_kernel={args.knn_kernel}, knn_bandwidth={args.knn_bandwidth}, "
                f"bandwidth={args.bandwidth}, regularisation={args.regularisation_method}",
                f"- hodge: n_function_basis={args.n_function_basis}, "
                f"n_coefficients={args.n_coefficients}, hodge_up_weight={args.hodge_up_weight}, "
                f"epsilon={args.epsilon}, k={args.k}",
                f"- outputs: `{hodge_path.name}`, `{decoding_path.name}`, `{figure_path.name}`"
                + pool_text,
                "",
            ],
        )

    print(f"Saved Hodge circular-coordinate diagnostics: {hodge_path}")
    print(f"Saved Hodge decoding: {decoding_path}")
    print(f"Saved physical-coordinate figure: {figure_path}")
    print(
        "Selected candidates: "
        + ", ".join(
            f"{candidate.index} "
            f"(error={candidate.reconstruction_error:.4g}, "
            f"similarity={candidate.similarity:.4g})"
            for candidate in candidates
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use methods.circular_coordinates on Gardner et al. grid-cell "
            "landmarks and plot physical location coloured by decoded angles."
        )
    )
    parser.add_argument("--rat", default="R", choices=("R", "Q", "S"))
    parser.add_argument("--module", default="1")
    parser.add_argument("--session", default="OF")
    parser.add_argument("--day", default="day2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TDA" / "gardner2022" / "output")
    parser.add_argument(
        "--figure-dir", type=Path, default=ROOT / "TDA" / "gardner2022" / "output"
    )
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--no-verify-archive", dest="verify_archive", action="store_false")
    parser.set_defaults(verify_archive=True)

    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--active-times", type=int, default=15000)
    parser.add_argument("--denoise-k", type=int, default=1000)
    parser.add_argument("--n-points", type=int, default=1200)
    parser.add_argument("--num-times", type=int, default=5)
    parser.add_argument("--sigma", type=int, default=1500)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--standardize-embedding", action="store_true")
    parser.add_argument(
        "--function-basis",
        choices=["diffusion", "monomial"],
        default="diffusion",
    )
    parser.add_argument("--monomial-degree", type=int, default=2)

    parser.add_argument("--n-function-basis", type=int, default=80)
    parser.add_argument("--n-coefficients", type=int, default=40)
    parser.add_argument("--knn-kernel", type=int, default=80)
    parser.add_argument("--knn-bandwidth", type=int, default=20)
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--regularisation-method", default="diffusion")
    parser.add_argument("--rcond", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max-exact-ratio", type=float, default=0.8)
    parser.add_argument("--min-coclosed-ratio", type=float, default=0.5)
    parser.add_argument("--hodge-up-weight", type=float, default=1.0)
    parser.add_argument("--imag-tol", type=float, default=1e-8)
    parser.add_argument("--max-pair-search", type=int, default=12)
    parser.add_argument(
        "--selection",
        choices=["physical", "intrinsic", "reconstruction"],
        default="physical",
    )
    parser.add_argument("--physical-neighbors", type=int, default=12)
    parser.add_argument("--physical-score-stride", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
