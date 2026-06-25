"""Run repo-native Hodge circular coordinates on Gardner et al. grid-cell data.

This follows the Gardner et al. preprocessing path up to the denoised PCA
landmarks, then replaces ripser cocycles with ``methods.circular_coordinates``.
The selected circular coordinates are decoded back onto the open-field
trajectory using the same population-vector projection used by the Gardner
notebook, and the physical location is coloured by the two learned angles.

Example
-------
python TDA/gardner2022_hodge_circular_coordinates.py --rat R --module 1 --session OF --day day2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_geometry import DiffusionGeometry
from methods.circular_coordinates import circular_coordinates
from TDA.gardner2022_persistent_homology import (
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


def two_best_candidates(result, max_pair_search: int):
    if len(result.candidates) < 2:
        raise RuntimeError("Need at least two circular-coordinate candidates.")

    pool = result.candidates[: min(max_pair_search, len(result.candidates))]
    best_pair = None
    best_score = np.inf
    for i, first in enumerate(pool):
        for second in pool[i + 1 :]:
            correlation = angle_pair_correlation(first.angle, second.angle)
            score = first.reconstruction_error + second.reconstruction_error + correlation
            if score < best_score:
                best_score = score
                best_pair = (first, second)
    return best_pair


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


def save_hodge_output(path: Path, result, candidates, landmarks, indstemp, movetimes, angles):
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
        all_candidate_indices=np.array([candidate.index for candidate in result.candidates]),
        all_reconstruction_errors=np.array(
            [candidate.reconstruction_error for candidate in result.candidates]
        ),
        all_similarities=np.array([candidate.similarity for candidate in result.candidates]),
    )


def run(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "Results").resolve()
    figure_dir = args.figure_dir.resolve()
    original_dir = args.original_dir.resolve()

    if args.verify_archive:
        require_archive_checksum(args.archive.resolve())
    if not data_dir.exists():
        raise FileNotFoundError(f"Extracted data directory not found: {data_dir}")

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "TDA" / "output" / ".matplotlib"))
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

        dg = DiffusionGeometry.from_point_cloud(
            landmarks,
            n_function_basis=args.n_function_basis,
            n_coefficients=args.n_coefficients,
            knn_kernel=args.knn_kernel,
            knn_bandwidth=args.knn_bandwidth,
            regularisation_method=args.regularisation_method,
            rcond=args.rcond,
        )
        result = circular_coordinates(
            dg,
            epsilon=args.epsilon,
            k=args.k,
            max_exact_ratio=args.max_exact_ratio,
            min_coclosed_ratio=args.min_coclosed_ratio,
            imag_tol=args.imag_tol,
        )
        candidates = two_best_candidates(result, args.max_pair_search)

        angles, centcosall, centsinall = decode_candidate_angles(
            utils, sspikes_ph, movetimes, indstemp, candidates
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

        save_hodge_output(hodge_path, result, candidates, landmarks, indstemp, movetimes, angles)
        np.savez_compressed(
            decoding_path,
            coords=coords,
            coordsbox=coordsbox,
            times=times,
            times_box=times_box,
            centcosall=centcosall,
            centsinall=centsinall,
        )
        plot_physical_coordinates(utils, xx, yy, times_box, coordsbox, figure_path)

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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--figure-dir", type=Path, default=ROOT / "TDA" / "output" / "gardner2022"
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

    parser.add_argument("--n-function-basis", type=int, default=80)
    parser.add_argument("--n-coefficients", type=int, default=40)
    parser.add_argument("--knn-kernel", type=int, default=80)
    parser.add_argument("--knn-bandwidth", type=int, default=20)
    parser.add_argument("--regularisation-method", default="diffusion")
    parser.add_argument("--rcond", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max-exact-ratio", type=float, default=0.8)
    parser.add_argument("--min-coclosed-ratio", type=float, default=0.5)
    parser.add_argument("--imag-tol", type=float, default=1e-8)
    parser.add_argument("--max-pair-search", type=int, default=12)
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
