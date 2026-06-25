"""Reproduce the Gardner et al. 2022 grid-cell persistent homology analysis.

This script is a small CLI wrapper around the original GridCellTorus notebook
pipeline. It keeps the upstream notebooks and helper functions unchanged under
``TDA/gardner2022_original`` while making the single-dataset analysis runnable
from the repository.

Example
-------
python TDA/gardner2022_persistent_homology.py --rat R --module 1 --session OF --day day2
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import warnings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "TDA" / "data" / "Toroidal_topology_grid_cell_data"
DEFAULT_ARCHIVE = ROOT / "TDA" / "data" / "Toroidal_topology_grid_cell_data.zip"
DEFAULT_ORIGINAL_DIR = ROOT / "TDA" / "gardner2022_original"
EXPECTED_MD5 = "379bfdca61cd54d5f58cab9d3ba477de"


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def file_md5(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def require_archive_checksum(archive: Path) -> None:
    if not archive.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {archive}. Download it from Figshare first."
        )
    observed = file_md5(archive)
    if observed != EXPECTED_MD5:
        raise RuntimeError(
            f"Unexpected dataset MD5 for {archive}: {observed}; "
            f"expected {EXPECTED_MD5}."
        )


def load_original_utils(original_dir: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the TDA dependencies first, e.g. `pip install .[tda]`.") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        needs_str_alias = not hasattr(np, "str")
    if needs_str_alias:
        np.str = str
    if not hasattr(np, "NaN"):
        np.NaN = np.nan

    utils_path = original_dir / "utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Original Gardner utils.py not found: {utils_path}")

    sys.path.insert(0, str(original_dir))
    spec = importlib.util.spec_from_file_location("gardner2022_original_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {utils_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Gardner TDA dependency. Install optional dependencies with "
            "`pip install .[tda]`."
        ) from exc
    finally:
        try:
            sys.path.remove(str(original_dir))
        except ValueError:
            pass
    return module


def dataset_stem(rat: str, module: str, session: str, day: str) -> str:
    stem = f"{rat}_{module}_{session}"
    if day:
        stem += f"_{day}"
    return stem


def load_spikes(
    utils,
    rat: str,
    module: str,
    day: str,
    session: str,
    data_dir: Path,
    smoothing_width=None,
):
    folder = f"{data_dir}/"
    kwargs = {}
    if smoothing_width is not None:
        kwargs["smoothing_width"] = smoothing_width
    if session in ("OF", "WW"):
        return utils.get_spikes(
            rat,
            module,
            day,
            session,
            bType="pure",
            bSmooth=True,
            bSpeed=True,
            folder=folder,
            **kwargs,
        )
    return utils.get_spikes(
        rat,
        module,
        day,
        session,
        bType="pure",
        bSmooth=True,
        bSpeed=False,
        folder=folder,
        **kwargs,
    )


def load_unsmoothed_spikes(utils, rat: str, module: str, day: str, session: str, data_dir: Path):
    folder = f"{data_dir}/"
    if session in ("OF", "WW"):
        return utils.get_spikes(
            rat,
            module,
            day,
            session,
            bType="pure",
            bSmooth=False,
            bSpeed=True,
            folder=folder,
        )
    return utils.get_spikes(
        rat,
        module,
        day,
        session,
        bType="pure",
        bSmooth=False,
        bSpeed=False,
        folder=folder,
    )


def compute_persistence(
    utils,
    sspikes,
    *,
    dim: int,
    active_times: int,
    num_times: int,
    denoise_k: int,
    n_points: int,
    metric: str,
    nbs: int,
    maxdim: int,
    coeff: int,
):
    np = utils.np
    preprocessing = utils.preprocessing
    pdist = utils.pdist
    squareform = utils.squareform
    coo_matrix = utils.coo_matrix
    ripser = utils.ripser

    times_cube = np.arange(0, len(sspikes[:, 0]), num_times)
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]

    dim_red_spikes_move_scaled, _, _ = utils.pca(
        preprocessing.scale(sspikes[movetimes, :]), dim=dim
    )
    indstemp, _, _ = utils.sample_denoising(
        dim_red_spikes_move_scaled, denoise_k, n_points, 1, metric
    )
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp, :]

    xdist = squareform(pdist(dim_red_spikes_move_scaled, metric))
    knn_indices = np.argsort(xdist)[:, :nbs]
    knn_dists = xdist[np.arange(xdist.shape[0])[:, None], knn_indices].copy()
    sigmas, rhos = utils.smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    rows, cols, vals = utils.compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )
    result = coo_matrix((vals, (rows, cols)), shape=(xdist.shape[0], xdist.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = result + transpose - prod_matrix
    result.eliminate_zeros()

    with np.errstate(divide="ignore"):
        distances = -np.log(result.toarray())
    np.fill_diagonal(distances, 0)

    persistence = ripser(
        distances,
        maxdim=maxdim,
        coeff=coeff,
        do_cocycles=True,
        distance_matrix=True,
    )
    return persistence, indstemp, movetimes


def decode_circular_coordinates(
    utils,
    persistence,
    indstemp,
    movetimes,
    sspikes,
    *,
    ph_classes: tuple[int, ...],
    dec_thresh: float,
    coeff: int,
    n_points: int,
):
    np = utils.np
    preprocessing = utils.preprocessing

    diagrams = persistence["dgms"]
    cocycles = persistence["cocycles"][1]
    dists_land = persistence["dperm2all"]
    births1 = diagrams[1][:, 0]
    deaths1 = diagrams[1][:, 1].copy()
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1 - births1
    i_max = np.argsort(lives1)
    if len(i_max) < max(ph_classes) + 1:
        raise RuntimeError("Not enough H1 cocycles were found to decode the requested classes.")

    coords1 = np.zeros((len(ph_classes), len(indstemp)))
    threshold = births1[i_max[-2]] + (deaths1[i_max[-2]] - births1[i_max[-2]]) * dec_thresh
    for out_index, class_index in enumerate(ph_classes):
        cocycle = cocycles[i_max[-(class_index + 1)]]
        coords1[out_index, :], _ = utils.get_coords(
            cocycle.copy(), threshold, len(indstemp), dists_land, coeff
        )

    num_neurons = len(sspikes[0, :])
    centcosall = np.zeros((num_neurons, len(ph_classes), n_points))
    centsinall = np.zeros((num_neurons, len(ph_classes), n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp], :])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid, :, :] = np.multiply(np.cos(coords1 * 2 * np.pi), spktemp)
        centsinall[neurid, :, :] = np.multiply(np.sin(coords1 * 2 * np.pi), spktemp)
    return centcosall, centsinall


def project_coordinates(utils, sspikes, spikes, centcosall, centsinall):
    np = utils.np
    preprocessing = utils.preprocessing

    times = np.where(np.sum(spikes > 0, 1) >= 1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times, :]
    dspk = dspk[times, :]
    num_neurons = centcosall.shape[0]
    num_circ = centcosall.shape[1]

    a = np.zeros((len(sspikes[:, 0]), num_circ, num_neurons))
    c = np.zeros((len(sspikes[:, 0]), num_circ, num_neurons))
    for n in range(num_neurons):
        a[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centcosall[n, :, :], 1))
        c[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centsinall[n, :, :], 1))

    mtot2 = np.sum(c, 2)
    mtot1 = np.sum(a, 2)
    coords = np.arctan2(mtot2, mtot1) % (2 * np.pi)
    return coords, times


def plot_physical_coordinates(utils, xx, yy, times_box, coordsbox, figure_path: Path) -> None:
    np = utils.np
    plt = utils.plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_times = np.arange(0, len(times_box), 10)
    plt.viridis()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    ax[0].axis("off")
    ax[0].scatter(
        xx[times_box][plot_times],
        yy[times_box][plot_times],
        c=np.cos(coordsbox[plot_times, 0]),
        s=8,
    )
    ax[0].set_aspect("equal", "box")
    ax[1].axis("off")
    ax[1].scatter(
        xx[times_box][plot_times],
        yy[times_box][plot_times],
        c=np.cos(coordsbox[plot_times, 1]),
        s=8,
    )
    ax[1].set_aspect("equal", "box")
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "Results").resolve()
    figure_dir = args.figure_dir.resolve()
    original_dir = args.original_dir.resolve()

    if args.verify_archive:
        require_archive_checksum(args.archive.resolve())
    if not data_dir.exists():
        raise FileNotFoundError(f"Extracted data directory not found: {data_dir}")

    utils = load_original_utils(original_dir)
    np = utils.np
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    folder = f"{data_dir}/"
    stem = dataset_stem(args.rat, args.module, args.session, args.day)

    with working_directory(original_dir):
        if args.session in ("OF", "WW"):
            sspikes, xx, yy, _, _ = load_spikes(
                utils, args.rat, args.module, args.day, args.session, data_dir
            )
        else:
            sspikes = load_spikes(
                utils, args.rat, args.module, args.day, args.session, data_dir
            )
            xx = yy = None

        persistence, indstemp, movetimes = compute_persistence(
            utils,
            sspikes,
            dim=args.dim,
            active_times=args.active_times,
            num_times=args.num_times,
            denoise_k=args.denoise_k,
            n_points=args.n_points,
            metric=args.metric,
            nbs=args.nbs,
            maxdim=args.maxdim,
            coeff=args.coeff,
        )

        persistence_path = output_dir / f"{stem}_persistence.npz"
        np.savez_compressed(
            persistence_path,
            persistence=persistence,
            indstemp=indstemp,
            movetimes=movetimes,
        )

        centcosall, centsinall = decode_circular_coordinates(
            utils,
            persistence,
            indstemp,
            movetimes,
            sspikes,
            ph_classes=tuple(args.ph_classes),
            dec_thresh=args.dec_thresh,
            coeff=args.coeff,
            n_points=args.n_points,
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
        raw = load_unsmoothed_spikes(utils, args.rat, args.module, args.day, args.session, data_dir)
        if args.session in ("OF", "WW"):
            sspikes_decode, xx, yy, _, _ = smoothed
            spikes_decode, _, _, _, _ = raw
        else:
            sspikes_decode = smoothed
            spikes_decode = raw
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

        decoding_path = output_dir / f"{stem}_decoding.npz"
        np.savez_compressed(
            decoding_path,
            coords=coords,
            coordsbox=coordsbox,
            times=times,
            times_box=times_box,
            centcosall=centcosall,
            centsinall=centsinall,
        )

        figure_path = figure_dir / f"{stem}_physical_location_by_torus_coordinates.png"
        plot_physical_coordinates(utils, xx, yy, times_box, coordsbox, figure_path)

        if args.barcode:
            utils.plot_barcode(persistence["dgms"])
            barcode_path = figure_dir / f"{stem}_barcode.png"
            utils.plt.savefig(barcode_path, dpi=200)
            utils.plt.close("all")
        else:
            barcode_path = None

    h1_count = len(persistence["dgms"][1]) if len(persistence["dgms"]) > 1 else 0
    print(f"Saved persistence: {persistence_path}")
    print(f"Saved decoding: {decoding_path}")
    print(f"Saved physical-coordinate figure: {figure_path}")
    if barcode_path is not None:
        print(f"Saved barcode: {barcode_path}")
    print(f"H1 classes: {h1_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Gardner et al. 2022 grid-cell torus PH analysis."
    )
    parser.add_argument("--rat", default="R", choices=("R", "Q", "S"))
    parser.add_argument("--module", default="1")
    parser.add_argument("--session", default="OF")
    parser.add_argument("--day", default="day2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "TDA" / "output" / "gardner2022")
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--no-verify-archive", dest="verify_archive", action="store_false")
    parser.set_defaults(verify_archive=True)

    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--active-times", type=int, default=15000)
    parser.add_argument("--denoise-k", type=int, default=1000)
    parser.add_argument("--n-points", type=int, default=1200)
    parser.add_argument("--nbs", type=int, default=800)
    parser.add_argument("--num-times", type=int, default=5)
    parser.add_argument("--sigma", type=int, default=1500)
    parser.add_argument("--coeff", type=int, default=47)
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument("--ph-classes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--dec-thresh", type=float, default=0.99)
    parser.add_argument("--barcode", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
