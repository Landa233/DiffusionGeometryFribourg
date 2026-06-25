"""Score decoded circular coordinates against Gardner open-field location.

This reads a saved ``*_decoding.npz`` file from either the persistent-homology
or Hodge runner and evaluates how smoothly each decoded angle varies over the
rat's physical ``(x, y)`` location.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TDA.gardner2022.physical_coordinate_scores import score_decoded_coordinates
from TDA.gardner2022.run_persistent_homology import (
    DEFAULT_DATA_DIR,
    DEFAULT_ORIGINAL_DIR,
    dataset_stem,
    load_original_utils,
    load_spikes,
    working_directory,
)


def default_decoding_path(args: argparse.Namespace) -> Path:
    stem = dataset_stem(args.rat, args.module, args.session, args.day)
    suffix = "_hodge_decoding.npz" if args.method == "hodge" else "_decoding.npz"
    return args.output_dir / f"{stem}{suffix}"


def run(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    original_dir = args.original_dir.resolve()
    decoding_path = args.decoding.resolve() if args.decoding else default_decoding_path(args)
    if not decoding_path.exists():
        raise FileNotFoundError(f"Decoding file not found: {decoding_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Extracted data directory not found: {data_dir}")

    os.environ.setdefault(
        "MPLCONFIGDIR", str(ROOT / "TDA" / "gardner2022" / "output" / ".matplotlib")
    )
    utils = load_original_utils(original_dir)
    with working_directory(original_dir):
        _, xx, yy, _, _ = load_spikes(
            utils,
            args.rat,
            args.module,
            args.day,
            "OF",
            data_dir,
            smoothing_width=args.sigma,
        )

    decoding = np.load(decoding_path)
    coordsbox = decoding["coordsbox"]
    times_box = decoding["times_box"]
    coordinate_scores = score_decoded_coordinates(
        coordsbox,
        xx,
        yy,
        times_box,
        physical_neighbors=args.physical_neighbors,
        stride=args.stride,
    )
    summary = {
        "decoding": str(decoding_path),
        "method": args.method,
        "rat": args.rat,
        "module": args.module,
        "session": args.session,
        "day": args.day,
        "physical_neighbors": args.physical_neighbors,
        "stride": args.stride,
        "mean_physical_smoothness": float(
            np.mean([score.physical_smoothness for score in coordinate_scores])
        ),
        "coordinates": [
            {
                "coordinate_index": score.coordinate_index,
                "physical_smoothness": score.physical_smoothness,
                "local_energy": score.local_energy,
                "circular_variance": score.circular_variance,
            }
            for score in coordinate_scores
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score decoded Gardner circular coordinates against OF location."
    )
    parser.add_argument("--method", choices=["hodge", "persistent"], default="hodge")
    parser.add_argument("--decoding", type=Path)
    parser.add_argument("--rat", default="R", choices=("R", "Q", "S"))
    parser.add_argument("--module", default="1")
    parser.add_argument("--session", default="OF")
    parser.add_argument("--day", default="day2")
    parser.add_argument("--sigma", type=int, default=1500)
    parser.add_argument("--physical-neighbors", type=int, default=12)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "TDA" / "gardner2022" / "output",
    )
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
