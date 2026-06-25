"""Compare Gardner persistent and Hodge decoded circular coordinates."""

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


def default_decoding_paths(args: argparse.Namespace) -> dict[str, Path]:
    stem = dataset_stem(args.rat, args.module, args.session, args.day)
    return {
        "persistent": args.output_dir / f"{stem}_decoding.npz",
        "hodge": args.output_dir / f"{stem}_hodge_decoding.npz",
    }


def score_decoding_file(
    path: Path,
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    physical_neighbors: int,
    stride: int,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Decoding file not found: {path}")

    decoding = np.load(path)
    coordinate_scores = score_decoded_coordinates(
        decoding["coordsbox"],
        xx,
        yy,
        decoding["times_box"],
        physical_neighbors=physical_neighbors,
        stride=stride,
    )
    values = [score.physical_smoothness for score in coordinate_scores]
    return {
        "decoding": str(path),
        "mean_physical_smoothness": float(np.mean(values)),
        "max_physical_smoothness": float(np.max(values)),
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


def build_comparison_summary(
    scores: dict[str, dict],
    *,
    rat: str,
    module: str,
    session: str,
    day: str,
    physical_neighbors: int,
    stride: int,
    baseline_tolerance: float = 0.0,
) -> dict:
    """Build the persistent-vs-Hodge comparison record.

    Lower physical smoothness is better. ``baseline_tolerance`` lets a sweep
    accept tiny numerical differences as a match while keeping the default
    comparison strict.
    """

    if "persistent" not in scores or "hodge" not in scores:
        raise ValueError("scores must contain 'persistent' and 'hodge' entries.")
    if baseline_tolerance < 0:
        raise ValueError("baseline_tolerance must be non-negative.")

    ranking = sorted(scores, key=lambda method: scores[method]["mean_physical_smoothness"])
    hodge_score = scores["hodge"]["mean_physical_smoothness"]
    persistent_score = scores["persistent"]["mean_physical_smoothness"]
    return {
        "rat": rat,
        "module": module,
        "session": session,
        "day": day,
        "physical_neighbors": physical_neighbors,
        "stride": stride,
        "baseline_tolerance": baseline_tolerance,
        "scores": scores,
        "ranking": ranking,
        "hodge_at_least_persistent": bool(
            hodge_score <= persistent_score + baseline_tolerance
        ),
        "mean_physical_smoothness_delta_hodge_minus_persistent": float(
            hodge_score - persistent_score
        ),
    }


def run(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    original_dir = args.original_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Extracted data directory not found: {data_dir}")

    paths = default_decoding_paths(args)
    if args.persistent_decoding is not None:
        paths["persistent"] = args.persistent_decoding.resolve()
    if args.hodge_decoding is not None:
        paths["hodge"] = args.hodge_decoding.resolve()

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

    scores = {
        method: score_decoding_file(
            path,
            xx=xx,
            yy=yy,
            physical_neighbors=args.physical_neighbors,
            stride=args.stride,
        )
        for method, path in paths.items()
    }
    summary = build_comparison_summary(
        scores,
        rat=args.rat,
        module=args.module,
        session=args.session,
        day=args.day,
        physical_neighbors=args.physical_neighbors,
        stride=args.stride,
        baseline_tolerance=args.baseline_tolerance,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare persistent and Hodge Gardner decoding files."
    )
    parser.add_argument("--rat", default="R", choices=("R", "Q", "S"))
    parser.add_argument("--module", default="1")
    parser.add_argument("--session", default="OF")
    parser.add_argument("--day", default="day2")
    parser.add_argument("--sigma", type=int, default=1500)
    parser.add_argument("--physical-neighbors", type=int, default=12)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--baseline-tolerance", type=float, default=0.0)
    parser.add_argument("--persistent-decoding", type=Path)
    parser.add_argument("--hodge-decoding", type=Path)
    parser.add_argument("--output-json", type=Path)
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
