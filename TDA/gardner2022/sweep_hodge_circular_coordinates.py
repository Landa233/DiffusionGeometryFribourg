"""Run a small Gardner Hodge circular-coordinate parameter sweep.

The sweep intentionally stays small enough for interactive iteration. Each
attempt delegates to ``run_hodge_circular_coordinates.py``, which appends the
measured physical smoothness scores to ``hodge_circular_coordinate_attempts.md``.

Example
-------
python TDA/gardner2022/sweep_hodge_circular_coordinates.py --no-verify-archive
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_hodge_circular_coordinates.py")


def _csv_floats(value: str) -> list[float | None]:
    values: list[float | None] = []
    for item in value.split(","):
        item = item.strip()
        values.append(None if item in {"", "none", "None"} else float(item))
    return values


def _csv_ints(value: str) -> list[int | None]:
    values: list[int | None] = []
    for item in value.split(","):
        item = item.strip()
        values.append(None if item in {"", "none", "None"} else int(item))
    return values


def _append_log(lines: list[str]) -> None:
    log_path = Path(__file__).with_name("hodge_circular_coordinate_attempts.md")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run(args: argparse.Namespace) -> int:
    common = [
        sys.executable,
        str(RUNNER),
        "--rat",
        args.rat,
        "--module",
        args.module,
        "--session",
        args.session,
        "--day",
        args.day,
        "--k",
        str(args.k),
        "--max-pair-search",
        str(args.max_pair_search),
        "--physical-score-stride",
        str(args.physical_score_stride),
    ]
    if not args.verify_archive:
        common.append("--no-verify-archive")

    attempts = list(
        product(
            args.hodge_up_weights,
            args.bandwidths,
            args.embedding_dims,
            args.selections,
            args.function_bases,
        )
    )
    _append_log(
        [
            f"## Sweep request: {args.rat}_{args.module}_{args.session}_{args.day}",
            "",
            f"- attempts requested: {len(attempts)}",
            f"- hodge_up_weights: {args.hodge_up_weights}",
            f"- bandwidths: {args.bandwidths}",
            f"- embedding_dims: {args.embedding_dims}",
            f"- selections: {args.selections}",
            f"- function_bases: {args.function_bases}",
            "",
        ]
    )

    for index, (hodge_up_weight, bandwidth, embedding_dim, selection, function_basis) in enumerate(
        attempts, start=1
    ):
        command = [
            *common,
            "--hodge-up-weight",
            str(hodge_up_weight),
            "--selection",
            selection,
            "--function-basis",
            function_basis,
        ]
        if bandwidth is not None:
            command.extend(["--bandwidth", str(bandwidth)])
        if embedding_dim is not None:
            command.extend(["--embedding-dim", str(embedding_dim), "--standardize-embedding"])

        print(f"[{index}/{len(attempts)}] {' '.join(command)}")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            _append_log(
                [
                    "## Sweep stopped",
                    "",
                    f"- failed attempt: {index}/{len(attempts)}",
                    f"- return code: {completed.returncode}",
                    f"- command: `{' '.join(command)}`",
                    "",
                ]
            )
            return completed.returncode

    _append_log(["## Sweep complete", "", f"- attempts completed: {len(attempts)}", ""])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat", default="R", choices=("R", "Q", "S"))
    parser.add_argument("--module", default="1")
    parser.add_argument("--session", default="OF")
    parser.add_argument("--day", default="day2")
    parser.add_argument("--no-verify-archive", dest="verify_archive", action="store_false")
    parser.set_defaults(verify_archive=True)
    parser.add_argument("--hodge-up-weights", type=_csv_floats, default=[1.0, 0.5, 0.25])
    parser.add_argument("--bandwidths", type=_csv_floats, default=[None, 1.0, 1.5, 2.0])
    parser.add_argument("--embedding-dims", type=_csv_ints, default=[None, 3, 4, 6])
    parser.add_argument(
        "--selections",
        nargs="+",
        choices=("physical", "intrinsic", "reconstruction"),
        default=["physical"],
    )
    parser.add_argument(
        "--function-bases",
        nargs="+",
        choices=("diffusion", "monomial"),
        default=["diffusion"],
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max-pair-search", type=int, default=12)
    parser.add_argument("--physical-score-stride", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
