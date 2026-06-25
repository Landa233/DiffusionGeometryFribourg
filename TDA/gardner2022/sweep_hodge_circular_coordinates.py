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
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_hodge_circular_coordinates.py")
PERSISTENT_RUNNER = Path(__file__).with_name("run_persistent_homology.py")
SCORER = Path(__file__).with_name("score_decoding.py")


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


def _score_method(args: argparse.Namespace, method: str) -> dict | None:
    command = [
        sys.executable,
        str(SCORER),
        "--method",
        method,
        "--rat",
        args.rat,
        "--module",
        args.module,
        "--session",
        args.session,
        "--day",
        args.day,
        "--physical-neighbors",
        str(args.physical_neighbors),
        "--stride",
        str(args.physical_score_stride),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return json.loads(completed.stdout)


def _run_persistent_baseline(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(PERSISTENT_RUNNER),
        "--rat",
        args.rat,
        "--module",
        args.module,
        "--session",
        args.session,
        "--day",
        args.day,
    ]
    if not args.verify_archive:
        command.append("--no-verify-archive")
    print(f"[baseline] {' '.join(command)}")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


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
        "--physical-neighbors",
        str(args.physical_neighbors),
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
            f"- physical_neighbors: {args.physical_neighbors}",
            f"- physical_score_stride: {args.physical_score_stride}",
            f"- hodge_up_weights: {args.hodge_up_weights}",
            f"- bandwidths: {args.bandwidths}",
            f"- embedding_dims: {args.embedding_dims}",
            f"- selections: {args.selections}",
            f"- function_bases: {args.function_bases}",
            "",
        ]
    )

    if args.run_persistent_baseline:
        baseline_returncode = _run_persistent_baseline(args)
        if baseline_returncode != 0:
            _append_log(
                [
                    "## Sweep stopped",
                    "",
                    f"- persistent baseline failed with return code: {baseline_returncode}",
                    "",
                ]
            )
            return baseline_returncode

    persistent_score = _score_method(args, "persistent")
    persistent_mean = None
    if persistent_score is None:
        _append_log(
            [
                "## Sweep baseline",
                "",
                "- persistent baseline score unavailable; Hodge attempts will run without equal-or-better comparison.",
                "",
            ]
        )
    else:
        persistent_mean = persistent_score["mean_physical_smoothness"]
        _append_log(
            [
                "## Sweep baseline",
                "",
                f"- persistent mean_physical_smoothness: {persistent_mean:.6g}",
                f"- decoding: `{Path(persistent_score['decoding']).name}`",
                "",
            ]
        )

    best_hodge = None
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
        hodge_score = _score_method(args, "hodge")
        if hodge_score is not None:
            hodge_mean = hodge_score["mean_physical_smoothness"]
            if best_hodge is None or hodge_mean < best_hodge["mean_physical_smoothness"]:
                best_hodge = {
                    "attempt": index,
                    "mean_physical_smoothness": hodge_mean,
                    "hodge_up_weight": hodge_up_weight,
                    "bandwidth": bandwidth,
                    "embedding_dim": embedding_dim,
                    "selection": selection,
                    "function_basis": function_basis,
                }
            matched = (
                persistent_mean is not None
                and hodge_mean <= persistent_mean + args.baseline_tolerance
            )
            _append_log(
                [
                    f"## Sweep attempt {index}",
                    "",
                    f"- hodge mean_physical_smoothness: {hodge_mean:.6g}",
                    f"- persistent baseline: {persistent_mean}",
                    f"- matched_or_better: {matched}",
                    f"- hodge_up_weight: {hodge_up_weight}",
                    f"- bandwidth: {bandwidth}",
                    f"- embedding_dim: {embedding_dim}",
                    f"- selection: {selection}",
                    f"- function_basis: {function_basis}",
                    "",
                ]
            )
            if matched and args.stop_on_match:
                _append_log(
                    [
                        "## Sweep stopped",
                        "",
                        f"- matched persistent baseline at attempt: {index}",
                        f"- hodge mean_physical_smoothness: {hodge_mean:.6g}",
                        f"- persistent mean_physical_smoothness: {persistent_mean:.6g}",
                        "",
                    ]
                )
                return 0

    completion_lines = ["## Sweep complete", "", f"- attempts completed: {len(attempts)}"]
    if persistent_mean is not None:
        completion_lines.append(f"- persistent mean_physical_smoothness: {persistent_mean:.6g}")
    if best_hodge is not None:
        completion_lines.extend(
            [
                f"- best Hodge attempt: {best_hodge['attempt']}",
                f"- best Hodge mean_physical_smoothness: {best_hodge['mean_physical_smoothness']:.6g}",
                f"- best Hodge matched_or_better: {persistent_mean is not None and best_hodge['mean_physical_smoothness'] <= persistent_mean + args.baseline_tolerance}",
            ]
        )
    completion_lines.append("")
    _append_log(completion_lines)
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
    parser.add_argument("--physical-neighbors", type=int, default=12)
    parser.add_argument("--physical-score-stride", type=int, default=10)
    parser.add_argument("--baseline-tolerance", type=float, default=0.0)
    parser.add_argument("--run-persistent-baseline", action="store_true")
    parser.add_argument("--stop-on-match", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
