# Hodge Circular Coordinate Attempts

## Checkpoint 1: Scoring harness

- Branch: `improve-gardner-circular-coords`
- Goal: choose Gardner Hodge circular coordinates by agreement with the rat's open-field physical trajectory, not just by intrinsic reconstruction error.
- Changes:
  - Added `hodge_up_weight` to `methods.circular_coordinates.circular_coordinates`, so candidate 1-forms can come from `down_laplacian(1) + hodge_up_weight * up_laplacian(1)`.
  - Added per-candidate Dirichlet energy of the learned coordinate functions.
  - Added Gardner CLI knobs for fixed bandwidth, embedding-coordinate truncation/standardization, monomial function bases, physical candidate selection, and physical scoring stride.
  - Added decoded physical smoothness scoring: nearest-neighbour wrapped angle variation over `(xx, yy)`, normalized by circular variance.
- Validation:
  - `PYTHONPYCACHEPREFIX=/tmp/dgf_pycache python3 -m py_compile methods/circular_coordinates.py TDA/gardner2022/run_hodge_circular_coordinates.py tests/test_methods/test_circular_coordinates.py` passed.
  - Could not run dependency tests in this shell: `conda` is not on `PATH`, and `/usr/bin/python3` has no `numpy`.
- Blocked measurement:
  - The extracted Gardner data directory is not present at `TDA/gardner2022/data/Toroidal_topology_grid_cell_data`, so the full rat-location score could not be run yet.

## Checkpoint 2: Repeatable parameter sweep

- Added `sweep_hodge_circular_coordinates.py` to run a compact grid over:
  - Hodge up-term weights,
  - constant bandwidths,
  - truncated/standardized embedding coordinates,
  - selection criteria,
  - diffusion versus monomial function bases.
- Each sweep delegates to `run_hodge_circular_coordinates.py`, so successful attempts append comparable physical smoothness measurements to this file.

## Checkpoint 3: Baseline comparison scorer

- Added `TDA/gardner2022/score_decoding.py` so saved persistent-homology and Hodge `*_decoding.npz` files can be evaluated by the same open-field physical smoothness score.
- Added `TDA/gardner2022/physical_coordinate_scores.py` so the Hodge runner and standalone scorer share exactly the same metric implementation.
- Added README commands for scoring both methods.
- This is the comparison gate for the goal: once the Gardner data and decoding NPZ files are present, use the scorer's `mean_physical_smoothness` to decide whether a Hodge attempt matches or beats the persistent baseline.
- Validation:
  - `python -m py_compile TDA/gardner2022/physical_coordinate_scores.py TDA/gardner2022/score_decoding.py TDA/gardner2022/run_hodge_circular_coordinates.py tests/test_methods/test_gardner_physical_scores.py` passed.
  - `conda run --no-capture-output -n basic-env pytest tests/test_methods/test_gardner_physical_scores.py tests/test_methods/test_circular_coordinates.py` passed.
  - `conda run --no-capture-output -n basic-env python TDA/gardner2022/score_decoding.py --help` passed.
