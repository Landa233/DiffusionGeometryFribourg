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
  - `conda run --no-capture-output -n basic-env pytest tests/test_methods/test_circular_coordinates.py` passed.
  - `python -m py_compile methods/circular_coordinates.py TDA/gardner2022/run_hodge_circular_coordinates.py tests/test_methods/test_circular_coordinates.py` passed.
- Blocked measurement:
  - The extracted Gardner data directory is not present at `TDA/gardner2022/data/Toroidal_topology_grid_cell_data`, so the full rat-location score could not be run yet.
