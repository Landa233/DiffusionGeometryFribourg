# Gardner et al. 2022 grid-cell examples

This folder contains wrappers for the Gardner et al. grid-cell torus analysis.
The `original/` folder is an unchanged copy of the upstream notebooks and helper
code used by the wrappers.

Place the Figshare archive at:

```text
TDA/gardner2022/data/Toroidal_topology_grid_cell_data.zip
```

Extract it to:

```text
TDA/gardner2022/data/Toroidal_topology_grid_cell_data/
```

The expected archive MD5 is `379bfdca61cd54d5f58cab9d3ba477de`.

Run:

```bash
python TDA/gardner2022/run_persistent_homology.py \
  --rat R --module 1 --session OF --day day2

python TDA/gardner2022/run_hodge_circular_coordinates.py \
  --rat R --module 1 --session OF --day day2
```

Outputs are written to `TDA/gardner2022/output/`.

To compare decoded coordinates from the persistent-homology and Hodge runners
with the same physical-location criterion, run:

```bash
python TDA/gardner2022/score_decoding.py --method persistent \
  --rat R --module 1 --session OF --day day2

python TDA/gardner2022/score_decoding.py --method hodge \
  --rat R --module 1 --session OF --day day2
```
