# Natural-image patch examples

This folder contains the 3x3 natural-image DCT patch experiments.

`download_data.py` downloads the classic javaPlex/applied-topology Matlab files
into `TDA/image_patches/data/`:

- `n50000Dct.mat`: 50,000 high-contrast 3x3 natural-image patches in the eight
  non-constant DCT coordinates.
- `nk300c30Dct.mat`: the `X(300, 30)` density core, useful as a circle-like
  sanity check.
- `nk15c30Dct.mat`: the `X(15, 30)` density core used for the
  Klein-bottle-like example.

Run:

```bash
python TDA/image_patches/download_data.py
python TDA/image_patches/run_klein_cut_coordinates.py \
  --mat-file TDA/image_patches/data/nk15c30Dct.mat \
  --n-points 1200 \
  --ph \
  --coeff 2 3
```

Outputs are written to `TDA/image_patches/output/`.
