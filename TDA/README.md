# TDA examples

This folder contains small topological-data-analysis examples for the diffusion
geometry code in this repository.

## Folder structure

- `synthetic/`: synthetic torus, cylinder, and Klein bottle checks.
- `image_patches/`: natural-image 3x3 DCT patch experiments, including the
  Klein-bottle cut-coordinate example.
- `gardner2022/`: Gardner et al. grid-cell torus reproductions. The `original/`
  subfolder is an unchanged copy of the upstream notebooks and helper code used
  by the wrappers.

Each example folder owns its own `data/` and `output/` directories when it needs
them. Those generated/downloaded directories are ignored by git.

Runnable scripts are named for their task:

```bash
python TDA/synthetic/run_circular_coordinates.py
python TDA/synthetic/run_cylinder_noise_ablation.py
python TDA/synthetic/run_torus_noise_ablation.py
python TDA/synthetic/run_klein_cut_noise_ablation.py
python TDA/image_patches/download_data.py
python TDA/image_patches/run_klein_cut_coordinates.py
python TDA/gardner2022/run_persistent_homology.py
python TDA/gardner2022/run_hodge_circular_coordinates.py
```

Run scripts from the repository root so their relative data and output paths
match the defaults.
