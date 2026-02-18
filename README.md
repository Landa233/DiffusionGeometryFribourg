# Diffusion Geometry

**Data-driven calculus, geometry, and topology on point clouds.**

Diffusion Geometry is a Python framework for computing differential geometry on discrete data. By reformulating calculus in terms of **heat diffusion**, this package allows you to perform robust geometric analysis on noisy point clouds, graphs, and non-manifold singularities without requiring a mesh.

This software implements the methods described in the paper *Computing Diffusion Geometry* (Jones & Lanners, 2026).

<!-- ## Key Features

*   **Vector Calculus**: Compute gradients, divergence, and Laplacians on raw data.
*   **Riemannian Geometry**: Estimate geodesic distances, metrics, and curvature directly from point clouds.
*   **Topological Data Analysis**: Compute de Rham cohomology, circular coordinates, and perform Morse Theory analysis orders of magnitude faster than persistent homology.
*   **Mesh-Free PDEs**: Solve Heat, Wave, and Geodesic Distance equations.
*   **Visualisation**: Interactive 3D plotting for scalar fields, vector fields, and forms. -->

## Installation

```bash
pip install -e .
```

### Dependencies

- **numpy**: Numerical computing.
- **scipy**: Scientific computing and sparse matrices.
- **opt_einsum**: Optimized Einstein summation.
- **scikit-learn**: Nearest neighbor search and utilities.
- **pytest**: (Optional) For running tests.

## Quick Start

Compute the gradient of a function on a noisy point cloud:

```python
import numpy as np
import diffusion_geometry as dg

# 1. Load Data (e.g., a noisy torus)
# shape: (n_points, 3)
points = np.load("torus_points.npy")
values = points[:, 2]  # Scalar signal (e.g., height)

# 2. Initialize Geometry
model = dg.DiffusionGeometry.from_point_cloud(points)

# 3. Create a Function
f = model.function(values)

# 4. Compute Calculus Operations
grad_f = f.grad()            # Returns a VectorField
laplacian_f = f.laplacian()  # Returns a Function

# 5. Spectra of geometric Laplacians (on 1-forms / vector fields)
hodge_eigenvalues, hodge_eigenvectors = model.laplacian(1).spectrum()

connection_laplacian = model.levi_civita.adjoint @ model.levi_civita
connection_eigenvalues, connection_eigenvectors = connection_laplacian.spectrum()
```

<!-- ## Core Concepts

The core idea is to avoid the "Manifold Hypothesis" and instead use the **Carré du Champ** operator ($\Gamma$) derived from a diffusion process.

1. **Heat Kernel**: We construct a graph where edges represent the probability of heat diffusion between points.
2. **Markov Chain**: This graph defines a random walk (Markov chain) on the data.
3. **Carré du Champ**: We use the covariance of the random walk to estimate the local geometry (metric).
4. **Calculus**: All geometric objects (gradients, curvature, topology) are computed statistically from this operator. -->

## Reference

If you use this software in your research, please cite:

```bibtex
@article{jones2026computing,
  title={Computing Diffusion Geometry},
  author={Jones, Iolo and Lanners, David},
  journal={arXiv preprint arXiv:2602.06006},
  year={2026}
}

@article{jones2024diffusion,
  title={Diffusion Geometry},
  author={Jones, Iolo},
  journal={arXiv preprint arXiv:2405.10858},
  year={2024}
}
```
