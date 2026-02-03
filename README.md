# Diffusion Geometry

**Data-driven calculus, geometry, and topology on point clouds.**

**Diffusion Geometry** is a Python framework for computing differential geometry on discrete data. Unlike traditional methods that rely on mesh structures or strict manifold assumptions, this package reformulates calculus using **heat diffusion**. This allows for robust analysis of noisy point clouds, graphs, and non-manifold singularities using the spectral properties of the heat kernel.

This software implements the methods described in the paper *Computing Diffusion Geometry* (Jones & Lanners, 2025).

## 🚀 Key Features

* **Vector Calculus**: Compute Gradients, Divergence, and Laplacians on raw point clouds.

* **Differential Geometry**: Estimate the **Levi-Civita connection**, **Geodesic distances**, and **Riemann/Sectional Curvature** directly from data.

* **Topological Data Analysis**: Compute **de Rham cohomology** (Betti numbers) via the Hodge Laplacian, find circular coordinates, and perform **Morse Theory** analysis.

* **Mesh-Free PDEs**: Solve the Heat and Wave equations or simulate fluid flows along vector fields.

* **Visualisation**: Built-in 2D/3D plotting for scalar fields, vector fields, and forms using Plotly.

## 📦 Installation

```bash
pip install -e .
```

### Dependencies
- **numpy**: Numerical computing.
- **scipy**: Scientific computing and sparse matrices.
- **opt_einsum**: Optimized Einstein summation.
- **scikit-learn**: Nearest neighbor search and utilities.
- **pytest**: (Optional) For running tests.

## ⚡ Quick Start

Here is how to compute the gradient of a function on a noisy point cloud and visualize the flow.

```python
import numpy as np
import diffusion_geometry as dg

# 1. Load Data (e.g., a noisy torus)
# shape: (n_points, 3)
points = np.load("torus_points.npy")
values = points[:, 2]  # Scalar signal (e.g., height z)

# 2. Initialize Geometry
# 'n_function_basis' controls the smoothness resolution
# 'n_coefficients' controls the complexity of vector fields
model = dg.DiffusionGeometry.from_point_cloud(
    points,
    n_function_basis=100,
    n_vector_basis=50,
    knn_kernel=32
)

# 3. Create a Function
f = model.function(values)

# 4. Compute Calculus Operations
# The API automatically dispatches to the correct operator
grad_f = model.grad(f)            # Returns a VectorField
laplacian_f = model.laplacian(f)  # Returns a Function

# 5. Visualise
# Automatically detects 3D data and renders interactive plots via Plotly
f.plot(title="Scalar Field (Height)")
grad_f.plot(color=f, title="Gradient Vector Field")
```

## 📖 Core Concepts

This package avoids the "Manifold Hypothesis." Instead of estimating tangent spaces (which fail at singularities or intersections), we represent geometric objects using the **Carré du Champ operator** ($\Gamma$), derived from the diffusion process.

### 1. The Geometry Object

The `DiffusionGeometry` class is the central entry point. It manages the spectral basis (eigenfunctions of the heat kernel) used to represent all geometric objects.

```python
# Initialize from raw coordinates
dg = DiffusionGeometry.from_point_cloud(data)

# Or initialize from a pre-computed graph/kernel
dg = DiffusionGeometry.from_knn_graph(indices, distances, immersion)
```

### 2. Tensor Objects

We provide high-level wrappers for geometric objects. These handle the complex spectral coefficients internally, allowing you to do math naturally.

#### Function (0-form)

Represents scalar functions on the data.

**Creation:**
```python
f = dg.function(f_data)  # From data basis values
```

**Properties & Methods:**
* `f.coeffs`: Diffusion basis coefficients
* `f.to_pointwise_basis()`: Convert to pointwise values (cached)
* `f.grad()`: Compute gradient (returns VectorField)

**Arithmetic:**
```python
g = f + h      # Addition
k = 2 * f      # Scalar multiplication
m = f + 5      # Add constant
```

#### VectorField

Represents vector fields on the data.

**Creation:**
```python
X = dg.vector_field(X_data)  # From data basis values (n×dim array)
```

**Properties & Methods:**
* `X.coeffs`: Diffusion basis coefficients
* `X.to_pointwise_basis()`: Convert to pointwise vectors (cached)
* `X.to_operator()`: Convert to linear operator matrix
* `X(f)` or `X @ f`: Apply to function (directional derivative)
* `X.div()`: Compute divergence (returns Function)
* `X.curl()`: Compute curl (returns Function for 2D, VectorField for 3D)

**Arithmetic:**
```python
Y = X + Z      # Addition
W = 2 * X      # Scalar multiplication
```

#### Other Tensors

* **Form**: Differential $k$-forms ($\Omega^k$). Extensible base class for higher-order forms.
* **Tensor02**: Bilinear forms (metrics, connections).

```python
# Inner products and norms (Riemannian metric)
magnitude = h.norm() 
angle = dg.inner(f, g)
```

### 3. Operators

Operators can be called as methods on tensors or as standalone operators from the model.

```python
# Object-Oriented style
X = f.grad()
div_X = X.div()

# Functional style
X = model.grad(f)
omega = model.d(f)  # Exterior derivative (0-form -> 1-form)
```

## 🔬 Advanced Usage

### Chained Operations

Complex expressions become simple using the object-oriented API.

```python
result = X((f + g).grad())           # X applied to grad(f + g)
laplacian_f = (f.grad()).div()       # Laplacian of f
mixed = X(f) + Y(g)                  # Linear combinations
```

### Basis Conversions

We manage representations efficiently under the hood.

```python
# Automatic caching
f_data_1 = f.to_pointwise_basis()  # Computed and cached
f_data_2 = f.to_pointwise_basis()  # Retrieved from cache

# Access raw coefficients directly if needed
coeffs = f.coeffs             # Diffusion basis
```

### Hodge Decomposition & Topology

Analyze the topology of your data by decomposing forms into exact, coexact, and harmonic components. Harmonic forms correspond to topological holes (cohomology).

```python
# Create a random 1-form
alpha = model.form(random_data, degree=1)

# Decompose: alpha = d(beta) + delta(gamma) + harmonic
exact, coexact, harmonic = alpha.hodge_decomposition()

# The harmonic part reveals topological features (e.g., loops on a torus)
harmonic.plot()
```

### Curvature Estimation

Compute the intrinsic curvature of the data without a mesh.

```python
# Define two vector fields using ambient coordinates
X = model.vector_field_from_coords(coords[:, 0:3]) # e.g. ambient X
Y = model.vector_field_from_coords(coords[:, 3:6]) # e.g. ambient Y

# Compute Sectional Curvature K(X, Y)
K_XY = model.sectional_curvature(X, Y)
```

### Vector Field Flows (PDEs)

Solve the flow of a vector field or the heat/wave equation on the data.

```python
# Solve heat equation: du/dt = -Delta u
u_t = model.solve_heat_equation(initial_f, time=1.0)

# Integrate flow lines of a vector field X
flow_lines = model.integrate_flow(X, time_steps=100)
```

## 📚 Reference

This software is based on the theory established in:

> **Computing Diffusion Geometry**  
> Iolo Jones and David Lanners  
> Durham University, 2025

If you use this software in your research, please cite:

```bibtex
@article{jones2025computing,
  title={Computing Diffusion Geometry},
  author={Jones, Iolo and Lanners, David},
  journal={arXiv preprint},
  year={2025}
}
```
