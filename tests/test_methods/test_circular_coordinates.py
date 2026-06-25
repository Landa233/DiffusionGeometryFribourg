import numpy as np

from diffusion_geometry import DiffusionGeometry
from TDA.synthetic_circular_coordinates import monomial_function_basis
from methods.circular_coordinates import circular_coordinates


def test_monomial_function_basis_degree_two_shape():
    data = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])

    basis = monomial_function_basis(data, degree=2)

    assert basis.shape == (3, 6)
    assert np.allclose(basis[:, 0], 1.0)
    assert np.isfinite(basis).all()


def test_circular_coordinates_smoke_on_circle():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=100)
    data = np.column_stack((np.cos(theta), np.sin(theta)))
    dg = DiffusionGeometry.from_point_cloud(
        data,
        n_function_basis=18,
        n_coefficients=9,
        knn_kernel=18,
        knn_bandwidth=7,
    )

    result = circular_coordinates(dg, epsilon=1.0, k=6)

    assert result.coordinate_values.shape == (dg.n, 2)
    assert result.angle.shape == (dg.n,)
    assert np.all(np.isfinite(result.coordinate_values))
    assert np.all(np.isfinite(result.angle))
    assert np.all((0.0 <= result.angle) & (result.angle < 2.0 * np.pi))
    assert len(result.candidates) > 0
    assert np.isfinite(result.candidate.reconstruction_error)
    assert result.candidate.similarity > 0.0
    filtered = [
        candidate for candidate in result.candidates if candidate.passed_hodge_filter
    ]
    expected = filtered[0] if filtered else result.candidates[0]
    assert result.candidate is expected
