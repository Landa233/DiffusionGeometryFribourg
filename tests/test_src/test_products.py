"""
Tests for the products module, particularly hodge_star function.
"""

from diffusion_geometry.classes.main import DiffusionGeometry
from diffusion_geometry.classes.tensors import Form, Function
from diffusion_geometry.classes.tensors.form import _wedge_product, wedge_operator
import numpy as np
from scipy.special import comb
from tests.conftest import setup_geom


def test_wedge_operator():
    """
    Verify that the linearised wedge operator produces the same result
    as directly computing the wedge product of two forms.
    """
    from diffusion_geometry.classes.operators import LinearOperator

    # Problem setup
    d = 6  # ambient dimension
    k1 = 2  # degree of first form
    k2 = 3  # degree of second form

    n = 100  # total number of sample points
    n0 = 70  # training points (DiffusionGeometry)
    n1 = 70  # evaluation points

    rng = np.random.default_rng(0)
    point_cloud = rng.standard_normal((n, d))

    # Ensure KNN parameters remain within valid bounds
    knn_kernel = min(32, max(1, n - 1))
    knn_bandwidth = min(8, knn_kernel)

    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=point_cloud,
        embedding_coords=point_cloud,
        n_function_basis=n0,
        n_coefficients=n1,
        knn_kernel=knn_kernel,
        knn_bandwidth=knn_bandwidth,
    )

    # Left form α
    alpha_coeffs = rng.standard_normal(n1 * int(comb(d, k1)))
    alpha = Form.from_coeffs(alpha_coeffs, dg, k1)

    # Construct the linear wedge operator: β ↦ α ∧ β
    wedge_op = wedge_operator(alpha, k2)

    # Right form β
    beta_coeffs = rng.standard_normal(n1 * int(comb(d, k2)))
    beta = Form.from_coeffs(beta_coeffs, dg, k2)

    # Compute wedge directly and via the operator
    wedge_direct = _wedge_product(alpha, beta)
    wedge_linearised = wedge_op @ beta.coeffs

    # Validate equality
    assert np.allclose(
        wedge_direct.coeffs, wedge_linearised, atol=1e-12
    ), "Linearised wedge operator does not match direct wedge product."


def test_wedge_pure_forms():
    d = 3

    n = 100
    n0 = 70
    n1 = 70

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, d))
    # Ensure KNN parameters remain within valid bounds for small sample sizes.
    knn_kernel = min(32, max(1, n - 1))
    knn_bandwidth = min(8, knn_kernel)

    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        embedding_coords=data,
        n_function_basis=n0,
        n_coefficients=n1,
        knn_kernel=knn_kernel,
        knn_bandwidth=knn_bandwidth,
    )
    a_data = np.zeros((n, 3))
    b_data = np.zeros((n, 3))
    a_data[:, 0] = 2.0
    a_data[:, 1] = 3.0

    b_data[:, 1] = 1.0
    b_data[:, 2] = 4.0

    a = Form.from_pointwise_basis(a_data, dg, 1)
    b = Form.from_pointwise_basis(b_data, dg, 1)

    batch_shape = (3, 4)
    a_batched_data = np.broadcast_to(a_data, batch_shape + a_data.shape)
    b_batched_data = np.broadcast_to(b_data, batch_shape + b_data.shape)

    a_batched = Form.from_pointwise_basis(a_batched_data, dg, 1)
    b_batched = Form.from_pointwise_basis(b_batched_data, dg, 1)

    #
    res = _wedge_product(a, b).to_pointwise_basis().reshape(n, -1)
    op_res = _wedge_product(b, a).to_pointwise_basis().reshape(n, -1)

    res_batched = (
        _wedge_product(a_batched, b_batched)
        .to_pointwise_basis()
        .reshape(*batch_shape, n, -1)
    )

    assert np.allclose(res[0], np.array([2.0, 8.0, 12.0]))
    assert np.allclose(res[0], -op_res[0])

    assert np.allclose(res_batched[0, 0, 0], np.array([2.0, 8.0, 12.0]))


def test_operator_overloading():
    d = 7

    n = 50
    n0 = 30
    n1 = 30

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, d))
    # Ensure KNN parameters remain within valid bounds for small sample sizes.
    knn_kernel = min(32, max(1, n - 1))
    knn_bandwidth = min(8, knn_kernel)

    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        embedding_coords=data,
        n_function_basis=n0,
        n_coefficients=n1,
        knn_kernel=knn_kernel,
        knn_bandwidth=knn_bandwidth,
    )
    function_coefficients = np.random.rand(n0)
    function = Function.from_coeffs(function_coefficients, dg)

    k1 = 2
    k2 = 3
    form_coeffs1 = np.random.rand(n1 * int(comb(d, k1)))
    form_coeffs2 = np.random.rand(n1 * int(comb(d, k2)))
    form1 = Form.from_coeffs(form_coeffs1, dg, k1)
    form2 = Form.from_coeffs(form_coeffs2, dg, k2)

    scalar = 0.5

    _ = scalar * function
    _ = function * scalar
    _ = scalar * form1
    _ = form1 * scalar
    _ = function * function
    _ = form1 * function
    _ = function * form1
    _ = form1 ^ form2
    _ = form2 ^ form1
    _ = function ^ form1
    _ = form1 ^ function


if __name__ == "__main__":
    test_hodge_star_degree_zero_forms()
    test_hodge_star_higher_degree_forms()
    test_hodge_star_all_degrees()
    test_wedge_operator()
    test_wedge_pure_forms()
    test_operator_overloading()
    print("All tests passed!")
