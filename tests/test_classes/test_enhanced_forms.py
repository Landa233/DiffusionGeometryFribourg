"""
Tests for the enhanced Form class with differential operators and function multiplication.
"""

import numpy as np
import pytest
from scipy.special import comb
from diffusion_geometry.classes.main import DiffusionGeometry
from diffusion_geometry.classes.tensors import Function, Form


def test_enhanced_form_creation(setup_geom):
    """Test enhanced Form creation with proper dimension handling."""
    dg = setup_geom

    # Test functions (0-forms) are handled separately
    coeffs_0 = np.random.rand(dg.n_function_basis)
    func0 = dg.function_space.wrap(coeffs_0)
    assert func0.coeffs.shape == (dg.n_function_basis,)
    if dg.dim >= 1:
        coeffs_flat = np.random.rand(dg.n_coefficients * dg.dim)
        with pytest.raises(AssertionError):
            Form(dg.form_space(1), coeffs_flat, degree=0)
    func_from_factory = dg.form(np.random.rand(dg.n), degree=0)
    assert isinstance(func_from_factory, Function)

    # Test 1-forms (use n1 coefficients)
    if dg.dim >= 1:
        coeffs_1 = np.random.rand(dg.n_coefficients * int(comb(dg.dim, 1)))
        form1 = dg.form_space(1).wrap(coeffs_1)
        assert form1.degree == 1
        assert form1.coeffs.shape == (dg.n_coefficients * dg.dim,)

    # Test higher degree forms
    if dg.dim >= 2:
        coeffs_2 = np.random.rand(dg.n_coefficients * int(comb(dg.dim, 2)))
        form2 = dg.form_space(2).wrap(coeffs_2)
        assert form2.degree == 2


def test_form_arithmetic(setup_geom):
    """Test arithmetic operations on forms."""
    dg = setup_geom
    np.random.seed(42)

    # Create two 1-forms if applicable
    if dg.dim >= 1:
        data1 = np.random.rand(dg.n, dg.dim)
        data2 = np.random.rand(dg.n, dg.dim)

        form1 = dg.form(data1, degree=1)
        form2 = dg.form(data2, degree=1)

        # Test addition
        sum_form = form1 + form2
        assert isinstance(sum_form, Form)
        assert sum_form.degree == 1

        # Test subtraction
        diff_form = form1 - form2
        assert isinstance(diff_form, Form)
        assert diff_form.degree == 1

        # Test scalar multiplication
        scaled_form = 2 * form1
        assert isinstance(scaled_form, Form)
        assert scaled_form.degree == 1
        assert np.allclose(scaled_form.coeffs, 2 * form1.coeffs)


def test_function_multiplication_division():
    """Test function multiplication and division."""
    np.random.seed(42)
    data = np.random.rand(50, 2)  # Increase size to avoid k-neighbors issue
    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        immersion_coords=data,
        n_function_basis=10,
        n_coefficients=8,
        knn_kernel=min(16, data.shape[0] - 1),
        knn_bandwidth=min(8, data.shape[0] - 1),
    )

    # Create test functions
    f_data = np.random.rand(dg.n) + 0.1  # Add offset to avoid division issues
    g_data = np.random.rand(dg.n) + 0.1

    f = dg.function(f_data)
    g = dg.function(g_data)

    # Test multiplication
    fg = f * g
    assert isinstance(fg, Function)

    # Verify multiplication in data basis
    fg_data = fg.to_pointwise_basis()
    expected_product = f_data * g_data
    relative_error = np.linalg.norm(fg_data - expected_product) / np.linalg.norm(
        expected_product
    )
    assert relative_error < 1.0  # Allow for some approximation error

    # Test division
    f_over_g = f / g
    assert isinstance(f_over_g, Function)

    # Verify division in data basis
    f_over_g_data = f_over_g.to_pointwise_basis()
    expected_quotient = f_data / g_data
    relative_error = np.linalg.norm(f_over_g_data - expected_quotient) / np.linalg.norm(
        expected_quotient
    )
    assert relative_error < 1.0  # Allow for some approximation error


def test_form_error_handling():
    """Test error handling in Form operations."""
    np.random.seed(42)
    data = np.random.rand(40, 2)  # Increase size
    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        immersion_coords=data,
        n_function_basis=8,
        n_coefficients=6,
        knn_kernel=min(16, data.shape[0] - 1),
        knn_bandwidth=min(8, data.shape[0] - 1),
    )

    # Test invalid degree
    coeffs_valid = np.random.rand(dg.n_coefficients * dg.dim)
    with pytest.raises(AssertionError):
        Form(dg.form_space(1), coeffs_valid, degree=-1)

    with pytest.raises(AssertionError):
        Form(dg.form_space(1), coeffs_valid, degree=dg.dim + 1)

    # Test invalid coefficient shape
    with pytest.raises(AssertionError):
        dg.form_space(1).wrap(np.random.rand(5))

    if dg.dim >= 1:
        wrong_shape = np.random.rand(dg.n_coefficients * dg.dim + 1)
        with pytest.raises(AssertionError):
            dg.form_space(1).wrap(wrong_shape)

    with pytest.raises(AssertionError):
        dg.form_space(1).wrap(np.random.rand(dg.n_function_basis))

    if dg.dim >= 2:
        form1 = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))
        form2 = dg.form_space(2).wrap(
            np.random.rand(dg.n_coefficients * int(comb(dg.dim, 2)))
        )
        with pytest.raises(AssertionError):
            form1 + form2
