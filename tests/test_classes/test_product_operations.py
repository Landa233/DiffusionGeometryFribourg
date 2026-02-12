"""
Tests for all tensor product operations, including batched and unbatched cases.
"""

import numpy as np
import pytest
import unittest
from opt_einsum import contract
from scipy.special import comb
from diffusion_geometry.tensors import Form, Function, VectorField


def test_scalar_multiplication(setup_geom):
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    # 1-form
    coeffs = np.random.rand(dg.n_coefficients * dg.dim)
    omega = dg.form_space(1).wrap(coeffs)

    # Scalar * Form
    scaled = 2.5 * omega
    assert isinstance(scaled, Form)
    assert np.allclose(scaled.coeffs, 2.5 * coeffs)

    # Form * Scalar
    scaled_right = omega * 2.5
    assert isinstance(scaled_right, Form)
    assert np.allclose(scaled_right.coeffs, 2.5 * coeffs)

    # Batched Form * Scalar
    batch_shape = (5,)
    coeffs_batched = np.random.rand(5, dg.n_coefficients * dg.dim)
    omega_batched = dg.form_space(1).wrap(coeffs_batched)

    scaled_batched = omega_batched * 3.0
    assert scaled_batched.batch_shape == (5,)
    assert np.allclose(scaled_batched.coeffs, coeffs_batched * 3.0)


def test_function_form_product(setup_geom):
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    g = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    omega = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))

    # Function * Form
    res = f * omega
    assert isinstance(res, Form)
    assert res.degree == 1

    # Check Distributivity: (f + g) * omega = f*omega + g*omega
    res_dist = (f + g) * omega
    res_sum = (f * omega) + (g * omega)
    assert np.allclose(res_dist.coeffs, res_sum.coeffs)

    # Form * Function
    res2 = omega * f
    assert np.allclose(res.coeffs, res2.coeffs)

    # Batched Function * Form
    f_batch = dg.function_space.wrap(np.random.rand(5, dg.n_function_basis))
    omega_batch = dg.form_space(1).wrap(np.random.rand(5, dg.n_coefficients * dg.dim))

    res_batch = f_batch * omega_batch
    assert res_batch.batch_shape == (5,)


def test_tensor_product_1_forms(setup_geom):
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    alpha = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))
    beta = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))
    gamma = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))

    # alpha * beta -> (0,2)-tensor
    prod = alpha * beta
    from diffusion_geometry.tensors.tensor02 import Tensor02

    assert isinstance(prod, Tensor02)

    # Check Distributivity: (alpha + beta) * gamma = alpha*gamma + beta*gamma
    p1 = (alpha + beta) * gamma
    p2 = (alpha * gamma) + (beta * gamma)
    assert np.allclose(p1.coeffs, p2.coeffs)


def test_wedge_product(setup_geom):
    dg = setup_geom
    if dg.dim < 2:
        pytest.skip("Requires dim >= 2")

    alpha = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))
    beta = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))

    # alpha ^ beta
    wedge = alpha ^ beta
    assert isinstance(wedge, Form)
    assert wedge.degree == 2

    # Anti-commutativity for 1-forms
    # alpha ^ beta = - beta ^ alpha
    wedge2 = beta ^ alpha
    assert np.allclose(wedge.coeffs, -wedge2.coeffs)

    # Function ^ Form = Function * Form
    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    res1 = f ^ alpha
    res2 = f * alpha
    assert np.allclose(res1.coeffs, res2.coeffs)

    res3 = alpha ^ f
    res4 = alpha * f
    assert np.allclose(res3.coeffs, res4.coeffs)


def test_vector_field_function_product(setup_geom):
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    # Create a random vector field
    vf_coeffs = np.random.rand(dg.n_coefficients * dg.dim)
    X = dg.vector_field_space.wrap(vf_coeffs)

    # Function * VectorField (now delegated to VectorField.__rmul__)
    prod1 = f * X
    assert isinstance(prod1, VectorField)

    # VectorField * Function
    prod2 = X * f
    assert isinstance(prod2, VectorField)

    assert np.allclose(prod1.coeffs, prod2.coeffs)

    # Check Distributivity: (f + g) * X == f*X + g*X
    g = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    dist_res = (f + g) * X
    sum_res = (f * X) + (g * X)
    assert np.allclose(dist_res.coeffs, sum_res.coeffs)


def test_form_function_delegation(setup_geom):
    """
    Explicitly test that Function delegates to Form for products.
    """
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    omega = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))

    # This should work via pure delegation:
    # Function.__mul__(Form) -> NotImplemented
    # Form.__rmul__(Function) -> Form.__mul__(Function) -> result
    res = f * omega
    assert isinstance(res, Form)
    assert res.degree == 1


def test_tensor02_product(setup_geom):
    """
    Test that generic tensors (like Tensor02) inherit Function multiplication.
    """
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    # Create (0,2) tensor (vector field tensor product)
    X = dg.vector_field_space.wrap(np.random.rand(dg.n_coefficients * dg.dim))
    T = X.flat() * X.flat()  # 1-form * 1-form -> Tensor02

    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))

    # Tensor02 * Function (inherited from Tensor)
    res = T * f
    from diffusion_geometry.tensors.tensor02 import Tensor02

    assert isinstance(res, Tensor02)
    assert res.shape == T.shape

    # Check simple scalar property
    scalar = 2.0
    res_scalar = T * scalar
    assert np.allclose(res_scalar.coeffs, T.coeffs * scalar)

    # Function * Tensor02 (inherited rmul)
    res_reverse = f * T
    assert np.allclose(res.coeffs, res_reverse.coeffs)


def test_division_operations(setup_geom):
    """
    Test generic division of tensors by functions.
    """
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    # Create test objects
    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    g = dg.function_space.wrap(
        np.random.rand(dg.n_function_basis) + 2.0
    )  # Ensure non-zero
    X = dg.vector_field_space.wrap(np.random.rand(dg.n_coefficients * dg.dim))

    # Function / Function (inherited)
    quotient_f = f / g
    assert isinstance(quotient_f, Function)

    # VectorField / Function (inherited)
    quotient_X = X / g
    assert isinstance(quotient_X, VectorField)

    # Function division is tricky due to basis truncation.
    # Instead, we verify that X / g behaves roughly correctly for a constant function g.

    constant_val = 2.0
    # Create constant function via pointwise basis
    g_const = Function.from_pointwise_basis(np.full((dg.n,), constant_val), dg)

    quotient_const = X / g_const
    assert np.allclose(
        quotient_const.coeffs, X.coeffs / constant_val, rtol=1e-5, atol=1e-5
    )

    # Tensor02 / Function
    T = X.flat() * X.flat()
    quotient_T = T / g


def test_exhaustive_arithmetic(setup_geom):
    """
    Exhaustive test of arithmetic operators (+, -) for all combinations.
    """
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    # Create one of each type
    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    v = dg.vector_field_space.wrap(np.random.rand(dg.n_coefficients * dg.dim))
    form = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))
    from diffusion_geometry.tensors.tensor02 import Tensor02

    t02 = Tensor02.from_coeffs(np.random.rand(dg.n_coefficients * dg.dim * dg.dim), dg)
    scalar = 3.0

    # Helper to assert type error or value error (due to space mismatch)
    def assert_type_error(op, a, b):
        with pytest.raises((TypeError, AssertionError)):
            op(a, b)

    import operator

    ops = [operator.add, operator.sub]

    # --- Function ---
    for op in ops:
        assert isinstance(op(f, f), type(f))  # Func +/- Func
        assert isinstance(op(f, scalar), type(f))  # Func +/- Scalar
        assert isinstance(op(scalar, f), type(f))  # Scalar +/- Func

        # Func +/- Other Tensor -> TypeError (incompatible spaces)
        assert_type_error(op, f, v)
        assert_type_error(op, f, form)
        assert_type_error(op, f, t02)

    # --- VectorField ---
    for op in ops:
        assert isinstance(op(v, v), type(v))  # Vec +/- Vec
        assert_type_error(op, v, scalar)  # Vec +/- Scalar
        assert_type_error(op, scalar, v)  # Scalar +/- Vec
        assert_type_error(op, v, f)

    # --- Form ---
    for op in ops:
        assert isinstance(op(form, form), type(form))  # Form +/- Form
        assert_type_error(op, form, scalar)
        assert_type_error(op, scalar, form)

    # --- Tensor02 ---
    for op in ops:
        assert isinstance(op(t02, t02), type(t02))
        assert_type_error(op, t02, scalar)
        assert_type_error(op, scalar, t02)


def test_exhaustive_products(setup_geom):
    """
    Exhaustive test of product operators (*, /) for all combinations.
    """
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    v = dg.vector_field_space.wrap(np.random.rand(dg.n_coefficients * dg.dim))
    form = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))
    from diffusion_geometry.tensors.tensor02 import Tensor02

    t02 = Tensor02.from_coeffs(np.random.rand(dg.n_coefficients * dg.dim * dg.dim), dg)
    scalar = 2.0

    # --- Scalar Products (valid for all) ---
    tensors = [f, v, form, t02]
    for T in tensors:
        # T * scalar
        assert isinstance(T * scalar, type(T))
        assert isinstance(scalar * T, type(T))
        # T / scalar
        assert isinstance(T / scalar, type(T))

    # --- Function Products ---
    # F * F -> F
    assert isinstance(f * f, type(f))
    # F / F -> F
    assert isinstance(f / f, type(f))

    # --- Tensor * Function ---
    # V * F -> V
    assert isinstance(v * f, type(v))
    assert isinstance(f * v, type(v))
    # V / F -> V
    assert isinstance(v / f, type(v))

    # Form * F -> Form
    assert isinstance(form * f, type(form))
    assert isinstance(f * form, type(form))

    # T02 * F -> T02
    assert isinstance(t02 * f, type(t02))
    assert isinstance(f * t02, type(t02))

    # --- Tensor Products ---
    # Form * Form -> Tensor02
    assert isinstance(form * form, Tensor02)

    # Invalid products
    # V * V -> NotImplemented / TypeError (no defined product yet in this code base?)
    # Based on current code: VectorField doesn't define __mul__ for VectorField, base returns NotImplemented


def test_exhaustive_wedge(setup_geom):
    """
    Exhaustive test of wedge product (^) logic.
    """
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    v = dg.vector_field_space.wrap(np.random.rand(dg.n_coefficients * dg.dim))
    # 1-form
    form = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))

    scalar = 2.5

    # --- Function/Scalar Wedge (Multiplication aliasing) ---
    # F ^ F -> F (*)
    assert isinstance(f ^ f, type(f))

    # F ^ Scalar -> F
    assert isinstance(f ^ scalar, type(f))
    assert isinstance(scalar ^ f, type(f))

    # F ^ Form -> Form (*)
    res_f_form = f ^ form
    assert isinstance(res_f_form, type(form))
    assert np.allclose(res_f_form.coeffs, (f * form).coeffs)

    # Form ^ F -> Form (*)
    res_form_f = form ^ f
    assert isinstance(res_form_f, type(form))
    assert np.allclose(res_form_f.coeffs, (form * f).coeffs)

    # Form ^ Scalar -> Form
    assert isinstance(form ^ scalar, type(form))
    assert isinstance(scalar ^ form, type(form))
    assert np.allclose((form ^ scalar).coeffs, (form * scalar).coeffs)

    # --- VectorField Wedge ---
    # V ^ Scalar -> V (via generic Tensor implementation)
    # V inherits Tensor.__xor__, which handles Scalar/Function
    assert isinstance(v ^ scalar, type(v))
    assert isinstance(scalar ^ v, type(v))

    # V ^ F -> V
    assert isinstance(v ^ f, type(v))

    # V ^ V -> NotImplemented -> TypeError (no wedge defined for V)
    with pytest.raises(TypeError):
        _ = v ^ v

    # --- Form Wedge ---
    # Form ^ Form -> Form (degree = deg1 + deg2)
    # 1-form ^ 1-form -> 2-form (if dim >= 2)
    res_wedge = form ^ form
    if dg.dim < 2:
        from diffusion_geometry.tensors import Function

        assert isinstance(res_wedge, Function)
    else:
        assert isinstance(res_wedge, Form)
        assert res_wedge.degree == form.degree + form.degree

    # Test DG check
    # Check that it verifies correct DG
    # Use a dummy object that pretends to be a Form but has a different DG
    # Test DG check
    # Check that it verifies correct DG

    # Define a clean MockForm that inherits from Form to pass isinstance checks
    # but does not run Form.__init__ to avoid setup overhead.
    class MockDG:
        pass

    class MockForm(Form):
        def __init__(self, dg_obj):
            # Bypass Form.__init__ entirely
            self._dg_obj = dg_obj

        @property
        def dg(self):
            return self._dg_obj

    fake_dg = MockDG()
    # We must construct it carefully to not break if something accesses other attrs (unlikely in the check)
    mock_form = MockForm(fake_dg)

    with pytest.raises(
        AssertionError, match="Operands must belong to the same DiffusionGeometry"
    ):
        _ = form ^ mock_form


def test_wedge_product_values(setup_geom):
    """
    Test specific values of wedge product for correctness.
    Checks dx^i ^ dx^j = sgn * dx^{i,j}.
    """
    dg = setup_geom
    if dg.dim < 2:
        pytest.skip("Requires dim >= 2")

    n = dg.n
    d = dg.dim

    # 1-form dx^0: component 0 is 1, others 0.
    data_dx0 = np.zeros((n, d))
    data_dx0[:, 0] = 1.0
    dx0 = Form.from_pointwise_basis(data_dx0, dg, 1)

    # 1-form dx^1: component 1 is 1, others 0.
    data_dx1 = np.zeros((n, d))
    data_dx1[:, 1] = 1.0
    dx1 = Form.from_pointwise_basis(data_dx1, dg, 1)

    # wedge
    w = dx0 ^ dx1  # should be dx^0 ^ dx^1
    C = int(comb(d, 2))
    w_pw = w.to_pointwise_basis().reshape(n, C)

    # Note on tolerance: in diffusion geometry approximations with small n_coefficients,
    # constant forms like dx^i are only approximately captured.
    # We check that the (0,1) component (index 0) is dominant and close to 1.

    # Check component 0 is dominant
    mean_val = np.mean(w_pw[:, 0])
    assert np.isclose(mean_val, 1.0, atol=2e-1, rtol=2e-1)

    # Check others are small relative to component 0
    if w_pw.shape[1] > 1:
        others_norm = np.mean(np.abs(w_pw[:, 1:]))
        assert others_norm < 0.3

    # Test anti-symmetry (this should be exact for coefficients)
    w_rev = dx1 ^ dx0
    assert np.allclose(w_rev.coeffs, -w.coeffs)


def test_wedge_product_zero(setup_geom):
    """
    Test that wedge product is zero if degree exceeds dimension.
    """
    dg = setup_geom
    # Create d-form
    # If d=1, 1-form. If d=2, 2-form.
    k = dg.dim
    coeffs = np.random.rand(dg.n_coefficients * int(comb(dg.dim, k)))
    omega = dg.form_space(k).wrap(coeffs)

    # 1-form
    alpha = dg.form_space(1).wrap(np.random.rand(dg.n_coefficients * dg.dim))

    # omega ^ alpha has degree d+1 > d.
    # The library returns a zero Function (0-form) in this case to avoid "oversized forms".
    res = omega ^ alpha
    from diffusion_geometry.tensors import Function

    assert isinstance(res, Function)
    assert np.allclose(res.coeffs, 0.0)


def test_arithmetic_operations(setup_geom):
    """
    Test generic arithmetic and Function scalar special cases.
    """
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    f = dg.function_space.wrap(np.random.rand(dg.n_function_basis))
    scalar = 3.5

    # 1. Function + Scalar
    res_add = f + scalar
    assert isinstance(res_add, type(f))
    # Check pointwise at a few points or check coeffs 0 if basis allows
    # Better: check that f - res_add is constant -scalar
    diff = res_add - f
    # diff should be constant function with value scalar
    # check mean or 0-th coeff
    if dg.n_function_basis > 0:
        # 0-th coeff for constant function is val * sqrt(vol)
        expected_0 = scalar * np.sqrt(dg.measure.sum())
        assert np.isclose(diff.coeffs[0], expected_0)
        assert np.allclose(diff.coeffs[1:], 0)

    # 2. Scalar + Function
    res_radd = scalar + f
    assert np.allclose(res_radd.coeffs, res_add.coeffs)

    # 3. Function - Scalar
    res_sub = f - scalar
    res_sub_check = f + (-scalar)
    assert np.allclose(res_sub.coeffs, res_sub_check.coeffs)

    # 4. Scalar - Function
    res_rsub = scalar - f
    res_rsub_check = -(f - scalar)
    assert np.allclose(res_rsub.coeffs, res_rsub_check.coeffs)

    # 5. Tensor + Tensor (VectorField)
    X = dg.vector_field_space.wrap(np.random.rand(dg.n_coefficients * dg.dim))
    Y = dg.vector_field_space.wrap(np.random.rand(dg.n_coefficients * dg.dim))

    res_vf_add = X + Y
    assert np.allclose(res_vf_add.coeffs, X.coeffs + Y.coeffs)

    # 6. Negation
    neg_X = -X
    assert np.allclose(neg_X.coeffs, -X.coeffs)

    # 7. Invalid: VectorField + Scalar
    with pytest.raises(TypeError):
        # Should return NotImplemented and eventually TypeError
        _ = X + scalar
