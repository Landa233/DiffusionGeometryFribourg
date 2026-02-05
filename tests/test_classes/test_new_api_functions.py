"""Tests for the high-level DiffusionGeometry API."""

import numpy as np
from opt_einsum import contract

from diffusion_geometry.classes.operators import BilinearOperator, LinearOperator
from diffusion_geometry.classes.tensors import Function, VectorField


def test_hessian_operator(setup_geom):
    dg = setup_geom
    # cache = dg.cache

    hess_op = dg.hessian
    assert isinstance(hess_op, LinearOperator)

    d_sym = dg.dim * (dg.dim + 1) // 2
    assert hess_op.matrix.shape == (dg.n_coefficients * d_sym, dg.n_function_basis)
    expected_matrix = dg.tensor02sym_space.gram_inv @ dg.hessian.weak
    assert np.allclose(hess_op.matrix, expected_matrix)
    assert np.allclose(hess_op.weak, dg.hessian.weak)

    f = dg.function(np.random.randn(dg.n))
    hess_f = dg.hessian(f)
    assert hess_f.coeffs.shape == (dg.n_coefficients * d_sym,)


# Removed outdated tests:
# - test_hessian_matrix_function
# - test_hessian_operator_application
# - test_hessian_operator_batched
# as dg.hessian is now a LinearOperator returning a Tensor02Sym, not a BilinearOperator.


def test_levi_civita_operator(setup_geom):
    dg = setup_geom
    # cache = dg.cache

    lc_02 = dg.levi_civita
    assert isinstance(lc_02, LinearOperator)

    n1_d = dg.n_coefficients * dg.dim
    assert lc_02.matrix.shape == (n1_d * dg.dim, n1_d)
    expected_matrix = dg.tensor02_space.gram_inv @ dg.levi_civita.weak
    assert np.allclose(lc_02.matrix, expected_matrix)
    assert np.allclose(lc_02.weak, dg.levi_civita.weak)

    X = dg.vector_field(np.random.randn(dg.n, dg.dim))
    lc_02_X = dg.levi_civita(X)
    assert lc_02_X.coeffs.shape == (n1_d * dg.dim,)


# Removed outdated tests:
# - test_levi_civita_11_operator
# - test_levi_civita_11_with_vector_fields
# as levi_civita_11 is unused/removed.


def test_lie_bracket_operator(setup_geom):
    dg = setup_geom
    # cache = dg.cache

    bracket = dg.lie_bracket
    assert isinstance(bracket, BilinearOperator)

    n1_d = dg.n_coefficients * dg.dim
    assert bracket.strong.shape == (n1_d, n1_d, n1_d)
    assert np.allclose(bracket.weak, dg.lie_bracket.weak)

    expected_strong = contract(
        "iA,Ajk->ijk",
        dg.vector_field_space.gram_inv,
        dg.lie_bracket.weak,
    )
    assert np.allclose(bracket.strong, expected_strong)


def test_lie_bracket_with_vector_fields(setup_geom):
    dg = setup_geom

    X = dg.vector_field(np.random.randn(dg.n, dg.dim))
    Y = dg.vector_field(np.random.randn(dg.n, dg.dim))

    bracket = dg.lie_bracket
    partial = bracket(X)
    assert isinstance(partial, LinearOperator)

    lb_result = bracket(X, Y)
    assert lb_result.coeffs.shape == (dg.n_coefficients * dg.dim,)

    lb_yx = bracket(Y, X)
    assert np.allclose(lb_result.coeffs, -lb_yx.coeffs)


def test_function_wrapper_methods(setup_geom):
    dg = setup_geom

    f = dg.function(np.random.randn(dg.n))
    hess = dg.hessian(f)
    expected = dg.hessian(f)
    assert np.allclose(hess.coeffs, expected.coeffs)

    hess_op = dg.hessian
    assert isinstance(hess_op, LinearOperator)


def test_api_values(setup_geom):
    dg = setup_geom

    hess_op = dg.hessian
    assert hess_op.weak is not None

    lc02_op = dg.levi_civita
    assert lc02_op.weak is not None

    lb_op = dg.lie_bracket
    assert lb_op.weak is not None
