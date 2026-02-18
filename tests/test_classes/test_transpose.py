import pytest
import numpy as np
from diffusion_geometry.tensors.tensor02 import Tensor02
from diffusion_geometry.tensors.tensor02sym import Tensor02Sym
from diffusion_geometry.operators import BilinearOperator


def test_tensor02_transpose(setup_geom):
    """Test standard (0,2)-tensor transpose."""
    dg = setup_geom
    n1, d = dg.n_coefficients, dg.dim
    # Tensor02 coeff dimension is n1*d*d
    # wrap() expects (..., coeff_dim)
    total_dim = n1 * d * d
    coeffs_flat = np.random.randn(total_dim)
    T = dg.tensor02_space.wrap(coeffs_flat)

    # Check T.T property
    assert isinstance(T.T, Tensor02)

    # To check values, we need to reshape T.coeffs to (n1, d, d) ourselves to compare
    # T.coeffs is flat (n1*d*d,)
    # T.T.coeffs is flat (n1*d*d,)

    T_reshaped = T.coeffs.reshape(n1, d, d)
    T_T_reshaped = T.T.coeffs.reshape(n1, d, d)

    assert np.allclose(T_T_reshaped, np.swapaxes(T_reshaped, -1, -2))

    # Check T.T.T == T
    assert np.allclose(T.T.T.coeffs, T.coeffs)

    # Check T(X, Y) == T.T(Y, X)
    # Create random X, Y
    X = dg.vector_field_space.wrap(
        np.random.randn(dg.vector_field_space.dim)
    )
    Y = dg.vector_field_space.wrap(
        np.random.randn(dg.vector_field_space.dim)
    )

    val1 = T(X, Y)
    val2 = T.T(Y, X)
    # Tensor02(X, Y) returns a numpy array representing the scalar field pointwise values
    assert np.allclose(val1, val2)


def test_tensor02sym_transpose(setup_geom):
    """Test symmetric (0,2)-tensor transpose."""
    dg = setup_geom
    d_sym = dg.dim * (dg.dim + 1) // 2
    coeffs = np.random.randn(dg.n_coefficients * d_sym)
    S = dg.tensor02sym_space.wrap(coeffs)

    # Check S.transpose() returns self
    assert S.transpose() is S
    assert S.T is S

    # Check it commutes with full tensor conversion
    T_full = S.full_tensor
    assert np.allclose(T_full.T.coeffs, T_full.coeffs)

    # Check symmetry relation with full tensor transpose manually
    # Should satisfy S.full_tensor.T == S.full_tensor
    assert np.allclose(S.full_tensor.T.coeffs, S.full_tensor.coeffs)


def test_bilinear_operator_transpose(setup_geom):
    """Test BilinearOperator transpose."""
    dg = setup_geom
    # Create a random bilinear operator
    domain_a = dg.vector_field_space
    domain_b = dg.vector_field_space
    codomain = dg.vector_field_space

    n1, d = dg.n_coefficients, dg.dim

    # Strong tensor shape: (codim, dim_a, dim_b)
    # codim = n1*d (approx, dim)
    shape = (
        codomain.dim,
        domain_a.dim,
        domain_b.dim,
    )

    # Creating a full size tensor might be too big for testing if n is large
    # But usually n is small in tests.
    strong = np.random.randn(*shape)

    B = BilinearOperator(
        domain_a=domain_a, domain_b=domain_b, codomain=codomain, strong_tensor=strong
    )

    B_T = B.T

    # Check domains swapped
    assert B_T.domain_a is domain_b
    assert B_T.domain_b is domain_a
    assert B_T.codomain is codomain

    # Check tensor values swapped
    assert np.allclose(B_T.strong, np.swapaxes(strong, -1, -2))

    # Check application B(x, y) == B.T(y, x)
    x = domain_a.wrap(np.random.randn(domain_a.dim))
    y = domain_b.wrap(np.random.randn(domain_b.dim))

    res1 = B(x, y)
    res2 = B_T(y, x)

    assert np.allclose(res1.coeffs, res2.coeffs)


def test_bilinear_operator_repr_includes_domains(setup_geom):
    dg = setup_geom
    B = dg.lie_bracket

    rep = repr(B)

    assert "BilinearOperator(" in rep
    assert "domain_a=VectorFieldSpace(dim=" in rep
    assert "domain_b=VectorFieldSpace(dim=" in rep
    assert "codomain=VectorFieldSpace(dim=" in rep
