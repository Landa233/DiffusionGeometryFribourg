"""
Tests for DirectSumSpace functionality.
"""

import pytest
import numpy as np
from diffusion_geometry.classes.tensor_spaces import DirectSumSpace


def test_direct_sum_creation(setup_geom):
    """Test creation and properties of DirectSumSpace."""
    dg = setup_geom

    # Create simpler spaces
    s1 = dg.function_space
    s2 = dg.vector_field_space

    # Test __add__ syntax
    ds1 = s1 + s2
    assert isinstance(ds1, DirectSumSpace)
    assert len(ds1.spaces) == 2
    assert ds1.coeff_dimension == s1.coeff_dimension + s2.coeff_dimension

    # Test associativity logic (implementation flattens)
    ds2 = ds1 + s1
    assert len(ds2.spaces) == 3
    assert ds2.spaces[0] is s1
    assert ds2.spaces[1] is s2
    assert ds2.spaces[2] is s1

    # Test explicit creation error
    with pytest.raises(AssertionError, match="at least one space"):
        DirectSumSpace(dg, [])

    # Test error on mixed DG instances (mocking a different DG is hard but let's try type check logic if we could)
    # The current DG fixture provides one instance, validation logic is simple: if space.dg != dg -> Error.


def test_direct_sum_gram_matrix(setup_geom):
    """Test Gram matrix structure."""
    dg = setup_geom
    s1 = dg.function_space
    s2 = dg.vector_field_space

    ds = s1 + s2
    gram = ds.gram

    d1 = s1.coeff_dimension
    d2 = s2.coeff_dimension

    assert gram.shape == (d1 + d2, d1 + d2)

    # Check block structure
    # Top-left should be s1.gram
    assert np.allclose(gram[:d1, :d1], s1.gram)
    # Bottom-right should be s2.gram
    assert np.allclose(gram[d1:, d1:], s2.gram)
    # Off-diagonals should be zero
    assert np.allclose(gram[:d1, d1:], np.zeros((d1, d2)))
    assert np.allclose(gram[d1:, :d1], np.zeros((d2, d1)))


def test_direct_sum_gram_inv(setup_geom):
    """Test Gram inverse (pseudoinverse) calculation."""
    dg = setup_geom
    s1 = dg.function_space
    s2 = dg.vector_field_space
    ds = s1 + s2

    gram_inv = ds.gram_inv
    d1 = s1.coeff_dimension

    # Should be block diagonal of inverses
    assert np.allclose(gram_inv[:d1, :d1], s1.gram_inv)
    assert np.allclose(gram_inv[d1:, d1:], s2.gram_inv)
    assert np.allclose(gram_inv[:d1, d1:], 0)


def test_direct_sum_orthonormal_basis(setup_geom):
    """Test orthonormal basis aggregation."""
    dg = setup_geom
    s1 = dg.function_space
    s2 = dg.vector_field_space
    ds = s1 + s2

    basis = ds.orthonormal_basis
    d1 = s1.coeff_dimension
    w1 = s1.orthonormal_basis.shape[1]

    assert basis.shape == (
        ds.coeff_dimension,
        s1.orthonormal_basis.shape[1] + s2.orthonormal_basis.shape[1],
    )

    # Top-left block
    assert np.allclose(basis[:d1, :w1], s1.orthonormal_basis)
    # Bottom-right block
    assert np.allclose(basis[d1:, w1:], s2.orthonormal_basis)
    # Off-diagonal blocks
    assert np.allclose(basis[:d1, w1:], 0)
    assert np.allclose(basis[d1:, :w1], 0)


def test_direct_sum_metric_apply(setup_geom):
    """Test metric application across blocks."""
    dg = setup_geom
    # Use two spaces of the same type to ensuring matching scalar field resolution (n0 vs n0)
    # DirectSumSpace implementation is simple and does not auto-pad resolutions.
    s1 = dg.function_space
    s2 = dg.function_space
    ds = s1 + s2

    rng = np.random.default_rng(42)

    # Create random coefficients
    c1_a = rng.standard_normal(s1.coeff_dimension)
    c2_a = rng.standard_normal(s2.coeff_dimension)
    coeffs_a = np.concatenate([c1_a, c2_a])

    c1_b = rng.standard_normal(s1.coeff_dimension)
    c2_b = rng.standard_normal(s2.coeff_dimension)
    coeffs_b = np.concatenate([c1_b, c2_b])

    # Computed via direct sum
    res = ds.metric_apply(coeffs_a, coeffs_b)

    # Expected: metric_apply(c1_a, c1_b) + metric_apply(c2_a, c2_b)
    term1 = s1.metric_apply(c1_a, c1_b)
    term2 = s2.metric_apply(c2_a, c2_b)

    expected = term1 + term2

    assert np.allclose(res, expected)


def test_packing_and_splitting(setup_geom):
    """Test pack and split_coeffs utilities."""
    dg = setup_geom
    s1 = dg.function_space
    s2 = dg.vector_field_space
    ds = s1 + s2

    rng = np.random.default_rng(99)
    coeffs1 = rng.standard_normal(s1.coeff_dimension)
    coeffs2 = rng.standard_normal(s2.coeff_dimension)

    # Wrap individually
    t1 = s1.wrap(coeffs1)
    t2 = s2.wrap(coeffs2)

    # Pack
    packed = ds.pack(t1, t2)
    assert packed.space is ds
    assert np.allclose(packed.coeffs[: s1.coeff_dimension], coeffs1)
    assert np.allclose(packed.coeffs[s1.coeff_dimension :], coeffs2)

    # Split
    parts = ds.split_coeffs(packed.coeffs)
    assert len(parts) == 2
    assert np.allclose(parts[0], coeffs1)
    assert np.allclose(parts[1], coeffs2)

    # Error checking
    with pytest.raises(AssertionError, match="Expected 2 tensors"):
        ds.pack(t1)
