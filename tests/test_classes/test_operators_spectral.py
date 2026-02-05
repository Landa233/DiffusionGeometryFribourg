import pytest
import numpy as np
from diffusion_geometry.operators import LinearOperator, identity, zero


def test_spectral_decomposition_properties(setup_geom):
    dg = setup_geom
    space = dg.function_space
    dim = space.coeff_dimension

    # 1. Self-adjoint case
    # Construct a random symmetric matrix for the weak form
    rng = np.random.default_rng(42)
    A = rng.standard_normal((dim, dim))
    weak_sym = A + A.T
    op_sym = LinearOperator(domain=space, codomain=space, weak_matrix=weak_sym)

    assert op_sym.is_self_adjoint

    vals, coords = op_sym._spectral_decomposition()
    # Check shape: (K,) and (K, K) where K is basis size
    K = space.orthonormal_basis.shape[1]
    assert vals.shape == (K,)
    assert coords.shape == (K, K)

    # Check eigenvalue sorting (ascending for real)
    assert np.all(np.diff(vals) >= 0)

    # Check spectral decomposition property: A_coords = V D V.T
    # The A in coords is basis.T @ weak @ basis
    basis = space.orthonormal_basis
    A_coords = basis.conj().T @ weak_sym @ basis
    assert np.allclose(coords @ np.diag(vals) @ coords.conj().T, A_coords)


def test_inverse_full_rank(setup_geom):
    dg = setup_geom
    space = dg.function_space

    # Use identity + small noise to ensure full rank but distinct eigvals
    op = identity(space)
    rng = np.random.default_rng(43)
    noise = rng.standard_normal(op.matrix.shape) * 0.01
    op_noisy = LinearOperator(
        domain=space, codomain=space, strong_matrix=op.matrix + noise
    )

    op_inv = op_noisy.inverse()

    # L * L^-1 should be identity (if full rank)
    res = op_noisy @ op_inv
    expected = identity(space)

    # Note: since the operator might be restricted by the basis,
    # we check consistency on the subspace.
    # The simplest check is (L @ L^-1).matrix vs restricted identity
    basis = space.orthonormal_basis
    gram = space.gram
    proj_identity = basis @ basis.conj().T @ gram

    assert np.allclose(res.matrix, proj_identity, atol=1e-10)


def test_inverse_low_rank_pseudo(setup_geom):
    dg = setup_geom
    space = dg.function_space
    basis = space.orthonormal_basis
    K = basis.shape[1]

    if K < 2:
        pytest.skip("Space too small for low-rank test")

    # Create rank-1 operator in the orthonormal basis
    # A = v v* where v is the first basis vector
    v_coords = np.zeros((K, 1))
    v_coords[0] = 1.0
    A_coords = v_coords @ v_coords.T

    # Lift to full space strong matrix: Φ A Φ* G
    strong = basis @ A_coords @ basis.conj().T @ space.gram
    op_low_rank = LinearOperator(domain=space, codomain=space, strong_matrix=strong)

    op_inv = op_low_rank.inverse(rcond=0.5)

    # Moore-Penrose property: L @ L_inv @ L = L
    res = op_low_rank @ op_inv @ op_low_rank
    assert np.allclose(res.matrix, op_low_rank.matrix)

    # Check that only one eigenvalue is kept
    vals = op_low_rank.spectrum(eigvals_only=True)
    assert np.count_nonzero(np.abs(vals) > 0.5) == 1


def test_inverse_rcond_masking(setup_geom):
    dg = setup_geom
    space = dg.function_space
    basis = space.orthonormal_basis
    K = basis.shape[1]

    # Create operator with specific eigenvalues: 1.0, 0.1, 0.01
    if K < 3:
        pytest.skip("Space too small for rcond test")

    vals_diag = np.zeros(K)
    vals_diag[0] = 1.0
    vals_diag[1] = 0.1
    vals_diag[2] = 0.01

    strong = basis @ np.diag(vals_diag) @ basis.conj().T @ space.gram
    op = LinearOperator(domain=space, codomain=space, strong_matrix=strong)

    # rcond = 0.05 should keep 1.0 and 0.1, but mask 0.01
    op_inv = op.inverse(rcond=0.05)
    vals_out = op_inv.spectrum(eigvals_only=True)

    # The eigenvalues of the inverse should be 1/1.0, 1/0.1, and 0
    kept_vals = vals_out[np.abs(vals_out) > 1e-10]
    expected_vals = [1.0 / 1.0, 1.0 / 0.1]

    # Check that our specific inverted values exist in the result
    for val in expected_vals:
        assert np.any(np.isclose(kept_vals, val))

    # Verify that 1/0.01 is NOT present
    assert not np.any(np.isclose(kept_vals, 1.0 / 0.01))

    # Verify we didn't keep too many
    # If the operator was zero everywhere else, we should have exactly 2
    assert np.count_nonzero(np.abs(vals_out) > 1e-10) == 2


def test_spectral_edge_cases(setup_geom):
    dg = setup_geom
    space = dg.function_space

    # Zero operator
    op_zero = zero(space)
    vals, vecs = op_zero.spectrum()
    assert np.allclose(vals, 0)

    op_inv = op_zero.inverse()
    assert np.allclose(op_inv.matrix, 0)

    # Test with empty space if possible (not typical, but spectrum() handles it)
    # We can't easily create an empty space via setup_geom without mocking,
    # but we can check the logic on a zero-dim array if the class allowed it.
    # For now, zero operator is a good surrogate for "no signal".
