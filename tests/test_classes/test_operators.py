import pytest
import numpy as np
import warnings
from diffusion_geometry.operators import (
    block,
    hstack,
    vstack,
    LinearOperator,
    identity,
    zero,
)
from diffusion_geometry.tensors import BaseTensorSpace


def test_block_operator_construction(setup_geom):
    dg = setup_geom
    # Use small spaces for testing
    space = dg.function_space

    # Create simple operators
    op1 = identity(space)
    op2 = zero(space)
    op3 = zero(space)
    op4 = identity(space)

    # Test block construction [[I, 0], [0, I]]
    block_op = block([[op1, op2], [op3, op4]])

    assert isinstance(block_op, LinearOperator)
    assert block_op.domain.dim == space.dim * 2
    assert block_op.codomain.dim == space.dim * 2

    # Check shape
    dim = space.dim
    assert block_op.matrix.shape == (2 * dim, 2 * dim)

    # Check content roughly (since identity is stronger than weak form in implementation detail usually,
    # but here we rely on what LinearOperator exposes)
    full_mat = block_op.matrix
    assert np.allclose(full_mat[:dim, :dim], np.eye(dim))
    assert np.allclose(full_mat[:dim, dim:], np.zeros((dim, dim)))
    assert np.allclose(full_mat[dim:, :dim], np.zeros((dim, dim)))
    assert np.allclose(full_mat[dim:, dim:], np.eye(dim))


def test_hstack_vstack(setup_geom):
    dg = setup_geom
    space = dg.function_space
    op = identity(space)

    # hstack [I, I]
    h_op = hstack([op, op])
    assert h_op.domain.dim == space.dim * 2
    assert h_op.codomain.dim == space.dim
    full_mat = h_op.matrix
    dim = space.dim
    assert np.allclose(full_mat[:, :dim], np.eye(dim))
    assert np.allclose(full_mat[:, dim:], np.eye(dim))

    # vstack [I, I]
    v_op = vstack([op, op])
    assert v_op.domain.dim == space.dim
    assert v_op.codomain.dim == space.dim * 2
    full_mat_v = v_op.matrix
    assert np.allclose(full_mat_v[:dim, :], np.eye(dim))
    assert np.allclose(full_mat_v[dim:, :], np.eye(dim))


def test_block_errors(setup_geom):
    dg = setup_geom
    space = dg.function_space
    op = identity(space)

    # Empty block
    with pytest.raises(AssertionError, match="Provide at least one row"):
        block([])

    # Empty row
    with pytest.raises(AssertionError, match="Provide at least one column"):
        block([[]])

    # Non-rectangular
    with pytest.raises(AssertionError, match="must be rectangular"):
        block([[op], [op, op]])

    # Wrong type
    with pytest.raises(TypeError, match="must be LinearOperator instances"):
        block([[op, "not-op"]])


def test_mixed_spaces_block(setup_geom):
    dg = setup_geom
    s1 = dg.function_space
    s2 = dg.vector_field_space

    # op1: s1 -> s1
    # op2: s2 -> s1
    # op3: s1 -> s2
    # op4: s2 -> s2

    # [op1 op2]
    # [op3 op4]
    # Domain should be s1 x s2
    # Codomain should be s1 x s2

    op1 = identity(s1)
    op4 = identity(s2)
    # mixed ops (zero ops for simplicity)
    op2 = zero(domain=s2, codomain=s1)
    op3 = zero(domain=s1, codomain=s2)

    block_op = block([[op1, op2], [op3, op4]])

    d1 = s1.dim
    d2 = s2.dim

    assert block_op.domain.dim == d1 + d2
    assert block_op.codomain.dim == d1 + d2

    mat = block_op.matrix
    assert mat.shape == (d1 + d2, d1 + d2)
    # Check identity blocks
    assert np.allclose(mat[:d1, :d1], np.eye(d1))
    assert np.allclose(mat[d1:, d1:], np.eye(d2))


def test_spectrum_api_defaults(setup_geom):
    dg = setup_geom
    space = dg.function_space
    # Identity operator has all 1s as eigenvalues
    op = identity(space)

    # default behavior: returns (eigvals, eigvecs)
    res = op.spectrum()
    assert isinstance(res, tuple)
    assert len(res) == 2
    vals, vecs = res
    assert np.allclose(vals, 1.0)

    # Explicit eigvals_only=True: returns just array
    vals_only = op.spectrum(eigvals_only=True)
    assert isinstance(vals_only, np.ndarray)
    assert np.allclose(vals_only, 1.0)

    # Explicit eigvals_only=False: returns (eigvals, eigvecs)
    res_explicit = op.spectrum(eigvals_only=False)
    assert isinstance(res_explicit, tuple)
    assert len(res_explicit) == 2


def test_spectrum_eigenvector_batch_scaling_with_numpy(setup_geom):
    dg = setup_geom
    if dg.dim < 1:
        pytest.skip("Requires dim >= 1")

    vals, vecs = (dg.up_laplacian(1) + 1.0 * dg.down_laplacian(1)).spectrum()
    weights = np.exp(-vals)
    expected = vecs.coeffs * weights.reshape(vecs.batch_shape + (1,))

    for out in (weights * vecs, vecs * weights, np.multiply(weights, vecs)):
        assert out.__class__ is vecs.__class__
        assert out.batch_shape == vecs.batch_shape
        assert np.allclose(out.coeffs, expected)


def test_grad_application_avoids_matmul_runtime_warnings(setup_geom):
    dg = setup_geom
    f = dg.function_space.zeros()
    if dg.n_function_basis > 1:
        f.coeffs[1] = 1.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        X = f.grad()
        _ = X.coeffs
        _ = X.to_ambient()

    matmul_warnings = [
        w
        for w in caught
        if "matmul" in str(w.message) and w.filename.endswith("linear.py")
    ]
    assert not matmul_warnings
