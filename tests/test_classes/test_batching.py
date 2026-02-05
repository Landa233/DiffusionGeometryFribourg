import pytest
import numpy as np

# from diffusion_geometry.classes.main import DiffusionGeometry


def test_metric_batching(setup_geom):
    dg = setup_geom
    # Create two arbitrary vectors in vector field space
    v1_coeffs = np.random.randn(dg.vector_field_space.coeff_dimension)
    v2_coeffs = np.random.randn(dg.vector_field_space.coeff_dimension)

    v1 = dg.vector_field_space.wrap(v1_coeffs)
    v2 = dg.vector_field_space.wrap(v2_coeffs)

    # 1. Matching batch shapes () and ()
    # Should work fine
    res = dg.g(v1, v2)
    assert res.shape == (dg.n,)

    # 2. Matching batch shapes (B,) and (B,)
    B = 5
    v1_batch = dg.vector_field_space.wrap(
        np.broadcast_to(v1_coeffs, (B, len(v1_coeffs)))
    )
    v2_batch = dg.vector_field_space.wrap(
        np.broadcast_to(v2_coeffs, (B, len(v2_coeffs)))
    )

    res_batch = dg.g(v1_batch, v2_batch)
    assert res_batch.shape == (B, dg.n)

    # 3. Mismatching batch shapes (B,) and (B+1,)
    v3_batch = dg.vector_field_space.wrap(
        np.broadcast_to(v2_coeffs, (B + 1, len(v2_coeffs)))
    )
    with pytest.raises(AssertionError, match="not compatible"):
        dg.g(v1_batch, v3_batch)

    # 4. Mismatching batch shapes () and (B,)
    # If broadcasting is supported, this should work.
    res_broadcast = dg.g(v1, v2_batch)
    assert res_broadcast.shape == (B, dg.n)


def test_inner_product_batching(setup_geom):
    dg = setup_geom
    f1_coeffs = np.random.randn(dg.function_space.coeff_dimension)
    f2_coeffs = np.random.randn(dg.function_space.coeff_dimension)

    f1 = dg.function_space.wrap(f1_coeffs)
    f2 = dg.function_space.wrap(f2_coeffs)

    # 1. Matching
    res = dg.inner(f1, f2)
    assert isinstance(res, float)

    # 2. Matching batched
    B = 3
    f1_batch = dg.function_space.wrap(
        np.broadcast_to(f1_coeffs, (B, len(f1_coeffs))))
    f2_batch = dg.function_space.wrap(
        np.broadcast_to(f2_coeffs, (B, len(f2_coeffs))))

    res_batch = dg.inner(f1_batch, f2_batch)
    assert res_batch.shape == (B,)

    # 3. Mismatch
    f3_batch = dg.function_space.wrap(
        np.broadcast_to(f2_coeffs, (B + 1, len(f2_coeffs)))
    )
    with pytest.raises(AssertionError, match="not compatible"):
        dg.inner(f1_batch, f3_batch)


def test_bilinear_operator_batching(setup_geom):
    dg = setup_geom
    # Lie bracket is bilinear: [X, Y]
    X_coeffs = np.random.randn(dg.vector_field_space.coeff_dimension)
    Y_coeffs = np.random.randn(dg.vector_field_space.coeff_dimension)

    X = dg.vector_field_space.wrap(X_coeffs)
    Y = dg.vector_field_space.wrap(Y_coeffs)

    # 1. Matching
    res = dg.lie_bracket(X, Y)
    assert res.space == dg.vector_field_space
    assert res.batch_shape == ()

    # 2. Matching batched
    B = 4
    X_batch = dg.vector_field_space.wrap(
        np.broadcast_to(X_coeffs, (B, len(X_coeffs))))
    Y_batch = dg.vector_field_space.wrap(
        np.broadcast_to(Y_coeffs, (B, len(Y_coeffs))))

    res_batch = dg.lie_bracket(X_batch, Y_batch)
    assert res_batch.batch_shape == (B,)

    # 3. Mismatch
    Y_mismatch = dg.vector_field_space.wrap(
        np.broadcast_to(Y_coeffs, (B + 1, len(Y_coeffs)))
    )
    with pytest.raises(AssertionError, match="Batch shape mismatch"):
        dg.lie_bracket(X_batch, Y_mismatch)
