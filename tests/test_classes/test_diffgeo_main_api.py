import numpy as np
import pytest
from scipy.special import comb

from src.tensors.vector_fields.vector_field_space import VectorFieldSpace


def test_projection_round_trip(setup_geom):
    dg = setup_geom
    f = np.random.RandomState(0).randn(dg.n)
    func = Function.from_pointwise_basis(f, dg)
    f_rec = func.to_pointwise_basis()
    r = f - f_rec
    # Residual should be orthogonal to span{u} in the measure-weighted inner product
    ortho = (dg.measure[:, None] * dg.function_basis).T @ r
    rel = np.linalg.norm(ortho) / (np.linalg.norm(f) + 1e-12)
    assert rel < 1e-6


def test_linear_operator_works_with_compatible_spaces(setup_geom):
    dg = setup_geom
    canonical_space = dg.vector_field_space
    noncanonical_space = VectorFieldSpace(dg)

    weak_canonical = np.zeros_like(canonical_space.gram)
    weak_noncanonical = np.zeros_like(noncanonical_space.gram)

    canon_op = LinearOperator(
        domain=canonical_space,
        codomain=canonical_space,
        weak_matrix=weak_canonical,
    )

    rogue_op = LinearOperator(
        domain=noncanonical_space,
        codomain=noncanonical_space,
        weak_matrix=weak_noncanonical,
    )

    vector = canonical_space.wrap(np.zeros(canonical_space.coeff_dimension))

    # Should not raise
    rogue_op(vector)

    # Should not raise
    _ = canon_op @ rogue_op
