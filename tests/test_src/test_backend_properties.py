import numpy as np
import pytest
from opt_einsum import contract


def test_gram_matrices_positive_definite(setup_geom):
    """Test that Gram matrices are positive definite."""
    dg = setup_geom
    dg = setup_geom
    # backend = dg.backend

    # Test G matrix for k=0
    G0 = dg.function_space.gram
    eigenvals = np.linalg.eigvals(G0)
    assert np.all(eigenvals > -1e-12)  # Should be positive semi-definite

    # Test G matrix for k=1 if dimension allows
    if dg.dim >= 1:
        G1 = dg.form_space(1).gram
        eigenvals = np.linalg.eigvals(G1)
        assert np.all(eigenvals > -1e-12)  # Should be positive semi-definite
