import numpy as np
import pytest

from diffusion_geometry.core.diffusion.symmetric_kernel import SymmetricKernelConstructor

def test_resolve_immersion_missing_inputs():
    nbr_indices = np.array([[0, 1], [1, 0]])
    kernel = np.array([[1.0, 0.5], [0.5, 1.0]])

    constructor = SymmetricKernelConstructor(nbr_indices, kernel)

    def dummy_regularise(x):
        return x

    with pytest.raises(ValueError, match="data_matrix and/or immersion_coords must be provided"):
        constructor.resolve_immersion(
            regularise=dummy_regularise,
            data_matrix=None,
            immersion_coords=None
        )

def test_resolve_immersion_with_coords():
    nbr_indices = np.array([[0, 1], [1, 0]])
    kernel = np.array([[1.0, 0.5], [0.5, 1.0]])

    constructor = SymmetricKernelConstructor(nbr_indices, kernel)

    def dummy_regularise(x):
        return x * 2.0

    immersion_coords = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = constructor.resolve_immersion(
        regularise=dummy_regularise,
        data_matrix=None,
        immersion_coords=immersion_coords
    )

    assert np.allclose(result, immersion_coords)

def test_resolve_immersion_with_data_matrix():
    nbr_indices = np.array([[0, 1], [1, 0]])
    kernel = np.array([[1.0, 0.5], [0.5, 1.0]])

    constructor = SymmetricKernelConstructor(nbr_indices, kernel)

    def dummy_regularise(x):
        return x * 2.0

    data_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = constructor.resolve_immersion(
        regularise=dummy_regularise,
        data_matrix=data_matrix,
        immersion_coords=None
    )

    assert np.allclose(result, data_matrix * 2.0)
