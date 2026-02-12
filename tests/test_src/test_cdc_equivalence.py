import pytest
import numpy as np
from diffusion_geometry.core import (
    carre_du_champ_knn,
    carre_du_champ_graph,
)


def generate_random_knn(n, k, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate random neighbor indices
    # Ensure no self-loops for simplicity, though not strictly required by functions
    nbr_indices = np.random.randint(0, n, size=(n, k))

    # Generate random normalized diffusion kernel
    diffusion_kernel = np.random.rand(n, k)
    diffusion_kernel /= diffusion_kernel.sum(axis=1, keepdims=True)

    return nbr_indices, diffusion_kernel


def convert_knn_to_graph(nbr_indices, diffusion_kernel):
    n, k = nbr_indices.shape

    # Create sources and targets
    # Source is the neighbor (j), Target is the node (i)
    # The functions use:
    # knn: means_f = (diffusion_kernel[:, :, None] * nbrs_f).sum(axis=1) where nbrs_f are values at nbr_indices
    # graph: w_f = nbrs_f * diffusion_kernel[:, None]; np.add.at(means_f, tgt, w_f)
    # So edge_index[0] should be source (j), edge_index[1] should be target (i)

    sources = nbr_indices.flatten()
    targets = np.repeat(np.arange(n), k)

    edge_index = np.array([sources, targets])
    flat_weights = diffusion_kernel.flatten()

    return edge_index, flat_weights


@pytest.mark.parametrize("use_mean_centres", [True, False])
@pytest.mark.parametrize("use_bandwidths", [True, False])
@pytest.mark.parametrize("shapes", ["equal", "diff"])
def test_cdc_equivalence(use_mean_centres, use_bandwidths, shapes):
    n = 20
    k = 5
    d1 = 3
    d2 = 4 if shapes == "diff" else 3

    np.random.seed(42)

    # Data Generation
    f = np.random.randn(n, d1)
    h = np.random.randn(n, d2)

    nbr_indices, diffusion_kernel = generate_random_knn(n, k, seed=42)

    if use_bandwidths:
        bandwidths = np.random.rand(n) + 0.1  # positive bandwidths
    else:
        bandwidths = None

    # Conversion Logic
    edge_index, flat_weights = convert_knn_to_graph(nbr_indices, diffusion_kernel)

    # Run both functions
    cdc_knn = carre_du_champ_knn(
        f=f,
        h=h,
        diffusion_kernel=diffusion_kernel,
        nbr_indices=nbr_indices,
        bandwidths=bandwidths,
        use_mean_centres=use_mean_centres,
    )

    cdc_graph = carre_du_champ_graph(
        f=f,
        h=h,
        diffusion_kernel=flat_weights,
        edge_index=edge_index,
        bandwidths=bandwidths,
        use_mean_centres=use_mean_centres,
    )

    # Assertion
    # The functions might return different shapes?
    # knn: [n, (f_shape), (h_shape)]
    # graph: [n, (f_shape), (h_shape)]
    # They should be identical.

    assert cdc_knn.shape == cdc_graph.shape

    # Check values
    np.testing.assert_allclose(cdc_knn, cdc_graph, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
