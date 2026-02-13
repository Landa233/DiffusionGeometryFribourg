import numpy as np
import pytest

from diffusion_geometry.core import DiffusionGeometry, knn_graph, markov_chain


def _build_knn_inputs(n=96, d=4, knn_kernel=24, knn_bandwidth=8):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, d))
    nbr_distances, nbr_indices = knn_graph(data_matrix=data, knn_kernel=knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances=nbr_distances,
        nbr_indices=nbr_indices,
        knn_bandwidth=knn_bandwidth,
    )
    return data, nbr_distances, nbr_indices, kernel, bandwidths


def _assert_valid_geometry(dg, data_shape):
    n, d = data_shape
    assert dg.n == n
    assert dg.dim == d
    assert dg.immersion_coords.shape == (n, d)
    assert dg.measure.shape == (n,)
    assert dg.function_basis.shape[0] == n
    assert np.isfinite(dg.immersion_coords).all()
    assert np.isfinite(dg.measure).all()
    assert np.isfinite(dg.function_basis).all()


@pytest.mark.parametrize("regularisation_method", ["diffusion", "bandlimit", "none"])
def test_from_point_cloud_regularisation_modes_construct(regularisation_method):
    data, _, _, _, _ = _build_knn_inputs()
    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        regularisation_method=regularisation_method,
        knn_kernel=24,
        knn_bandwidth=8,
        n_function_basis=20,
    )
    _assert_valid_geometry(dg, data.shape)

    probe = np.arange(data.size, dtype=float).reshape(data.shape)
    regularised = dg._regularise(probe)
    assert regularised.shape == probe.shape
    if regularisation_method == "none":
        assert np.allclose(regularised, probe)


@pytest.mark.parametrize("regularisation_method", ["diffusion", "bandlimit", "none"])
def test_from_knn_graph_regularisation_modes_construct(regularisation_method):
    data, nbr_distances, nbr_indices, _, _ = _build_knn_inputs()
    dg = DiffusionGeometry.from_knn_graph(
        nbr_indices=nbr_indices,
        nbr_distances=nbr_distances,
        data_matrix=data,
        regularisation_method=regularisation_method,
        knn_bandwidth=8,
        n_function_basis=20,
    )
    _assert_valid_geometry(dg, data.shape)


@pytest.mark.parametrize("regularisation_method", ["diffusion", "bandlimit", "none"])
def test_from_knn_kernel_regularisation_modes_construct(regularisation_method):
    data, _, nbr_indices, kernel, bandwidths = _build_knn_inputs()
    dg = DiffusionGeometry.from_knn_kernel(
        nbr_indices=nbr_indices,
        kernel=kernel,
        bandwidths=bandwidths,
        immersion_coords=None,
        data_matrix=data,
        regularisation_method=regularisation_method,
        n_function_basis=20,
    )
    _assert_valid_geometry(dg, data.shape)


def test_from_knn_kernel_requires_data_or_immersion_for_none_regularisation():
    _, _, nbr_indices, kernel, bandwidths = _build_knn_inputs()
    with pytest.raises(ValueError, match="data_matrix and/or immersion_coords"):
        DiffusionGeometry.from_knn_kernel(
            nbr_indices=nbr_indices,
            kernel=kernel,
            bandwidths=bandwidths,
            immersion_coords=None,
            regularisation_method="none",
            n_function_basis=20,
        )


def test_from_graph_kernel_respects_n_coefficients():
    n = 9
    sources = np.arange(n)
    targets = np.roll(sources, -1)
    edge_index = np.vstack(
        [np.concatenate([sources, targets]), np.concatenate([targets, sources])]
    )
    kernel = np.ones(edge_index.shape[1], dtype=float)
    rng = np.random.default_rng(1)
    immersion_coords = rng.standard_normal((n, 2))

    dg = DiffusionGeometry.from_graph_kernel(
        edge_index=edge_index,
        kernel=kernel,
        immersion_coords=immersion_coords,
        n_coefficients=3,
    )

    assert dg.n == n
    assert dg.n_function_basis == n
    assert dg.n_coefficients == 3


def test_from_edges_respects_n_coefficients():
    n = 10
    sources = np.arange(n)
    targets = np.roll(sources, -1)
    edge_index = np.vstack([sources, targets])
    rng = np.random.default_rng(2)
    immersion_coords = rng.standard_normal((n, 3))

    dg = DiffusionGeometry.from_edges(
        edge_index=edge_index,
        immersion_coords=immersion_coords,
        n_coefficients=4,
    )

    assert dg.n == n
    assert dg.n_function_basis == n
    assert dg.n_coefficients == 4


def test_from_point_cloud_rejects_unknown_regularisation_method():
    data, _, _, _, _ = _build_knn_inputs()
    with pytest.raises(ValueError, match="Unknown regularisation method"):
        DiffusionGeometry.from_point_cloud(
            data_matrix=data,
            regularisation_method="invalid-method",
            knn_kernel=24,
            knn_bandwidth=8,
            n_function_basis=20,
        )


def test_from_point_cloud_small_dataset_with_default_n_function_basis():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((20, 2))

    dg = DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        knn_kernel=10,
        knn_bandwidth=5,
    )

    assert dg.n == 20
    assert dg.n_function_basis == 20
    assert dg.function_basis.shape == (20, 20)
