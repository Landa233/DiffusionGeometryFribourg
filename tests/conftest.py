import numpy as np
import pytest
from scipy.special import comb

from diffusion_geometry.core import DiffusionGeometry


SAMPLE_COUNT = 20


def sample_indices(n, m=SAMPLE_COUNT, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    m = min(m, n)
    return np.sort(rng.choice(n, size=m, replace=False))


def flat_idx_n1Ck(i_sel, Ck):
    # shape: [len(i_sel)*Ck]
    return (i_sel[:, None] * Ck + np.arange(Ck)[None, :]).ravel()


def flat_idx_n1d(i_sel, d):
    return (i_sel[:, None] * d + np.arange(d)[None, :]).ravel()


def flat_idx_n1d2(i_sel, d):
    return (i_sel[:, None] * (d * d) + np.arange(d * d)[None, :]).ravel()


TEST_CONFIGS = [
    (1, 60, 12, 5),
    (2, 80, 16, 1),
    (3, 120, 24, 24),
    (4, 100, 20, 10),
]


@pytest.fixture(params=TEST_CONFIGS)
def setup_geom(request):
    d, n, n0, n1 = request.param
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, d))
    # Ensure KNN parameters remain within valid bounds for small sample sizes.
    knn_kernel = min(32, max(1, n - 1))
    knn_bandwidth = min(8, knn_kernel)
    return DiffusionGeometry.from_point_cloud(
        data_matrix=data,
        immersion_coords=data,
        n_function_basis=n0,
        n_coefficients=n1,
        knn_kernel=knn_kernel,
        knn_bandwidth=knn_bandwidth,
    )
