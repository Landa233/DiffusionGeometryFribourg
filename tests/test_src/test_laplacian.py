import numpy as np
import pytest
from scipy.special import comb
from itertools import combinations

from tests.helpers import sample_indices, flat_idx_n1Ck, flat_idx_n1d


@pytest.mark.parametrize("k", list(range(0, 7)))
def test_up_delta_weak_k(setup_geom, k):
    dg = setup_geom
    # backend = dg.backend  # Use backend for backend-specific tests
    n, n0, n1, d = dg.n, dg.n_function_basis, dg.n_coefficients, dg.dim

    if k >= d:
        pytest.skip(f"k={k} is not smaller than dimension d={d}")

    up_delta_weak_computed = dg.up_laplacian(k).weak

    if k == 0:
        up_delta_weak_manual = np.einsum(
            "p,pij->ij", dg.measure, dg.cache.gamma_functions
        )
        assert np.allclose(up_delta_weak_computed, up_delta_weak_manual)
    else:
        gamma_n1, gamma_mixed_n1, gamma_coords, mu = (
            dg.cache.gamma_functions[:, :n1, :n1],
            dg.cache.gamma_mixed[:, :, :n1],
            dg.cache.gamma_coords,
            dg.measure,
        )
        Ck = int(comb(d, k))
        Js = list(combinations(range(d), k))
        i_sel = sample_indices(n1)
        I_sel = flat_idx_n1Ck(i_sel, Ck)
        up_delta_weak_manual_sel = np.zeros((len(I_sel), len(I_sel)))
        for a, i_prime_idx in enumerate(i_sel):
            for J_prime_idx, J_prime in enumerate(Js):
                for b, i_idx in enumerate(i_sel):
                    for J_idx, J in enumerate(Js):
                        val = 0.0
                        for p in range(n):
                            det_matrix = np.zeros((k + 1, k + 1))
                            det_matrix[0, 0] = gamma_n1[p, i_prime_idx, i_idx]
                            det_matrix[0, 1:] = gamma_mixed_n1[p, list(J), i_prime_idx]
                            det_matrix[1:, 0] = gamma_mixed_n1[p, list(J_prime), i_idx]
                            det_matrix[1:, 1:] = gamma_coords[p][
                                np.ix_(list(J_prime), list(J))
                            ]
                            val += np.linalg.det(det_matrix) * mu[p]
                        row_idx = a * Ck + J_prime_idx
                        col_idx = b * Ck + J_idx
                        up_delta_weak_manual_sel[row_idx, col_idx] = val
        assert np.allclose(
            up_delta_weak_computed[np.ix_(I_sel, I_sel)], up_delta_weak_manual_sel
        )
