import numpy as np
import pytest
from scipy.special import comb
from itertools import combinations

from tests.helpers import sample_indices, flat_idx_n1Ck, flat_idx_n1d


@pytest.mark.parametrize("k", list(range(0, 7)))
def test_derivative_weak_k(setup_geom, k):
    dg = setup_geom
    # backend = dg.backend  # Use backend for backend-specific tests
    n, n0, n1, d = dg.n, dg.n_function_basis, dg.n_coefficients, dg.dim
    if k >= d:
        pytest.skip(f"k={k} is not smaller than dimension d={d}")
    d_weak_computed = dg.d(k).weak
    u, measure, gamma_coords = dg.function_basis, dg.measure, dg.backend.gamma_coords
    if k == 0:
        d_weak_manual = np.einsum(
            "pi,pjI,p->ijI", u[:, :n1], dg.backend.gamma_mixed, measure
        ).reshape(n1 * d, n0)
        i_sel = sample_indices(n1)
        I_sel = flat_idx_n1d(i_sel, d)
        assert np.allclose(d_weak_computed[I_sel], d_weak_manual[I_sel], atol=1e-6)
    else:
        _, compound_matrices = dg.backend.gamma_coords_compound(k)
        gamma_mixed = dg.backend.gamma_mixed[:, :, :n1]
        u_n_rows = u[:, :n1]
        Ck, Ck1 = int(comb(d, k)), int(comb(d, k + 1))
        Js, J_primes = list(combinations(range(d), k)), list(
            combinations(range(d), k + 1)
        )
        i_sel = sample_indices(n1)
        I_rows = flat_idx_n1Ck(i_sel, Ck1)
        I_cols = flat_idx_n1Ck(i_sel, Ck)
        d_weak_manual_sel = np.zeros((len(I_rows), len(I_cols)))
        for a, i_prime_idx in enumerate(i_sel):
            for J_prime_idx, J_prime in enumerate(J_primes):
                for b, i_idx in enumerate(i_sel):
                    for J_idx, J in enumerate(Js):
                        val = 0.0
                        for p in range(n):
                            det_matrix = np.zeros((k + 1, k + 1))
                            det_matrix[:, 0] = gamma_mixed[p, list(J_prime), i_idx]
                            det_matrix[:, 1:] = gamma_coords[p][
                                np.ix_(list(J_prime), list(J))
                            ]
                            val += (
                                u_n_rows[p, i_prime_idx]
                                * np.linalg.det(det_matrix)
                                * measure[p]
                            )
                        row_idx = a * Ck1 + J_prime_idx
                        col_idx = b * Ck + J_idx
                        d_weak_manual_sel[row_idx, col_idx] = val
        assert np.allclose(
            d_weak_computed[np.ix_(I_rows, I_cols)], d_weak_manual_sel, atol=1e-6
        )
