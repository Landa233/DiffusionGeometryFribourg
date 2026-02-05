import numpy as np

from tests.helpers import sample_indices, flat_idx_n1d


def test_lie_bracket_weak(setup_geom):
    dg = setup_geom
    # backend = dg.backend  # Removed backend
    n, n1, d = dg.n, dg.n_coefficients, dg.dim
    lie_bracket_weak_computed = dg.lie_bracket.weak
    u_n1, measure, gamma_coords = (
        dg.function_basis[:, :n1],
        dg.measure,
        dg.cache.gamma_coords,
    )
    product = u_n1[:, :, None, None] * gamma_coords[:, None, :, :]
    gamma_comp_lie = dg.triple.cdc(
        dg.immersion_coords,
        product,
    )

    i_sel = sample_indices(n1)
    I_sel = flat_idx_n1d(i_sel, d)
    lie_sel = np.zeros((len(i_sel), d, len(i_sel), d, len(i_sel), d))
    for a, i_prime in enumerate(i_sel):
        for j_prime in range(d):
            for b, i1 in enumerate(i_sel):
                for j1 in range(d):
                    for c, i2 in enumerate(i_sel):
                        for j2 in range(d):
                            val = 0.0
                            for p in range(n):
                                term1 = (
                                    u_n1[p, i1] * gamma_comp_lie[p, j1, i2, j2, j_prime]
                                )
                                term2 = (
                                    u_n1[p, i2] * gamma_comp_lie[p, j2, i1, j1, j_prime]
                                )
                                val += u_n1[p, i_prime] * (term1 - term2) * measure[p]
                            lie_sel[a, j_prime, b, j1, c, j2] = val
    lie_sel = lie_sel.reshape(len(I_sel), len(I_sel), len(I_sel))
    assert np.allclose(lie_bracket_weak_computed[np.ix_(I_sel, I_sel, I_sel)], lie_sel)
