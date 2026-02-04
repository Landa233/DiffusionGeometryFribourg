import numpy as np

from tests.helpers import sample_indices, flat_idx_n1d


def test_levi_civita_02_weak(setup_geom):
    dg = setup_geom
    # backend = dg.backend  # Use backend for backend-specific tests
    n1, d = dg.n_coefficients, dg.dim
    lc_02_weak_computed = dg.levi_civita.weak
    u_n1, gamma_mixed_n1, gamma_coords, hessian_coords, measure = (
        dg.function_basis[:, :n1],
        dg.cache.gamma_mixed[:, :, :n1],
        dg.cache.gamma_coords,
        dg.cache.hessian_coords,
        dg.measure,
    )

    # Select subsets of indices for manual verification
    i_sel = sample_indices(n1)
    j1_sel = sample_indices(d)
    j2_sel = sample_indices(d)
    jprime_sel = sample_indices(d)

    # Manual assembly following the weak (0,2)-tensor formula
    # Shape: [len(i_sel), len(j1_sel), len(j2_sel), len(i_sel), len(jprime_sel)]
    lc_sel = np.zeros(
        (
            len(i_sel),
            len(j1_sel),
            len(j2_sel),
            len(i_sel),
            len(jprime_sel),
        )
    )

    for a, i in enumerate(i_sel):
        for b, j1 in enumerate(j1_sel):
            for c, j2 in enumerate(j2_sel):
                for ap, ip in enumerate(i_sel):
                    for bp, jp in enumerate(jprime_sel):
                        # Integral over p: mu[p] * [ U_{pi} * Γ_mix(p, j1, ip) * Γ_coord(p, j2, jp)
                        #                               + U_{pi} * U_{pip} * H_coords(p, j1, j2, jp) ]
                        val = 0.0
                        for p in range(dg.n):
                            val += (
                                u_n1[p, i]
                                * gamma_mixed_n1[p, j1, ip]
                                * gamma_coords[p, j2, jp]
                                * measure[p]
                            )
                            val += (
                                u_n1[p, i]
                                * u_n1[p, ip]
                                * hessian_coords[p, j1, j2, jp]
                                * measure[p]
                            )
                        lc_sel[a, b, c, ap, bp] = val

    # Compare with selected entries of the computed matrix
    # Build row/col indices matching our selected (i,j1,j2) and (ip,jp)
    rows_idx = (
        (i_sel[:, None, None] * (d * d))
        + (j1_sel[None, :, None] * d)
        + (j2_sel[None, None, :])
    ).reshape(-1)
    cols_idx = ((i_sel[:, None] * d) + (jprime_sel[None, :])).reshape(-1)

    lc_comp_sub = lc_02_weak_computed[np.ix_(rows_idx, cols_idx)]

    # Flatten manual tensor to match the submatrix ordering
    lc_sel_flat = lc_sel.reshape(
        len(i_sel) * len(j1_sel) * len(j2_sel),
        len(i_sel) * len(jprime_sel),
    )

    assert np.allclose(lc_comp_sub, lc_sel_flat)
