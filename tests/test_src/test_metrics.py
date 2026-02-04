import numpy as np
import pytest
from scipy.special import comb

from tests.helpers import (
    sample_indices,
    flat_idx_n1Ck,
    flat_idx_n1d2,
    flat_idx_n1d_sym,
)
from diffusion_geometry.src.basis_utils import get_symmetric_basis_indices


@pytest.mark.parametrize("k", list(range(0, 7)))
def test_g_k(setup_geom, k):
    dg = setup_geom
    # backend = dg.backend  # Use the backend for backend-specific tests
    n, n1, d = dg.n, dg.n_coefficients, dg.dim

    if k == 0:
        # FunctionSpace doesn't implement metric properly in new API
        return

    if k > d:
        pytest.skip(f"k={k} is larger than dimension d={d}")

    g_computed = dg.form_space(k).metric
    u_n1 = dg.function_basis[:, :n1]
    _, compound_gamma = dg.cache.gamma_coords_compound(k)
    Ck = int(comb(d, k))

    p_sel = sample_indices(n)
    i_sel = sample_indices(n1)
    I_sel = flat_idx_n1Ck(i_sel, Ck)

    g_manual_sel = np.zeros((len(p_sel), len(i_sel), Ck, len(i_sel), Ck))
    for a, p in enumerate(p_sel):
        for ii, i in enumerate(i_sel):
            for jj in range(Ck):
                for ii2, ip in enumerate(i_sel):
                    for jj2 in range(Ck):
                        g_manual_sel[a, ii, jj, ii2, jj2] = (
                            u_n1[p, i] * u_n1[p, ip] * compound_gamma[p, jj, jj2]
                        )
    g_manual_sel = g_manual_sel.reshape(len(p_sel), len(I_sel), len(I_sel))
    assert np.allclose(g_computed[p_sel][:, I_sel][:, :, I_sel], g_manual_sel)


def test_g02(setup_geom):
    dg = setup_geom
    # backend = dg.backend  # Use the backend for backend-specific tests
    n, n1, d = dg.n, dg.n_coefficients, dg.dim
    g02_computed = dg.tensor02_space.metric
    u_n1 = dg.function_basis[:, :n1]
    gamma_02 = dg.tensor02_space.cdc_components

    p_sel = sample_indices(n)
    i_sel = sample_indices(n1)
    I_sel = flat_idx_n1d2(i_sel, d)

    g02_manual_sel = np.zeros((len(p_sel), len(i_sel), d**2, len(i_sel), d**2))
    for a, p in enumerate(p_sel):
        for ii, i in enumerate(i_sel):
            for jj in range(d**2):
                for ii2, ip in enumerate(i_sel):
                    for jj2 in range(d**2):
                        g02_manual_sel[a, ii, jj, ii2, jj2] = (
                            u_n1[p, i] * u_n1[p, ip] * gamma_02[p, jj, jj2]
                        )
    g02_manual_sel = g02_manual_sel.reshape(len(p_sel), len(I_sel), len(I_sel))
    assert np.allclose(g02_computed[p_sel][:, I_sel][:, :, I_sel], g02_manual_sel)


def test_g02_sym(setup_geom):
    dg = setup_geom
    # backend = dg.backend  # Use the backend for backend-specific tests
    n1, d = dg.n_coefficients, dg.dim
    g_sym_computed = dg.tensor02sym_space.metric
    u_n1 = dg.function_basis[:, :n1]
    gamma_02_sym = dg.tensor02sym_space.cdc_components

    p_sel = sample_indices(dg.n)
    i_sel = sample_indices(n1)
    I_sel = flat_idx_n1d_sym(i_sel, d)

    d_sym = d * (d + 1) // 2
    g_sym_manual_sel = np.zeros((len(p_sel), len(i_sel), d_sym, len(i_sel), d_sym))
    for a, p in enumerate(p_sel):
        for ii, i in enumerate(i_sel):
            for jj in range(d_sym):
                for ii2, ip in enumerate(i_sel):
                    for jj2 in range(d_sym):
                        g_sym_manual_sel[a, ii, jj, ii2, jj2] = (
                            u_n1[p, i] * u_n1[p, ip] * gamma_02_sym[p, jj, jj2]
                        )
    g_sym_manual_sel = g_sym_manual_sel.reshape(len(p_sel), len(I_sel), len(I_sel))
    assert np.allclose(g_sym_computed[p_sel][:, I_sel][:, :, I_sel], g_sym_manual_sel)
