import numpy as np
import pytest
from scipy.special import comb

from tests.helpers import (
    sample_indices,
    flat_idx_n1Ck,
    flat_idx_n1d2,
    flat_idx_n1d_sym,
)


@pytest.mark.parametrize("k", list(range(0, 7)))
def test_G_k(setup_geom, k):
    dg = setup_geom
    # backend = dg.backend  # Use the backend for backend-specific tests
    n, n1, d = dg.n, dg.n_coefficients, dg.dim

    if k == 0:
        G_computed = dg.function_space.gram
        G_manual = (
            dg.measure[:, None, None]
            * dg.function_basis[:, :, None]
            * dg.function_basis[:, None, :]
        ).sum(axis=0)
        i0_sel = sample_indices(dg.n_function_basis)
        assert np.allclose(
            G_computed[np.ix_(i0_sel, i0_sel)],
            G_manual[np.ix_(i0_sel, i0_sel)],
        )
        return

    if k > d:
        pytest.skip(f"k={k} is larger than dimension d={d}")

    G_computed = dg.form_space(k).gram
    Ck = int(comb(d, k))
    i_sel = sample_indices(n1)
    I_sel = flat_idx_n1Ck(i_sel, Ck)
    g_k = dg.form_space(k).metric
    measure = dg.measure
    G_manual = (g_k * measure[:, None, None]).sum(axis=0)
    assert np.allclose(G_computed[np.ix_(I_sel, I_sel)], G_manual[np.ix_(I_sel, I_sel)])


def test_G02(setup_geom):
    dg = setup_geom
    # backend = dg.backend  # Use the backend for backend-specific tests
    n, n1, d = dg.n, dg.n_coefficients, dg.dim
    G02_computed = dg.tensor02_space.gram
    g02 = dg.tensor02_space.metric
    measure = dg.measure

    i_sel = sample_indices(n1)
    I_sel = flat_idx_n1d2(i_sel, d)
    G02_manual = (g02 * measure[:, None, None]).sum(axis=0)
    assert np.allclose(
        G02_computed[np.ix_(I_sel, I_sel)], G02_manual[np.ix_(I_sel, I_sel)]
    )


def test_G02_sym_matrix(setup_geom):
    dg = setup_geom
    # backend = dg.backend  # Use the backend for backend-specific tests
    n1, d = dg.n_coefficients, dg.dim
    G_sym_computed = dg.tensor02sym_space.gram
    g_sym = dg.tensor02sym_space.metric
    measure = dg.measure

    i_sel = sample_indices(n1)
    I_sel = flat_idx_n1d_sym(i_sel, d)
    G_sym_manual = (g_sym * measure[:, None, None]).sum(axis=0)
    assert np.allclose(
        G_sym_computed[np.ix_(I_sel, I_sel)], G_sym_manual[np.ix_(I_sel, I_sel)]
    )
