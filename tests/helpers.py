import numpy as np
from scipy.special import comb

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


def flat_idx_n1d_sym(i_sel, d):
    d_sym = d * (d + 1) // 2
    return (i_sel[:, None] * d_sym + np.arange(d_sym)[None, :]).ravel()
