import numpy as np
import pytest

from diffusion_geometry.utils.basis_utils import (
    get_wedge_basis_indices,
    kp1_children_and_signs,
)
from diffusion_geometry.core import gamma_compound


def test_get_wedge_basis_indices_content():
    d, k = 4, 2
    idx = get_wedge_basis_indices(d, k)
    from itertools import combinations as cmb

    expected = np.array(list(cmb(range(d), k)))
    assert idx.shape == (len(expected), k)
    assert np.array_equal(idx, expected)


def testkp1_children_and_signs_semantics():
    d, k = 4, 2
    idx_k, idx_kp1, children, signs = kp1_children_and_signs(d, k)
    assert np.array_equal(signs, (-1) ** np.arange(k + 1))
    mapping = {tuple(t): i for i, t in enumerate(map(tuple, idx_k))}
    for Jp_idx, Jp in enumerate(idx_kp1):
        for r in range(k + 1):
            child = tuple([Jp[c] for c in range(k + 1) if c != r])
            assert children[Jp_idx, r] == mapping[child]

def test_gamma_compound_cases():
    rng = np.random.default_rng(7)
    n, d = 3, 3
    A = rng.normal(size=(n, d, d))
    gamma = np.einsum("pij,pkj->pik", A, A) + 0.5 * np.eye(d)
    sub0, det0 = gamma_compound(gamma, 0)
    assert sub0.shape == (n, 1, 1, 0, 0)
    assert np.allclose(det0, np.ones((n, 1, 1)))
    sub1, det1 = gamma_compound(gamma, 1)
    assert sub1.shape == (n, d, d, 1, 1)
    assert np.allclose(det1, gamma)
    subd, detd = gamma_compound(gamma, d)
    assert subd.shape == (n, 1, 1, d, d)
    assert np.allclose(detd[:, 0, 0], np.linalg.det(gamma))
    if d >= 3:
        sub2, det2 = gamma_compound(gamma, 2)
        from itertools import combinations as cmb

        idx = list(cmb(range(d), 2))
        for p in range(n):
            for a, Ja in enumerate(idx):
                for b, Jb in enumerate(idx):
                    minor = gamma[p][np.ix_(Ja, Jb)]
                    assert np.isclose(det2[p, a, b], np.linalg.det(minor))
