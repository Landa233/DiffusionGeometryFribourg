from opt_einsum import contract
from .basis_utils import kp1_children_and_signs
import numpy as np


def derivative_weak(u, gamma_mixed, compound_matrices_k, measure, k, n_coefficients):
    """
    Computes the weak exterior derivative of k-forms.

    d : Ωᵏ(M) ↦ Ωᵏ⁺¹(M)
    ⟨ d(φ_i dx_J), φ_I dx_{J'} ⟩ = ∫ φ_I det(Γ(dφ_i | dx_J, dx_{J'})) dμ

    p: sample points (performing the integration)
    i, I: coefficient function indices
    j, J: form component indices (multi-indices)
    r: terms in the Laplace expansion

    Parameters
    ----------
    u : array
        Values of the coefficient functions φ_i.
        Shape: [n, n0]
    gamma_mixed : array
        Mixed carré du champ Γ(x_j, φ_i) between coordinates and n_function_basis coefficient functions.
        Shape: [n, d, n_function_basis]
    compound_matrices_k : array
        Precomputed k-th compound determinants of Γ_coords. Required for k > 0.
        det(Γ(x_J, x_{J'})).
        Shape: [n, Ck, Ck] where Ck = comb(d, k)
    measure : array
        Measure μ.
        Shape: [n]
    k : int
        Degree of the form (0 <= k < d).
    n_coefficients : int
        Number of coefficient functions to retain.

    Returns
    -------
    d_w : matrix
        Weak exterior derivative matrix d^{(k),weak}.
        Shape: [n_coefficients * C_{k+1}, n_coefficients * C_k]
    """
    n1 = n_coefficients
    d = gamma_mixed.shape[1]
    assert 0 <= k < d, f"Form degree k={k} must be between 0 and {d-1}"

    if k == 0:
        # ⟨ φ_i dx_j, d φ_I ⟩ = ∫ φ_i Γ(x_j, φ_I) dμ
        # Shape: (n, n1), (n, d, n0), (n) -> (n1, d, n0)
        # p: point index, i: output coeff index (φ_i), j: coord index, I: input coeff index (φ_I)
        d0_w = contract("pi,pjI,p->ijI", u[:, :n1], gamma_mixed, measure)
        n0 = gamma_mixed.shape[2]
        return d0_w.reshape(n1 * d, n0)

    else:
        # Get cached helper data for Laplace expansion
        idx_k, idx_kp1, children, signs = kp1_children_and_signs(d, k)
        Ck, Ckp1 = idx_k.shape[0], idx_kp1.shape[0]

        # Gather tensors needed for the expansion
        # Γ_mix rows for each (k+1)-form basis element J'
        # Shape: (n, Ckp1, k+1, n1) -> [p, J', r, i]
        gamma_rows = np.take(gamma_mixed[:, :, :n1], idx_kp1, axis=1)

        # Minor determinants det(Γ_coords[rows=J'\{j_r}, cols=J]) gathered via children indices
        # Shape: (n, Ckp1, k+1, Ck) -> [p, J', r, j]
        det_minors = np.take(compound_matrices_k, children, axis=1)

        # Combined integration and Laplace expansion.
        # This single contraction sums over p and r to compute the weak matrix:
        # ⟨ φ_I dx_{J'} , d(φ_i dx_j) ⟩ =
        # ∫ φ_I ∑_r (-1)^{r+1} Γ(x_{j'_r}, φ_i) det(Γ(x_{J' \setminus \{j'_r\}}, x_j)) dμ
        #
        # Shape: (n, n1), (n, Ckp1, k+1, n1), (n, Ckp1, k+1, Ck), (n), (k+1) -> (n1, Ckp1, n1, Ck)
        # p: point index, I: output coeff index, J: output index (J'), r: Laplace term, i: input coeff, j: input index (J)
        d_w = contract(
            "pI,pJri,pJrj,p,r->IJij",
            u[:, :n1],
            gamma_rows,
            det_minors,
            measure,
            signs,
        )

        return d_w.reshape(n1 * Ckp1, n1 * Ck)
