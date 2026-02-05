from opt_einsum import contract
from .basis_utils import get_wedge_basis_indices
import numpy as np


def up_delta_weak(
    gamma_functions,
    gamma_mixed,
    gamma_coords,
    gamma_submatrices,
    compound_matrices,
    measure,
    k,
    n_coefficients=None,
):
    """
    Computes the weak up-Laplacian (up-Hodge energy) matrix on k-forms.

    Δ_up = δ d : Ωᵏ(M) ↦ Ωᵏ(M)
    ⟨ d(φ_i dx_J), d(φ_I dx_{J'}) ⟩ = ∫ g(d(φ_i dx_J), d(φ_I dx_{J'})) dμ

    Evaluation of the Schur determinant formula to save computation:
    det([[a, b],[c, D]]) = det(D) * (a - b D⁻¹ c)

    Parameters
    ----------
    gamma_functions : array
        Carré du champ among coefficient functions Γ(φ_i, φ_I).
        Shape: [n, n0, n0]
    gamma_mixed : array
        Mixed carré du champ between coordinates and coefficient functions Γ(x_j, φ_i).
        Shape: [n, d, n0]
    gamma_coords : array
        Coordinate carré du champ matrices Γ(x_i, x_j).
        Shape: [n, d, d]
    gamma_submatrices : array
        Pre-computed submatrices of gamma_coords.
        Shape: [n, Ck, Ck, k, k]
    compound_matrices : array
        Pre-computed compound matrices of gamma_coords (determinants of submatrices).
        Shape: [n, Ck, Ck]
    measure : array
        Measure μ.
        Shape: [n]
    k : int
        Degree of the form (0 <= k < d).
    n_coefficients : int, optional
        Number of coefficient functions to use for k > 0.

    Returns
    -------
    up_delta_w : matrix
        Weak up-Laplacian operator matrix.
        Shape: [n1 * C_k, n1 * C_k]
    """
    n, dim, _ = gamma_mixed.shape
    assert 0 <= k < dim, f"Form degree k={k} must be between 0 and {dim-1}"

    if k == 0:
        # For 0-forms, up_w = ⟨dφ_i, dφ_I⟩ = ∫ Γ(φ_i, φ_I) dμ
        # Shape: (n), (n, n0, n0) -> (n0, n0)
        # p: point index, i, I: coefficient function indices
        return contract("p,piI->iI", measure, gamma_functions)

    assert n_coefficients is not None, "n_coefficients must be provided for k > 0"
    n1 = n_coefficients

    if k == 1:
        # When k=1, the Schur determinant formula simplifies to the standard
        # det([[a, b],[c, d]]) = ad - bc
        # Compute the two terms separately (n1, dim, n1, dim).

        # term1: ∫ ad dμ = ∫ Γ(φ_i, φ_I) Γ(x_j, x_k) dμ
        # Shape: (n), (n, n1, n1), (n, d, d) -> (n1, d, n1, d)
        # p: point index, i, l: coefficient function indices, k, j: coordinate indices
        term1 = contract(
            "p,pik,pjl->ijkl", measure, gamma_functions[:, :n1, :n1], gamma_coords
        )

        # term2: ∫ bc dμ = ∫ Γ(x_l, φ_i) Γ(x_j, φ_I) dμ
        # Shape: (n), (n, d, n1), (n, d, n1) -> (n1, d, n1, d)
        # p: point index, i, k: output/input coeff indices, l, j: coordinate indices
        term2 = contract(
            "p,pli,pjk->ijkl", measure, gamma_mixed[:, :, :n1], gamma_mixed[:, :, :n1]
        )
        integral = term1 - term2
        return integral.reshape(n1 * dim, n1 * dim)

    # Schur complement with respect to D (k×k):
    # For fixed (p, i, I, j, J), with a = Γ_n_[p,i,I] (scalar in (i,I)),
    # det([[a, b],[c, D]]) = det(D) * (a - b D⁻¹ c).
    idx_k = get_wedge_basis_indices(dim, k)  # (Ck, k)
    Ck = idx_k.shape[0]

    # Use pre-computed determinants (compounds) of gamma submatrices
    D = gamma_submatrices  # (n, Ck, Ck, k, k)
    detD = compound_matrices  # (n, Ck, Ck)

    # Build b[p, i, J, r] and c[p, j, s, I] from Γ_mix
    mixed_lookup = np.take(gamma_mixed[:, :, :n1], idx_k, axis=1)  # (n, Ck, k, n1)
    b = np.transpose(mixed_lookup, (0, 3, 1, 2))  # (n, n1, Ck, k) -> [p, i, J, r]
    c = mixed_lookup  # (n, Ck, k, n1) -> [p, j, s, I]

    # Solve D_{j,J} * v = c_j for v.
    # Add J axis and broadcast -> [p, j, J, s, I]
    c_broadcast = np.broadcast_to(c[:, :, None, :, :], (n, Ck, Ck, k, n1))
    V = np.linalg.solve(D, c_broadcast)  # (n, Ck, Ck, k, n1) -> [p, j, J, s, I]

    # Compute the two terms in the Schur determinant formula (n1, Ck, n1, Ck).
    detD_measure = detD * measure[:, None, None]  # (n, Ck, Ck)

    # term1: ∫ det(D) a dμ = ∫ Γ(φ_i, φ_I) det(Γ(x_J, x_{J'})) dμ
    # Shape: (n, n1, n1), (n, Ck, Ck) -> (n1, Ck, n1, Ck)
    # p: point index, i, I: coefficient function indices, j, J: compound block indices (Ck)
    term1 = contract("piI,pjJ->ijIJ", gamma_functions[:, :n1, :n1], detD_measure)

    # term2: ∫ det(D) b D⁻¹ c dμ = ∫ det(D) (Γ(x, φ) D⁻¹ Γ(x, φ)) dμ
    # Shape: (n, Ck, Ck), (n, n1, Ck, k), (n, Ck, Ck, k, n1) -> (n1, Ck, n1, Ck)
    # p: point index, j, J: compound block indices, i, I: coefficient function indices, r: inner sum index (k) for V
    term2 = contract("pjJ,piJr,pjJrI->ijIJ", detD_measure, b, V)
    integral = term1 - term2

    return integral.reshape(n1 * Ck, n1 * Ck)
