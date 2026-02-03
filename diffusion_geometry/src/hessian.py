from opt_einsum import contract
import numpy as np
from .basis_utils import get_symmetric_basis_indices


def hessian_functions(u, coords, gamma_coords, gamma_mixed, cdc):
    """
    Computes the pointwise Hessian tensor H(φ_I)(∇x_i, ∇x_j).
    H(f)(∇x_i, ∇x_j) = 1/2 ( Γ(x_i, Γ(x_j, f)) + Γ(x_j, Γ(x_i, f)) - Γ(f, Γ(x_i, x_j)) )

    Parameters
    ----------
    u : array
        Values of the coefficient functions φ_I.
        Shape: [n, n0]
    coords : array
        Coordinates x_i.
        Shape: [n, d]
    gamma_coords : array
        Coordinate carré du champ Γ(x_i, x_j).
        Shape: [n, d, d]
    gamma_mixed : array
        Mixed carré du champ Γ(x_i, φ_I).
        Shape: [n, d, n0]
    cdc : callable
        Function to compute carré du champ Γ(f, h).

    Returns
    -------
    H : array
        Pointwise Hessian tensor.
        Shape: [n, d, d, n0]
    """

    # Γ(x_i, Γ(x_j, φ_I))
    gamma_composed_1 = cdc(coords, gamma_mixed)  # (n, d, d, n0)

    # Γ(Γ(x_i, x_j), φ_I)
    gamma_composed_2 = cdc(gamma_coords, u)  # (n, d, d, n0)

    # Compute the Hessian as a 4-tensor of shape [n, d, d, n0].
    H = 0.5 * (
        gamma_composed_1  # Γ(x_i, Γ(x_j, φ_I))
        + gamma_composed_1.transpose((0, 2, 1, 3))  # Γ(x_j, Γ(x_i, φ_I))
        - gamma_composed_2  # Γ(Γ(x_i, x_j), φ_I)
    )

    return H


def hessian_coords(coords, gamma_coords, cdc):
    """
    Computes Hessian of the coordinates H(x_k)(∇x_i, ∇x_j).

    Returns
    -------
    H : array
        Pointwise coordinate Hessian tensor.
        Shape: [n, d, d, d]
    """

    # Γ(x_i, Γ(x_j, x_k))
    gamma_composed = cdc(coords, gamma_coords)  # (n, d, d, d)

    # Compute the Hessian as a 4-tensor of shape [n, d, d, d].
    H = 0.5 * (
        gamma_composed  # Γ(x_i, Γ(x_j, x_k))
        + gamma_composed.transpose((0, 2, 1, 3))  # Γ(x_j, Γ(x_i, x_k))
        - gamma_composed.transpose((0, 3, 1, 2))  # Γ(x_k, Γ(x_i, x_j))
    )

    return H


def hessian_02_weak(u, hessian_matrix, measure, n_coefficients):
    """
    Computes the weak formulation of the Hessian operator matrix on functions.

    H : A ↦ Ω¹(M) ⊗ Ω¹(M)
    ⟨ H(φ_i), φ_I dx_j ⊗ dx_k ⟩ = ∫ φ_I H(φ_i)(∇x_j, ∇x_k) dμ

    Parameters
    ----------
    u : array
        Values of the coefficient functions φ_I.
        Shape: [n, n0]
    hessian_matrix : array
        Pointwise Hessian tensor H(φ_i)(∇x_j, ∇x_k).
        Shape: [n, d, d, n0]
    measure : array
        Measure μ.
        Shape: [n]
    n_coefficients : int
        Number of coefficient functions (n1).

    Returns
    -------
    H_w : matrix
        Weak Hessian operator matrix H^{(0,2),weak}.
        Shape: [n1 * d * d, n0]
    """
    _, _, dim, n0 = hessian_matrix.shape
    n1 = n_coefficients

    # ⟨ H(φ_I), φ_i dx_j ⊗ dx_k ⟩ = ∫ φ_i H(φ_I)(∇x_j, ∇x_k) dμ
    # Shape: (n, d, d, n0), (n, n1), (n) -> (d, d, n1, n0)
    # p: point index, j, k: coord indices, I: input coeff index, i: test coeff index
    H_w = contract(
        "pjkI,pi,p->ijkI", hessian_matrix, u[:, :n1], measure
    )  # [n1, d, d, n0]

    return H_w.reshape(n1 * dim**2, n0)


def hessian_02_sym_weak(u, hessian_matrix, measure, n_coefficients):
    """
    Weak formulation of the Hessian as a symmetric (0,2)-tensor.

    The off-diagonal entries are weighted by 2 to align with the
    symmetric basis convention.

    Parameters
    ----------
    u : array
        Values of the coefficient functions φ_I.
        Shape: [n, n0]
    hessian_matrix : array
        Pointwise Hessian tensor H(φ_i)(∇x_j, ∇x_k).
        Shape: [n, d, d, n0]
    measure : array
        Measure μ.
        Shape: [n]
    n_coefficients : int
        Number of coefficient functions (n1).

    Returns
    -------
    H_sym : matrix
        Weak symmetric Hessian operator matrix.
        Shape: [n1 * d_sym, n0]
    """
    _, _, d, n0 = hessian_matrix.shape
    n1 = n_coefficients
    sym_idx = get_symmetric_basis_indices(d)  # (d_sym, 2)
    d_sym = sym_idx.shape[0]
    j1, j2 = sym_idx[:, 0], sym_idx[:, 1]

    # Extract upper-triangular entries
    hessian_matrix_sym = hessian_matrix[:, j1, j2, :]  # (n, d_sym, n0)

    # Double off-diagonals to account for multiplicity in the weak formulation
    diag_mask = j1 == j2
    weights = np.where(diag_mask, 1.0, 2.0).astype(hessian_matrix.dtype)
    hessian_matrix_sym = hessian_matrix_sym * weights[None, :, None]

    # Contract to form weak matrix: integrate φ_i * H_sym[p,S,I] * μ_p
    # ⟨ H(φ_I), φ_i dx_{S} ⟩ = ∫ φ_i H_{sym}(φ_I)_S dμ
    # Shape: (n, d_sym, n0), (n, n1), (n) -> (n1, d_sym, n0)
    # p: point index, S: symmetric basis index, I: input coeff index, i: test coeff index
    H_sym = contract("pSI,pi,p->iSI", hessian_matrix_sym, u[:, :n1], measure)

    return H_sym.reshape(n1 * d_sym, n0)


# NOT USED:
# Direct formulation of the Hessian as a bilinear operator on vector fields.
def hessian_operator(u, hessian_matrix, n_coefficients):

    # Input:
    # u: [n, n0] array of coefficient functions.
    # n_coefficients: number of coefficient functions to retain.
    # hessian_matrix: [n, d, d, n0] Hessian matrix tensor.

    # Return:
    # H: [n, n1*d, n1*d, n0] Hessian operator tensor.

    n, _, dim, n0 = hessian_matrix.shape
    n1 = n_coefficients

    H = contract(
        "pjkI,pa,pb->pajbkI", hessian_matrix, u[:, :n1], u[:, :n1]
    )  # [n, n1, d, n1, d, n0]

    return H.reshape(n, n1 * dim, n1 * dim, n0)
