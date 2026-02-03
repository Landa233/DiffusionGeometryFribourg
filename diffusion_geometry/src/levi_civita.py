from opt_einsum import contract


def levi_civita_02_weak(
    u, gamma_mixed, gamma_coords, hessian_coords, measure, n_coefficients
):
    """
    Computes the weak formulation of the Levi-Civita connection on vector fields.

    ∇ : 𝔛(M) → Ω¹(M) ⊗ Ω¹(M)
    ⟨ ∇(φ_i ∇x_j), φ_I dx_k ⊗ dx_l ⟩
    = ∫ φ_I Γ(x_k, φ_i) Γ(x_l, x_j) dμ + ∫ φ_i φ_I H(x_j)(∇x_k, ∇x_l) dμ

    Parameters
    ----------
    u : array
        Values of the coefficient functions φ_i.
        Shape: [n, n0]
    gamma_mixed : array
        Mixed carré du champ Γ(x_j, φ_i).
        Shape: [n, d, n0]
    gamma_coords : array
        Coordinate carré du champ Γ(x_i, x_j).
        Shape: [n, d, d]
    hessian_coords : array
        Coordinate Hessian H(x_K)(∇x_i, ∇x_j).
        Shape: [n, d, d, d]
    measure : array
        Measure μ.
        Shape: [n]
    n_coefficients : int
        Number of coefficient functions (n1).

    Returns
    -------
    LC_w : matrix
        Weak Levi-Civita connection matrix ∇^{weak}_{Ikl, ij}.
        Shape: [n1 * d², n1 * d]
    """
    _, dim, _ = gamma_mixed.shape
    n1 = n_coefficients

    # term1: ∫ φ_i Γ(x_j, φ_I) Γ(x_k, x_J) dμ
    # Shape: (n, n1), (n, d, n1), (n, d, d), (n) -> (n1, d, d, n1, d)
    # p: point index, i: output coeff, j: output coord1, k: output coord2, I: input coeff, J: input vector coord
    term1 = contract(
        "pi,pjI,pkJ,p->ijkIJ", u[:, :n1], gamma_mixed[:, :, :n1], gamma_coords, measure
    )
    # term2: ∫ φ_i φ_I H(x_J)(∇x_k, ∇x_l) dμ
    # Shape: (n, n1), (n, n1), (n, d, d, d), (n) -> (n1, d, d, n1, d)
    # p: point index, i: output coeff, I: input coeff, j, k: output coords, J: input vector coord
    term2 = contract(
        "pi,pI,pjkJ,p->ijkIJ", u[:, :n1], u[:, :n1], hessian_coords, measure
    )
    LC_w = term1 + term2

    return LC_w.reshape(n1 * dim**2, n1 * dim)


# NOT USED:
# operator form of the Levi-Civita connection (1,1)-tensor
def levi_civita_11_weak(
    u, gamma_mixed, gamma_coords, hessian_coords, mu, n_coefficients
):

    # Input:
    # u: [n, n0] array of coefficient functions.
    # n_coefficients: number of coefficient functions to retain.
    # gamma_mixed: [n, d, n0] carre du champ tensor of the data coordinates and coefficient functions.
    # gamma_coords: [n, d, d] carre du champ tensor of the data coordinates.
    # hessian_coords: [n, d, d, d] 'coord/coord' composed carre du champ tensor.
    # mu: [n] array of sample densities.

    # Return:
    # LC_11: [n1*d, n1*d, n1*d] weak formulation of the Levi-Civita connection (1,1)-tensor.

    _, dim, _ = gamma_mixed.shape
    n1 = n_coefficients

    # Compute the two terms of the connection as 6-tensors of shape [n1, d, n1, d, n1, d].
    lie_term1 = contract(
        "pi,pI,pjs,pJt,p->ijIJst",
        u[:, :n1],
        u[:, :n1],
        gamma_mixed[:, :, :n1],
        gamma_coords,
        mu,
    )  # (n1, d, n1, d, n1, d)
    lie_term2 = contract(
        "pi,pI,ps,pjtJ,p->ijIJst", u[:, :n1], u[:, :n1], u[:, :n1], hessian_coords, mu
    )  # (n1, d, n1, d, n1, d)

    LC_11 = lie_term1 + lie_term2
    return LC_11.reshape(n1 * dim, n1 * dim, n1 * dim)
