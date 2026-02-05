from opt_einsum import contract


def lie_bracket_weak(u, coords, gamma_coords, measure, n_coefficients, cdc):
    """
    Computes the weak Lie bracket matrix of vector fields.

    [X, Y] : 𝔛(M) × 𝔛(M) ↦ 𝔛(M)
    ⟨ [φ_{i1} ∇x_{j1}, φ_{i2} ∇x_{j2}], φ_{I} ∇x_{J} ⟩
    = ∫ φ_{I} ( φ_{i1} Γ(x_{j1}, φ_{i2} Γ(x_{j2}, x_{J})) - φ_{i2} Γ(x_{j2}, φ_{i1} Γ(x_{j1}, x_{J})) ) dμ

    Parameters
    ----------
    u : array
        Values of the coefficient functions φ_i.
        Shape: [n, n0]
    coords : array
        Coordinates x_j.
        Shape: [n, d]
    gamma_coords : array
        Coordinate carré du champ Γ(x_i, x_j).
        Shape: [n, d, d]
    measure : array
        Measure μ.
        Shape: [n]
    n_coefficients : int
        Number of coefficient functions (n1).
    cdc : callable
        Function to compute carré du champ Γ(f, h).

    Returns
    -------
    lie_w : matrix
        Weak Lie bracket operator matrix Lie^{weak}_{IJ, i1j1, i2j2}.
        Shape: [n1 * d, n1 * d, n1 * d]
    """
    _, dim = coords.shape
    n1 = n_coefficients

    # Create the product φ_I Γ(x_J, x_t): [n, n1, d, d]
    product = u[:, :n1][:, :, None, None] * gamma_coords[:, None, :, :]
    # Γ(x_j, φ_I Γ(x_J, x_t))
    gamma_comp_lie = cdc(coords, product)  # [n, d, n1, d, d]

    # Compute the weak Lie bracket as a 6-tensor of shape [n1, d, n1, d, n1, d].
    # ∫ φ_s φ_i Γ(x_j, φ_I Γ(x_J, x_t)) dμ
    # Shape: (n, n1), (n, n1), (n, d, n1, d, d), (n) -> (n1, d, n1, d, n1, d)
    # p: point index, s: test coeff (φ_I), t: test coord (x_J), i: coeff1, j: coord1, I: coeff2, J: coord2
    lie = contract(
        "ps,pi,pjIJt,p->stijIJ",
        u[:, :n1],
        u[:, :n1],
        gamma_comp_lie,
        measure,
    )
    lie -= lie.transpose((0, 1, 4, 5, 2, 3))

    return lie.reshape(n1 * dim, n1 * dim, n1 * dim)
