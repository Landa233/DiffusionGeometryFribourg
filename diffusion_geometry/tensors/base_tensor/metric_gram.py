from opt_einsum import contract


def metric(u_n1, matrices):
    """
    Unified metric tensor computation for all tensor types.

    Parameters
    ----------
    u_n1 : array
        Values of the coefficient functions at each point.
        Shape: [n, n1]
    matrices : array
        Precomputed matrices where C varies by tensor type:
        - For k-forms: compound matrices Γ(dx_J, dx_{J'}) with C = comb(d, k)
        - For symmetric (0,2): sym_gamma with C = d*(d+1)/2
        - For full (0,2): gamma_02 with C = d²
        Shape: [n, C, C]

    Returns
    -------
    g : array
        Metric tensor field g(e_a, e_b).
        Shape: [n, n1*C, n1*C]
    """
    n, n1 = u_n1.shape
    C = matrices.shape[1]

    # Compute g as a 5-tensor of shape (n, n1, C, n1, C)
    g = (
        u_n1[:, :, None, None, None]
        * u_n1[:, None, None, :, None]
        * matrices[:, None, :, None, :]
    )

    return g.reshape(n, n1 * C, n1 * C)


def _metric_apply(u_n1, regularise_func, a_coeffs, b_coeffs, matrices):
    """
    Computes inner product between two tensors using factorized metric.

    Parameters
    ----------
    u_n1 : array
        Coefficient functions at each point.
        Shape: [n, n1]
    regularise_func : callable
        Regularisation function for the output.
    a_coeffs : array
        Coefficients of first tensor.
        Shape: [..., n1*C]
    b_coeffs : array
        Coefficients of second tensor.
        Shape: [..., n1*C]
    matrices : array
        Local inner product matrices.
        Shape: [n, C, C]

    Returns
    -------
    array
        Pointwise values of the metric scalar field.
        Shape: [..., n]
    """
    basis_count = u_n1.shape[1]
    components = matrices.shape[1]

    a_view = a_coeffs.reshape(a_coeffs.shape[:-1] + (basis_count, components))
    b_view = b_coeffs.reshape(b_coeffs.shape[:-1] + (basis_count, components))

    # Project coefficients to pointwise tensor values: u^i(x) * c_i
    # Shape: (n, n1) * (..., n1, C) -> (..., n, C)
    a_point = contract("pi,...ic->...pc", u_n1, a_view)

    if a_coeffs is b_coeffs:
        b_point = a_point
    else:
        b_point = contract("pi,...ic->...pc", u_n1, b_view)

    # Compute pointwise inner product using local metric matrices:
    # g(A, B)(p) = A^c(p) * g_cd(p) * B^d(p)
    # Shape: (..., n, C), (n, C, C), (..., n, C) -> (..., n)
    metric_vals = contract("...pc,pcd,...pd->...p", a_point, matrices, b_point)
    return regularise_func(metric_vals)


def gram(u_n1, matrices, measure):
    """
    Computes the Gram matrix of a tensor basis.
    G_{i a, I b} = ⟨ φ_i e_a, φ_I e_b ⟩ = ∫ φ_i φ_I g(e_a, e_b) dμ

    Parameters
    ----------
    u_n1 : array
        Values of the coefficient functions φ_i.
        Shape: [n, n1]
    matrices : array
        Local inner product matrices g(e_a, e_b) at each point.
        - For k-forms: det(Γ(dx_J, dx_K)) where C = comb(d, k)
        - For symmetric (0,2): g(dx_j ⊗ dx_k, dx_l ⊗ dx_m) where C = d*(d+1)/2
        Shape: [n, C, C]
    measure : array
        Measure μ.
        Shape: [n]

    Returns
    -------
    G : matrix
        Gram matrix of the tensor basis G_{i a, I b}.
        Shape: [n1*C, n1*C]
    """
    n1 = u_n1.shape[1]
    C = matrices.shape[1]

    # Compute G as a 4-tensor of shape [n1, C, n1, C]
    # ⟨ φ_i e_a, φ_I e_b ⟩ = ∫ φ_i φ_I g(e_a, e_b) dμ
    # Shape: (n), (n, n1), (n, n1), (n, C, C) -> (n1, C, n1, C)
    # p: point index, i, I: coefficient function indices, a, b: tensor component indices
    G = contract("p,pi,pI,pab->iaIb", measure, u_n1, u_n1, matrices)

    return G.reshape(n1 * C, n1 * C)
