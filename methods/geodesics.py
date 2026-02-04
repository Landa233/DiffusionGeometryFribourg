import numpy as np
import cvxpy as cp


def geodesic_distances_function(dg, index, reg=False, num_subsample=None):
    """
    Compute an intrinsic geodesic distance function from a fixed source point
    using a convex optimisation enforcing 1-Lipschitz constraints with respect
    to the carré-du-champ metric Γₓ.

    Parameters
    ----------
    dg : DiffusionGeometry
        Diffusion geometry object providing Γₓ and the diffusion basis.
    index : int
        Index of the source point (distance = 0 constraint).
    reg : bool, default=False
        If True, adds a global smoothness regularisation constraint based on
        the Laplacian, enforcing δ²(a + v) <= 1 in average (Laplacian norm ≤ 1).
    num_subsample : int or None, optional
        Number of random points to enforce the pointwise Lipschitz constraint.
        If None, uses all points (exact but slower).

    Returns
    -------
    dist : np.ndarray, shape (n,)
        Estimated intrinsic geodesic distances at all data points.
    v_function : Function
        Correction function in the diffusion basis.

    Notes
    -----
    The optimisation problem solved is:

        maximise_v   (a + v)_0
        subject to   ||Lₓ (a + v)||₂ ≤ 1     for sampled points x
                     ||Δ(a + v)||₂ ≤ 1       (if reg=True)
                     uₓᵀ v = 0               (distance zero at source)

    where each Lₓ is formed from the top-d eigenpairs of Γₓ and Δ is the Laplacian.
    """
    eps = 1e-10  # floor for small/negative eigenvalues

    # Ambient distance
    data = dg.cache.data_matrix
    ambient_dist_pointwise = np.linalg.norm(data - data[index], axis=1)
    ambient_dist = dg.function(ambient_dist_pointwise).coeffs

    # Optimisation variable is the correction
    v = cp.Variable(dg.n_function_basis)
    intrinsic_coeffs = ambient_dist + v

    # Subsample points for Lipschitz constraints
    n_points = dg.n
    if num_subsample is None or num_subsample >= n_points:
        sampled_indices = np.arange(n_points)
    else:
        rng = np.random.default_rng(0)
        sampled_indices = rng.choice(n_points, num_subsample, replace=False)

    # 1-Lipschitz constraints
    L_list = []
    for p in sampled_indices:
        Gp = dg.cache.gamma_functions[p]  # (n0, n0)
        eigvals, eigvecs = np.linalg.eigh(Gp)
        top_d = np.argsort(eigvals)[-dg.dim :]
        eigvals_top = np.maximum(eigvals[top_d], eps)
        Lp = np.diag(np.sqrt(eigvals_top)) @ eigvecs[:, top_d].T
        L_list.append(Lp)
    constraints = [cp.norm(Lp @ intrinsic_coeffs, 2) <= 1.0 for Lp in L_list]

    # Optional: Dirichlet regularisation i.e. 1-Lipschitz on average
    if reg:
        evals, evecs = dg.laplacian(0).spectrum()
        sqrt_evals = np.sqrt(evals)
        lap_matrix = np.diag(sqrt_evals) @ evecs.coeffs.T
        constraints.append(cp.norm(lap_matrix @ intrinsic_coeffs, 2) <= 1.0)

    # d(x,x) == 0
    constraints.append(dg.cache.u[index].T @ v == 0)

    # Objective:
    objective = cp.Maximize(v[0])

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed: {problem.status}")

    # Reconstruct distance function in data space
    v_value = np.array(v.value).ravel()
    v_function = dg.function_space.wrap(v_value)
    dist = ambient_dist_pointwise + v_function.to_ambient()

    return dist, v_function
