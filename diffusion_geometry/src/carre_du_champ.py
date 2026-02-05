from opt_einsum import contract
from .basis_utils import get_wedge_basis_indices, get_symmetric_basis_indices
import numpy as np
from numpy.linalg import det


def carre_du_champ_knn(
    f, h, diffusion_kernel, nbr_indices, bandwidths=None, use_mean_centres=True
):
    """
    General carré du champ computation between two arbitrary functions.

    Γ_p(f, h) = (1/2ρ) * Cov_p(f, h)

    If use_mean_centres=True, then
    Cov_p(f, h) = E_p[(f - E_p[f])(h - E_p[h])]
    in the local neighbourhood of each point p.

    If use_mean_centres=False, then
    Cov_p(f, h) = E_p[(f - f(p))(h - h(p))]
    is centred at the point itself.

    Parameters
    ----------
    f, h : array
        Functions with arbitrary trailing dimensions.
        Shape: [n, ...]
    diffusion_kernel : array
        Diffusion kernel matrix (rows sum to 1).
        Shape: [n, k]
    nbr_indices : array
        Indices for the k nearest neighbours.
        Shape: [n, k]
    bandwidths : array, optional
        Local bandwidths ρ.
        Shape: [n]
    use_mean_centres : bool
        Whether to use local means as centres for covariance.

    Returns
    -------
    cdc : array
        carré du champ (covariance) field.
        Shape: [n, f_shape, h_shape]
    """
    # Note: this implementation uses the 'expectation of differences' implementation
    # rather than computing E(fh) - E(f)E(h). This avoids catastrophic cancellation.
    # If f_shape, h_shape < k the expanded version is marginally better in compute time
    # and memory, but for f_shape, h_shape >> k (the important case) this 'differences'
    # implementation is much more memory efficient.

    n, k = diffusion_kernel.shape
    f_shape = f.shape[1:]
    h_shape = h.shape[1:]

    # Flatten f and h to 2D
    f_flat = f.reshape(n, -1)  # [n, f_flat]
    h_flat = h.reshape(n, -1)  # [n, h_flat]

    # Get neighbour values
    nbrs_f = f_flat[nbr_indices]  # [n, k, f_flat]
    nbrs_h = h_flat[nbr_indices]  # [n, k, h_flat]

    if use_mean_centres:
        # Compute local means E_i[f] and E_j[h]
        means_f = (diffusion_kernel[:, :, None] * nbrs_f).sum(axis=1)  # [n, f_flat]
        means_h = (diffusion_kernel[:, :, None] * nbrs_h).sum(axis=1)  # [n, h_flat]
        # Differences (f_i - E_i[f]) and (h_j - E_j[h])
        diff_f = nbrs_f - means_f[:, None, :]  # [n, k, f_flat]
        diff_h = nbrs_h - means_h[:, None, :]  # [n, k, h_flat]
    else:
        # Differences (f_i - f_i) and (h_j - h_j)
        diff_f = nbrs_f - f_flat[:, None, :]  # [n, k, f_flat]
        diff_h = nbrs_h - h_flat[:, None, :]  # [n, k, h_flat]

    # Covariance Cov(f_i, h_j): [n, f_flat, h_flat]
    # Shape: (n, k, f_flat), (n, k, h_flat), (n, k) -> (n, f_flat, h_flat)
    # Formula: E_p[(f - E_p[f])(h - E_p[h])] = ∑_k w_{pk} (f_k - \bar{f}_p)(h_k - \bar{h}_p)
    cdc_flat = contract("pki,pkj,pk->pij", diff_f, diff_h, diffusion_kernel)
    if bandwidths is None:
        cdc_flat /= 2
    else:
        cdc_flat /= 2 * bandwidths[:, None, None]

    # Reshape result to [n, (f_shape), (h_shape)]
    result_shape = (n,) + f_shape + h_shape
    return cdc_flat.reshape(result_shape)


def carre_du_champ_graph(
    f, h, diffusion_kernel, edge_index, bandwidths=None, use_mean_centres=True
):
    """
    General carré du champ computation between two arbitrary functions (General Graph).

    Γ_p(f, h) = (1/2ρ) * Cov_p(f, h)

    If use_mean_centres=True, then
    Cov_p(f, h) = E_p[(f - E_p[f])(h - E_p[h])]
    in the local neighbourhood of each point p.

    If use_mean_centres=False, then
    Cov_p(f, h) = E_p[(f - f(p))(h - h(p))]
    is centred at the point itself.

    Parameters
    ----------
    f, h : array
        Functions with arbitrary trailing dimensions.
        Shape: [n, ...]
    diffusion_kernel : array
        Diffusion weights. Assumed to be row-stochastic.
        Shape: [num_edges]
    edge_index : array
        Source-target index pairs.
        row 0: source j, row 1: target i.
        Shape: [2, num_edges]
    bandwidths : array, optional
        Local bandwidths ρ.
        Shape: [n]
    use_mean_centres : bool
        Whether to use local means as centres for covariance.

    Returns
    -------
    cdc : array
        carré du champ (covariance) field.
        Shape: [n, f_shape, h_shape]
    """
    # Unlike the kNN version which uses a specialised einsum contraction to avoid
    # memory explosion, this graph version must explicitly compute the outer product
    # per edge ([E, D, D]) before aggregating.

    n = f.shape[0]
    num_edges = diffusion_kernel.shape[0]
    f_shape = f.shape[1:]
    h_shape = h.shape[1:]

    # Flatten f and h to 2D
    f_flat = f.reshape(n, -1)  # [n, f_flat]
    h_flat = h.reshape(n, -1)  # [n, h_flat]

    # Unpack edges (j -> i)
    src, tgt = edge_index[0], edge_index[1]

    # Get neighbour values
    nbrs_f = f_flat[src]  # [num_edges, f_flat]
    nbrs_h = h_flat[src]  # [num_edges, h_flat]

    if use_mean_centres:
        # Compute local means E_i[f] and E_i[h] via scatter add
        # We assume diffusion_kernel contains the weights w_{ji}
        w_f = nbrs_f * diffusion_kernel[:, None]
        w_h = nbrs_h * diffusion_kernel[:, None]

        means_f = np.zeros_like(f_flat)
        np.add.at(means_f, tgt, w_f)  # [n, f_flat]

        means_h = np.zeros_like(h_flat)
        np.add.at(means_h, tgt, w_h)  # [n, h_flat]

        # Differences (f_j - E_i[f]) and (h_j - E_i[h])
        # We gather the computed means back to the edges to subtract
        diff_f = nbrs_f - means_f[tgt]  # [num_edges, f_flat]
        diff_h = nbrs_h - means_h[tgt]  # [num_edges, h_flat]
    else:
        # Differences (f_j - f_i) and (h_j - h_i)
        diff_f = nbrs_f - f_flat[tgt]  # [num_edges, f_flat]
        diff_h = nbrs_h - h_flat[tgt]  # [num_edges, h_flat]

    # Covariance Cov(f_i, h_j): [n, f_flat, h_flat]
    # We calculate the weighted outer product per edge: w_e * (diff_f * diff_h^T)
    # Shape: [num_edges, f_flat, 1] * [num_edges, 1, h_flat] * [num_edges, 1, 1]
    terms = (diff_f[:, :, None] * diff_h[:, None, :]) * diffusion_kernel[:, None, None]

    # Flatten the feature dimensions to aggregate efficiently: [num_edges, f_flat * h_flat]
    out_dim = f_flat.shape[1] * h_flat.shape[1]
    terms_flat = terms.reshape(num_edges, out_dim)

    # Aggregate edges to nodes
    cdc_flat = np.zeros((n, out_dim), dtype=terms.dtype)
    np.add.at(cdc_flat, tgt, terms_flat)

    if bandwidths is None:
        cdc_flat /= 2
    else:
        cdc_flat /= (
            2 * bandwidths[:, None]
        )  # Note: bandwidths is [n], need [n, 1] for broadcast

    # Reshape result to [n, (f_shape), (h_shape)]
    result_shape = (n,) + f_shape + h_shape
    return cdc_flat.reshape(result_shape)


def gamma_compound(gamma_coords, k):
    """
    Compute the k-th compound matrix submatrices and their determinants for each point.

    Parameters
    ----------
    gamma_coords : array
        Coordinate carré du champ matrices at each point Γ(x_i, x_j).
        Shape: [n, d, d]
    k : int
        Form degree.

    Returns
    -------
    submatrices : array
        k×k submatrices for each point and combination pair.
        Shape: [n, Ck, Ck, k, k] where Ck = comb(d, k)
    dets : array
        k-th compound matrix determinant for each point det(Γ(x_J, x_{J'})).
        Shape: [n, Ck, Ck]
        Entry [p, a, b] is det(Γ[p][J_a, J_b]) where J_a, J_b are
        k-combinations of {0,...,d-1} in lex order.
    """
    n, d, _ = gamma_coords.shape

    if k == 0:
        # Convention: 0th compound = 1
        submatrices = np.empty((n, 1, 1, 0, 0))  # empty 0×0 matrices
        dets = np.ones((n, 1, 1), dtype=gamma_coords.dtype)
        return submatrices, dets

    if k == 1:
        # 1st compound is just the original matrix
        submatrices = gamma_coords[:, :, :, None, None]  # (n, d, d, 1, 1)
        dets = gamma_coords  # (n, d, d)
        return submatrices, dets

    if k == d:
        # dth compound = determinant of the whole matrix
        submatrices = gamma_coords[:, None, None, :, :]  # (n, 1, 1, d, d)
        dets = np.linalg.det(gamma_coords).reshape(n, 1, 1)
        return submatrices, dets

    idx = get_wedge_basis_indices(d, k)  # (Ck, k)
    Ck = idx.shape[0]

    # Slice out [n, Ck, Ck] lots of k×k submatrices
    rows_taken = np.take(gamma_coords, idx, axis=1)  # (n, Ck, k, d)
    cols_taken = np.take(rows_taken, idx, axis=3)  # (n, Ck, k, Ck, k)
    submatrices = np.moveaxis(cols_taken, 2, -2)  # (n, Ck, Ck, k, k)

    dets = det(submatrices.reshape(-1, k, k))  # (n*Ck*Ck,)
    dets = dets.reshape(n, Ck, Ck)

    return submatrices, dets


def gamma_02(gamma_coords):
    """
    Create expanded carré du champ tensor for full (0,2)-tensors.

    This creates the tensor used in g^{(0,2)} computations, representing all
    pairwise combinations of coordinate indices.

    Γ_{0,2}(dx_j ⊗ dx_J, dx_k ⊗ dx_K) = Γ(x_j, x_k) Γ(x_J, x_{K})

    Parameters
    ----------
    gamma_coords : array
        Coordinate carré du champ matrices at each point Γ(x_i, x_j).
        Shape: [n, d, d]

    Returns
    -------
    gamma_02 : array
        Expanded Γ matrix where gamma_02[p, j ⊗ k, J ⊗ K] =
        Γ(x_j, x_J) Γ(x_k, x_K) with indices flattened.
        Shape: [n, d*d, d*d]
    """
    n, d, _ = gamma_coords.shape

    # Γ_p(x_j, x_J) Γ_p(x_k, x_K)
    # Shape: (n, d, d), (n, d, d) -> (n, d, d, d, d)
    # p: point index, j, J, k, K: coordinate indices
    expanded = contract("pjJ,pkK->pjkJK", gamma_coords, gamma_coords)

    # Reshape to (n, d^2, d^2) format
    expanded = expanded.reshape(n, d * d, d * d)

    return expanded


def gamma_02_sym(gamma_coords):
    """
    Compute the symmetric part of the coordinate carré du champ matrix with proper
    weighting for off-diagonal elements in the inner product.

    For symmetric tensors, off-diagonal basis elements should contribute
    twice as much to the inner product to account for both (i,j) and (j,i)
    positions in the full tensor.

    Parameters
    ----------
    gamma_coords : array
        Coordinate carré du champ matrices at each point Γ(x_i, x_j).
        Shape: [n, d, d]

    Returns
    -------
    sym_gamma : array
        Symmetric Γ matrix with proper off-diagonal weighting.
        Shape: [n, d_sym, d_sym] where d_sym = d * (d + 1) / 2
    """
    d = gamma_coords.shape[1]
    sym_idx = get_symmetric_basis_indices(d)

    j1 = sym_idx[:, 0]
    j2 = sym_idx[:, 1]

    gamma_j1k1 = gamma_coords[:, j1, :][:, :, j1]
    gamma_j2k2 = gamma_coords[:, j2, :][:, :, j2]
    gamma_j1k2 = gamma_coords[:, j1, :][:, :, j2]
    gamma_j2k1 = gamma_coords[:, j2, :][:, :, j1]

    sym_gamma = 0.5 * (gamma_j1k1 * gamma_j2k2 + gamma_j1k2 * gamma_j2k1)

    # Apply weighting factors for off-diagonal elements
    # Create masks for diagonal vs off-diagonal basis elements
    is_diag_row = j1 == j2
    is_diag_col = sym_idx[:, 0] == sym_idx[:, 1]  # Same as is_diag_row, for clarity

    # Weight matrix: 1 for diagonal-diagonal, 2 for diagonal-off-diagonal or off-diagonal-diagonal,
    # 4 for off-diagonal-off-diagonal
    weight_matrix = np.ones((len(sym_idx), len(sym_idx)))
    weight_matrix[~is_diag_row, :] *= 2  # Off-diagonal rows get factor 2
    weight_matrix[:, ~is_diag_col] *= 2  # Off-diagonal columns get factor 2

    # Apply weights
    sym_gamma = sym_gamma * weight_matrix[None, :, :]

    return sym_gamma
