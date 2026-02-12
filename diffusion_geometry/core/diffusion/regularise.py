import numpy as np


def regularise_diffusion(
    x: np.ndarray, kernel: np.ndarray, nbr_indices: np.ndarray
) -> np.ndarray:
    """
    Regularise x using diffusion steps (averaging over neighbors).

    Parameters
    ----------
    x : array
        Input signal.
        Shape: [n, ...]
    kernel : array
        Diffusion kernel.
        Shape: [n, k] or [n, n] sparse
    nbr_indices : array
        Indices of neighbors (if kernel is dense/local).
        Shape: [n, k]

    Returns
    -------
    x_reg : array
        Regularised signal.
        Shape: [n, ...]
    """
    x_flat = x.reshape(kernel.shape[0], -1)
    nbrs_x = x_flat[nbr_indices]
    # x_reg(p) = sum_k K(p, k) * x(nbr_k)
    x_reg_flat = (kernel[:, :, None] * nbrs_x).sum(axis=1)
    return x_reg_flat.reshape(x.shape)


def regularise_bandlimit(
    x: np.ndarray, u: np.ndarray, measure: np.ndarray
) -> np.ndarray:
    """
    Regularise x by projecting onto the first n_coefficients coefficient functions.

    Parameters
    ----------
    x : array
        Input signal.
        Shape: [n, ...]
    u : array
        Coefficient functions φ_i.
        Shape: [n, n0]
    measure : array
        Measure μ.
        Shape: [n]

    Returns
    -------
    x_reg : array
        Bandlimited signal.
        Shape: [n, ...]
    """
    x_flat = x.reshape(u.shape[0], -1)
    # coeffs = ⟨u, x⟩_measure = u.T @ (measure * x)
    coeffs = u.conj().T @ (measure[:, None] * x_flat)
    # x_reg = u @ coeffs
    x_reg_flat = u @ coeffs

    return x_reg_flat.reshape(x.shape)
