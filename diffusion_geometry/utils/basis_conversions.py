"""
Utility functions for basis conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from opt_einsum import contract
from diffusion_geometry.utils.batch_utils import (
    _flatten_batch_dims,
    _restore_batch_dims,
)


if TYPE_CHECKING:
    from diffusion_geometry.tensors import BaseTensorSpace


def _from_pointwise_basis(
    data: np.ndarray, space: "BaseTensorSpace", basis_count: int | None = None
) -> np.ndarray:
    """
    Change coefficients from pointwise basis to function basis.

    Expects data with trailing shape (dg.n, component_dimension).
    Returns coeffs with last dim (basis_count * component_dimension).

    Parameters
    ----------
    data : ndarray
        Pointwise data with shape (..., n, components).
    space : BaseTensorSpace
        The target tensor space.
    basis_count : int, optional
        Number of basis functions to use. Defaults to dg.n_coefficients.
        Use dg.n_function_basis for Functions.

    Returns
    -------
    ndarray
        Coefficients in the function basis.
    """
    components = space.component_dimension
    if basis_count is None:
        basis_count = space.coeff_dimension // components

    data = np.asarray(data)
    if data.ndim < 2 or data.shape[-2:] != (space.dg.n, components):
        raise ValueError(
            f"Data must have trailing shape ({space.dg.n}, {components}), got {data.shape}"
        )

    batch_shape = data.shape[:-2]
    data_flat = data.reshape((-1, space.dg.n, components))  # (batch, n, components)
    coeffs_flat_weak = contract(
        "p,pi,bpI->biI",
        space.dg.measure,
        space.dg.function_basis[:, :basis_count],
        data_flat,
    )  # (batch, basis_count, components)

    # Solve the weak formulation to get coefficients. If the function basis
    # is orthonormal, we can skip the Gram matrix inversion step.
    if space.dg.function_space._is_orthonormal:
        coeffs_flat = coeffs_flat_weak
    else:
        # Note: we use the full Gram inverse because eigenfunctions for Functions
        # may overlap in terms of dependencies even if n_coefficients is smaller
        # than n_function_basis. Slicing is valid as n_coefficients is leading.
        coeffs_flat = contract(
            "si,biI->bsI",
            space.dg.function_space.gram_inv[:basis_count, :basis_count],
            coeffs_flat_weak,
        )
    coeffs = coeffs_flat.reshape(batch_shape + (basis_count * components,))
    return coeffs


def _to_pointwise_basis(
    coeffs: np.ndarray, space: "BaseTensorSpace", basis_count: int | None = None
) -> np.ndarray:
    """
    Change coefficients from function basis to pointwise basis.

    Expects coeffs with trailing shape (basis_count * component_dimension).
    Returns data with last dim (dg.n * component_dimension).

    Parameters
    ----------
    coeffs : ndarray
        Coefficients with shape (..., basis_count * components).
    space : BaseTensorSpace
        The tensor space.
    basis_count : int, optional
        Number of basis functions used. Defaults to dg.n_coefficients.

    Returns
    -------
    ndarray
        Pointwise data.
    """
    if basis_count is None:
        basis_count = space.dg.n_coefficients
    component_dim = space.component_dimension

    coeffs_flat, batch_shape = _flatten_batch_dims(coeffs)
    coeffs_flat_expanded = coeffs_flat.reshape(-1, basis_count, component_dim)

    # Contract: u is (n, basis_count), coeffs is (B, basis_count, c) -> (B, n, c)
    data_flat_expanded = contract(
        "ps,bsc->bpc", space.dg.function_basis[:, :basis_count], coeffs_flat_expanded
    )

    data_flat = data_flat_expanded.reshape(-1, space.dg.n * component_dim)
    return _restore_batch_dims(data_flat, batch_shape)
