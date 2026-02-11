from functools import cached_property, lru_cache
from typing import Tuple

import numpy as np

from diffusion_geometry.core.diffusion.carre_du_champ import gamma_compound
from diffusion_geometry.core.diffusion.markov_triples import ImmersedMarkovTriple
# from diffusion_geometry.classes.base import GeometryEngine
from diffusion_geometry.operators.differential_operators.hessian import hessian_coords, hessian_functions
from diffusion_geometry.core.geometry.geometry_engine import GeometryEngine


class DiffusionGeometryCache(GeometryEngine):
    """
    Cache engine for computing and caching Diffusion Geometry tensors.
    """

    def __init__(self, data: ImmersedMarkovTriple, xp=None):
        super().__init__(xp=xp)
        self.triple = data

    @cached_property
    def gamma_functions(self) -> np.ndarray:
        """
        Compute the carré du champ tensor of the basis of coefficient functions.
        Γ(φ_i, φ_j) for i,j ≤ n_function_basis.

        Returns
        -------
        CdC : np.ndarray
            Carré du champ tensor. Shape: [n, n_function_basis, n_function_basis]
        """
        return self.triple.cdc(self.triple.function_basis, self.triple.function_basis)

    @cached_property
    def gamma_coords(self) -> np.ndarray:
        """
        Carré du champ tensor of the immersion coordinates.
        Γ(x_i, x_j)

        Returns
        -------
        CdC : np.ndarray
            Carré du champ tensor. Shape: [n, dim, dim]
        """
        return self.triple.cdc(
            self.triple.immersion_coords, self.triple.immersion_coords
        )

    @cached_property
    def gamma_coords_regularised(self) -> np.ndarray:
        """
        Compute the regularised carré du champ tensor of immersion_coords functions.
        Gamma(x_i, x_j)

        WARNING: this will lose positive semi-definiteness and so should not be used
        as a metric! Only for passing to an iterated carré du champ computation.

        Returns:
        --------
        CdC: np.ndarray (n, dim, dim)
            Carré du champ tensor of the data.
        """
        return self.triple.regularise(self.gamma_coords)

    @cached_property
    def gamma_mixed(self) -> np.ndarray:
        """
        Carré du champ tensor of the immersion coordinates and coefficient functions.
        Γ(x_i, φ_j)

        Returns
        -------
        CdC : np.ndarray
            Carré du champ tensor. Shape: [n, dim, n_function_basis]
        """
        return self.triple.cdc(self.triple.immersion_coords, self.triple.function_basis)

    @cached_property
    def gamma_ambient(self) -> np.ndarray:
        """
        Carré du champ tensor of immersion_coords functions
        and ambient coordinates.
        Gamma(x_i, x_j)

        Returns:
        --------
        CdC: np.ndarray (n, ambient_dim, dim)
            Carré du champ tensor of the data.
        """
        if self.triple.data_matrix is None:
            return self.gamma_coords
        else:
            return self.triple.cdc(
                self.triple.data_matrix, self.triple.immersion_coords
            )

    @lru_cache(maxsize=None)
    def gamma_coords_compound(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and cache the compound matrix of gamma_coords and its determinants.
        """
        assert (
            1 <= k <= self.triple.dim
        ), f"Form degree k={k} must be between 1 and {self.triple.dim}"

        # Shape: (n, d, d) -> (n, d, d, 1, 1), (n, d, d)
        if k == 1:
            return self.gamma_coords[:, :, :, None, None], self.gamma_coords

        return gamma_compound(self.gamma_coords, k)

    @cached_property
    def hessian_functions(self) -> np.ndarray:
        """
        Hessian matrix tensor H(phi_k)(x_i, x_j).
        Regularised.
        """
        hess = hessian_functions(
            self.triple.function_basis,
            self.triple.immersion_coords,
            self.gamma_coords_regularised,
            # this is the only use of gamma_mixed_regularised so we do not cache it
            self.triple.regularise(self.gamma_mixed),
            cdc=self.triple.cdc,
        )
        return self.triple.regularise(hess)

    @cached_property
    def hessian_coords(self) -> np.ndarray:
        """
        Hessian of the coordinates H(x_k)(x_i, x_j).
        Regularised.
        """
        hess = hessian_coords(
            self.triple.immersion_coords,
            self.gamma_coords_regularised,
            cdc=self.triple.cdc,
        )
        return self.triple.regularise(hess)
