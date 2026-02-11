"""
Function space descriptor for diffusion geometry.

The space of scalar functions.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from opt_einsum import contract
from diffusion_geometry.tensors.base_tensor.base_tensor_space import BaseTensorSpace
from .function import Function

# from .base import BaseTensorSpace

if TYPE_CHECKING:
    from .function import Function


class FunctionSpace(BaseTensorSpace):
    """
    Space of scalar functions A (or L²(M, μ)).

    Functions f: M → ℝ have coefficient dimension n0.

    Functions have coefficient dimension n0 (the number of function basis elements),
    which may differ from n1 used by higher-order tensors.
    """

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @cached_property
    def coeff_dimension(self) -> int:
        """Number of coefficients n_function_basis for the function space."""
        return self.dg.n_function_basis

    @property
    def cdc_components(self) -> np.ndarray:
        """
        Pointwise metric components for functions (always 1 at each point).
        """
        return np.ones((self.dg.n, 1, 1))

    # -------------------------------------------------------------------------
    # Metric and Gram matrix
    # -------------------------------------------------------------------------

    def metric_apply(self, a_coeffs: np.ndarray, b_coeffs: np.ndarray) -> np.ndarray:
        """
        Pointwise inner product for functions is just product.
        Returns coefficients of the product function.
        """
        f = self.wrap(a_coeffs)
        g = self.wrap(b_coeffs)
        return f.to_pointwise_basis() * g.to_pointwise_basis()

    @property
    def gram(self) -> np.ndarray:
        """
        Gram matrix.

        Returns
        -------
        (n0, n0) array
            Gram matrix.
        """
        return contract(
            "p,pi,pI->iI",
            self.dg.triple.measure,
            self.dg.function_basis,
            self.dg.function_basis,
        )

    @cached_property
    def _gram_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors of the Gram matrix."""
        if self._is_orthonormal:
            # Shortcut for orthonormal function bases.
            evals = np.ones(self.dg.n_function_basis)
            evecs = np.eye(self.dg.n_function_basis)
            return evals, evecs
        return super()._gram_spectrum

    @cached_property
    def gram_inv(self) -> np.ndarray:
        """Moore-Penrose pseudoinverse of the Gram matrix."""
        if self._is_orthonormal:
            # Shortcut for orthonormal function bases.
            return np.eye(self.dg.n_function_basis)
        return super().gram_inv

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def wrap(self, coeffs: np.ndarray) -> "Function":
        from .function import Function

        return Function(self, coeffs)
