"""
Function tensor for diffusion geometry.

Represents a scalar function (0-form).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt_einsum import contract
from tensors.base_tensor.base_tensor import Tensor


from utils.basis_conversions import _from_pointwise_basis, _to_pointwise_basis
from utils.batch_utils import _infer_batch_shape


if TYPE_CHECKING:
    from tensors.functions.function_space import FunctionSpace
    from core.geometry.diffusion_geometry import DiffusionGeometry
    from tensors.vector_fields.vector_field import VectorField
    from tensors.forms.form import Form
    from tensors.tensor02sym.tensor02sym import Tensor02Sym


class Function(Tensor):
    """
    A scalar function f ∈ A on the space.

    Functions are represented as linear combinations of the basis of
    coefficient functions {φ_i}.
    """

    def __init__(self, space: "FunctionSpace", coeffs: np.ndarray):
        from tensors.functions.function_space import FunctionSpace

        if not isinstance(space, FunctionSpace):
            raise TypeError(
                "Function requires a FunctionSpace; " f"got {type(space).__name__}."
            )

        coeffs_arr, batch_shape = _infer_batch_shape(
            coeffs, (space.dg.n_function_basis,), name="Function"
        )
        super().__init__(space, coeffs_arr, rank=None, batch_shape=batch_shape)

    def __repr__(self):
        return (
            f"Function(shape={self.shape}, batch_shape={self.batch_shape}, "
            f"dg.n={self.dg.n}, dg.dim={self.dg.dim})"
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def degree(self) -> int:
        """Return the differential form degree (0 for functions)."""
        return 0

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: np.ndarray, dg: "DiffusionGeometry") -> "Function":
        """Create a function from coefficients in the basis {φ_i}."""
        return dg.function_space.wrap(coeffs)

    @classmethod
    def from_pointwise_basis(
        cls, data: np.ndarray, dg: "DiffusionGeometry"
    ) -> "Function":
        """
        Create function from pointwise values at sample points.

        Parameters
        ----------
        data : ndarray
            Function values with shape (..., n) where n is the number of points.
        dg : DiffusionGeometry
            The diffusion geometry instance.

        Returns
        -------
        Function
            The function in coefficient form.
        """
        data = np.asarray(data)
        assert (
            data.shape[-1] == dg.n
        ), f"Function data must have last dimension {dg.n}, got {data.shape}"

        # Use shared helper with basis_count=dg.n_function_basis.
        # Function data has shape (..., n), helper expects (..., n, components)
        # So we add a dummy component dimension.
        data_expanded = data[..., np.newaxis]
        coeffs = _from_pointwise_basis(
            data_expanded, dg.function_space, basis_count=dg.n_function_basis
        )

        return dg.function_space.wrap(coeffs)

    # -------------------------------------------------------------------------
    # Basis conversion
    # -------------------------------------------------------------------------

    def to_pointwise_basis(self) -> np.ndarray:
        """
        Convert to pointwise function values at sample points.

        Returns
        -------
        ndarray
            Array of function values at sample points.
            Shape: batch_shape + (dg.n,)
        """
        if self._data_cache is None:
            # We treat the function values as having a component dimension of 1
            # to be compatible with the shared helper _to_pointwise_basis.
            # The helper returns shape (..., n * 1), which is equivalent to (..., n).
            self._data_cache = _to_pointwise_basis(
                self.coeffs, self.space, basis_count=self.dg.n_function_basis
            )
        return self._data_cache

    # -------------------------------------------------------------------------
    # Ambient space action
    # -------------------------------------------------------------------------

    def to_ambient(self) -> np.ndarray:
        """Convert to ambient function representation (same as pointwise)."""
        return self.to_pointwise_basis()

    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------

    def __add__(self, other):
        """
        Add a function or a scalar.

        Parameters
        ----------
        other : Function or scalar
            The object to add.

        Returns
        -------
        Function
            The resulting sum.
        """
        if np.isscalar(other):
            # Create constant function
            constant_data = np.full(
                self.batch_shape + (self.dg.n,), other, dtype=self.coeffs.dtype
            )
            other = Function.from_pointwise_basis(constant_data, self.dg)
            return self.space.wrap(self.coeffs + other.coeffs)
        return super().__add__(other)

    def __pow__(self, other):
        """
        Pointwise exponentiation fᵖ.

        Parameters
        ----------
        other : scalar
            The exponent p.

        Returns
        -------
        f_pow : Function
            The resulting function fᵖ.
        """
        if np.isscalar(other):
            data = self.to_pointwise_basis()
            powered_data = np.power(data, other)
            return Function.from_pointwise_basis(powered_data, self.dg)
        return NotImplemented

    # -------------------------------------------------------------------------
    # Differential operators
    # -------------------------------------------------------------------------

    def grad(self) -> "VectorField":
        """Apply the gradient operator ∇ : A ↦ 𝔛(M)."""
        return self.dg.grad(self)

    def d(self) -> "Form":
        """Apply the exterior derivative d : A ↦ Ω¹(M)."""
        return self.dg.d(0)(self)

    def up_laplacian(self) -> "Function":
        """Apply the standard Laplacian Δ : A ↦ A."""
        return self.dg.up_laplacian(0)(self)

    def laplacian(self) -> "Function":
        """Apply the standard Laplacian f ↦ Δf."""
        return self.dg.up_laplacian(0)(self)

    def hessian(self) -> "Tensor02Sym":
        """Apply the Hessian operator Hess : A ↦ Sym²(Ω¹(M))."""
        return self.dg.hessian(self)

    def hodge_decomposition(self):
        """
        Hodge decomposition components.

        For functions, returns (coexact_potential, harmonic_part).
        """
        coexact_potential = self.dg.down_laplacian(1).inverse()(self.d())
        coexact_part = coexact_potential.codifferential()

        harmonic_part = self - coexact_part
        return coexact_potential, harmonic_part
