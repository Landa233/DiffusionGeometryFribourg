"""
Vector field tensor for diffusion geometry.

Represents a vector field.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from opt_einsum import contract

from .base import Tensor, _infer_batch_shape, _from_pointwise_basis

if TYPE_CHECKING:
    from ..operators import LinearOperator
    from ..tensor_spaces import VectorFieldSpace
    from ..main import DiffusionGeometry
    from .function import Function
    from .form import Form
    from .tensor02 import Tensor02


class VectorField(Tensor):
    """
    A vector field X ∈ 𝔛(M).

    Vector fields are represented in the basis {φ_i ∇x_j}.
    """

    def __init__(self, space: "VectorFieldSpace", coeffs: np.ndarray):
        from ..tensor_spaces import VectorFieldSpace

        if not isinstance(space, VectorFieldSpace):
            raise TypeError(
                "VectorField requires a VectorFieldSpace; "
                f"got {type(space).__name__}."
            )

        expected_size = space.dg.n_coefficients * space.dg.dim
        coeffs_arr, batch_shape = _infer_batch_shape(
            coeffs, (expected_size,), name="VectorField"
        )
        super().__init__(space, coeffs_arr, rank=1, batch_shape=batch_shape)

    def __repr__(self):
        return (
            f"VectorField(shape={self.shape}, batch_shape={self.batch_shape}, "
            f"dg.n={self.dg.n}, dg.dim={self.dg.dim})"
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def degree(self) -> int:
        """Return the associated differential form degree (always 1)."""
        return 1

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: np.ndarray, dg: "DiffusionGeometry") -> "VectorField":
        """Create a vector field from coefficients."""
        return dg.vector_field_space.wrap(coeffs)

    @classmethod
    def from_pointwise_basis(
        cls, data: np.ndarray, dg: "DiffusionGeometry"
    ) -> "VectorField":
        """
        Create vector field from pointwise vectors at sample points.

        Parameters
        ----------
        data : ndarray
            Vector field data with shape (..., n, dim).
        dg : DiffusionGeometry
            The diffusion geometry instance.
        """
        coeffs = _from_pointwise_basis(data, dg.vector_field_space)
        return dg.vector_field_space.wrap(coeffs)

    @classmethod
    def from_reconstruction(
        cls, data: np.ndarray, dg: "DiffusionGeometry"
    ) -> "VectorField":
        """
        Create vector field by least-squares fit from ambient space vectors.

        Parameters
        ----------
        data : ndarray
            Ambient space vectors with shape (..., n, dim).
        dg : DiffusionGeometry
            The diffusion geometry instance.
        """
        data = np.asarray(data)
        assert data.ndim >= 2 and data.shape[-2:] == (
            dg.n,
            dg.dim,
        ), f"Vector field data must have trailing shape ({dg.n}, {dg.dim}), got {data.shape}"

        batch_shape = data.shape[:-2]
        data_flat = data.reshape((-1, dg.n * dg.dim))

        # Reconstruct from ambient representation via least-squares fit.
        # Shape: (n * d, n1 * d)
        quiver_map = dg.operators_engine.vector_field_to_quiver
        A = quiver_map.reshape(dg.n * dg.dim, dg.n_coefficients * dg.dim)
        ATA = A.T @ A
        ATb = data_flat @ A
        try:
            solution = np.linalg.solve(ATA, ATb.T).T
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(A, data_flat.T, rcond=dg.rcond)[0].T

        coeffs = solution.reshape(batch_shape + (dg.n_coefficients * dg.dim,))
        return dg.vector_field_space.wrap(coeffs)

    # -------------------------------------------------------------------------
    # Ambient space action
    # -------------------------------------------------------------------------

    def to_ambient(self) -> np.ndarray:
        """Convert to ambient quiver representation."""
        return self.flat().to_ambient()

    # -------------------------------------------------------------------------
    # Duality
    # -------------------------------------------------------------------------

    def flat(self) -> "Form":
        """Lower index using the metric to get a 1-form."""
        from .form import Form

        return self.dg.form_space(1).wrap(self._coeffs)

    # -------------------------------------------------------------------------
    # Action
    # -------------------------------------------------------------------------

    @cached_property
    def operator(self) -> "LinearOperator":
        """
        Convert the vector field to a linear operator.

        Returns a LinearOperator that computes the directional derivative
        of functions along this vector field. Only applies to unbatched vector fields.
        """
        assert not self.batch_shape, "to_operator only supports unbatched VectorFields."

        # Compute the operator conversion as a matrix of shape [n0, n0].
        weak_matrix = contract(
            "ij,ps,pi,pjt,p->st",
            self.coeffs.reshape(self.dg.n_coefficients, self.dg.dim),
            self.dg.function_basis,
            self.dg.function_basis[:, : self.dg.n_coefficients],
            self.dg.cache.gamma_mixed,
            self.dg.measure,
        )

        from ..operators import LinearOperator

        return LinearOperator(
            domain=self.dg.function_space,
            codomain=self.dg.function_space,
            weak_matrix=weak_matrix,
        )

    def __call__(self, f: "Function") -> "Function":
        """
        Apply the vector field as a directional derivative to a function.

        X(f) = g(X, ∇f)

        Parameters
        ----------
        f : Function
            The function to differentiate.

        Returns
        -------
        Xf : Function
            The directional derivative X(f).
        """
        return self.operator(f)

    def __matmul__(self, f):
        """Shorthand for calling the vector field on a function."""
        from .function import Function

        if isinstance(f, Function):
            return self.__call__(f)
        return NotImplemented

    # -------------------------------------------------------------------------
    # Differential operators
    # -------------------------------------------------------------------------

    def div(self) -> "Function":
        """Apply the divergence operator div : 𝔛(M) ↦ A."""
        return self.dg.div(self)

    def levi_civita(self) -> "Tensor02":
        """Apply the Levi-Civita connection ∇ : 𝔛(M) ↦ Ω⁰²(M)."""
        return self.dg.levi_civita(self)
