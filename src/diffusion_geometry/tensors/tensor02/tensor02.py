"""
General (0,2)-tensor for diffusion geometry.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np


from opt_einsum import contract
from diffusion_geometry.tensors.base_tensor.base_tensor import Tensor

from diffusion_geometry.utils.basis_conversions import _from_pointwise_basis
from diffusion_geometry.utils.basis_utils import symmetrise_tensor_coeffs
from diffusion_geometry.utils.batch_utils import _infer_batch_shape

# from diffusion_geometry.src.basis_utils import symmetrise_tensor_coeffs

# from .base import Tensor, _infer_batch_shape, _from_pointwise_basis

if TYPE_CHECKING:
    from diffusion_geometry.tensors.tensor02.tensor02_space import Tensor02Space
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry
    from diffusion_geometry.tensors.vector_fields.vector_field import VectorField
    from diffusion_geometry.operators.types.linear import LinearOperator
    from diffusion_geometry.tensors.tensor02sym.tensor02sym import Tensor02Sym

    # from ..operators import LinearOperator
    # from ..tensor_spaces import Tensor02Space
    # from ..main import DiffusionGeometry
    # from .vector_field import VectorField
    # from .tensor02sym import Tensor02Sym


class Tensor02(Tensor):
    """
    A (0,2)-tensor T ∈ Ω⁰²(M).

    (0,2)-tensors are represented as linear combinations of the basis {φ_i dx_j ⊗ dx_k}.
    """

    def __init__(self, space: "Tensor02Space", coeffs: np.ndarray):
        from diffusion_geometry.tensors.tensor02.tensor02_space import Tensor02Space

        if not isinstance(space, Tensor02Space):
            raise TypeError(
                "Tensor02 requires a Tensor02Space; " f"got {type(space).__name__}."
            )

        dg = space.dg
        expected_size = dg.n_coefficients * dg.dim * dg.dim
        coeffs_arr, batch_shape = _infer_batch_shape(
            coeffs, (expected_size,), name="Tensor02"
        )
        super().__init__(space, coeffs_arr, rank=2, batch_shape=batch_shape)

    def __repr__(self):
        return (
            f"Tensor02(shape={self.shape}, batch_shape={self.batch_shape}, "
            f"dg.n={self.dg.n}, dg.dim={self.dg.dim})"
        )

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: np.ndarray, dg: "DiffusionGeometry") -> "Tensor02":
        """Create a (0,2)-tensor from coefficients."""
        return dg.tensor02_space.wrap(coeffs)

    @classmethod
    def from_pointwise_basis(
        cls, data: np.ndarray, dg: "DiffusionGeometry"
    ) -> "Tensor02":
        """Construct a (0,2)-tensor from pointwise data."""
        coeffs = _from_pointwise_basis(data, dg.tensor02_space)
        return cls(dg.tensor02_space, coeffs)

    # -------------------------------------------------------------------------
    # Ambient space action
    # -------------------------------------------------------------------------

    def to_ambient(self) -> np.ndarray:
        """
        Convert this (0,2)-tensor to its ambient-coordinate representation.

        Returns
        -------
        ambient : ndarray, shape (n, d_ambient, d_ambient)
            Ambient-space symmetric or general (0,2)-tensor field.
        """
        assert (
            not self.batch_shape
        ), "to_ambient only supports unbatched Tensor02 objects."

        # Get tensor data in pointwise components
        # TODO: this is not correct - use ambient coords
        data = self.to_pointwise_basis().reshape(self.dg.n, self.dg.dim, self.dg.dim)
        data = self.to_pointwise_basis().reshape(self.dg.n, self.dg.dim, self.dg.dim)
        gamma = self.dg.triple.cdc(
            self.dg.immersion_coords, self.dg.immersion_coords)
        gamma = self.dg._regularise(gamma)

        # Raise both covariant indices: T_{ij} -> γ^{αi} γ^{βj} T_{ij}
        # Shape: (n, d, d), (n, d, d), (n, d, d) -> (n, d, d)
        ambient = np.einsum("nai,nbj,nij->nab", gamma, gamma, data)

        return ambient

    # -------------------------------------------------------------------------
    # Action
    # -------------------------------------------------------------------------

    def __call__(self, X, Y=None):
        """
        Evaluate the (0,2)-tensor pointwise on a pair of vector fields,
        or treat it as an operator on one vector field.

        Parameters
        ----------
        X, Y : VectorField
            Vector fields on the same DiffusionGeometry.

        Returns
        -------
        ndarray or LinearOperator
            Pointwise function values if Y is provided, otherwise the operator applied to X.
        """

        # --- Operator form: single argument ---
        if Y is None:
            return self.operator(X)

        from diffusion_geometry.tensors.vector_fields.vector_field import VectorField

        # --- Bilinear form: two arguments ---
        if not isinstance(X, VectorField) or not isinstance(Y, VectorField):
            raise TypeError("α(X, Y) expects two VectorField arguments.")
        assert (
            Y.dg is self.dg
        ), "Both vector fields must belong to the same DiffusionGeometry instance."
        assert (
            X.batch_shape == Y.batch_shape == self.batch_shape
        ), f"T, X, and Y must share batch_shape; got T: {self.batch_shape}, X: {X.batch_shape}, Y: {Y.batch_shape}."

        dg = self.dg
        n, n1, d = dg.n, dg.n_coefficients, dg.dim

        # --- Reshape coefficients ---
        A = self.coeffs.reshape(
            self.batch_shape + (n1, d, d))  # (..., i, j, j')
        Xc = X.coeffs.reshape(self.batch_shape + (n1, d))  # (..., i1, j1)
        Yc = Y.coeffs.reshape(self.batch_shape + (n1, d))  # (..., i2, j2)

        U = self.dg.function_basis[:, :n1]  # (n, n1)
        Gamma = self.dg.cache.gamma_coords  # (n, d, d)
        # α_abc X_ij Y_IJ φ_a φ_i φ_I Γ(x_b, φ_j) Γ(x_c, x_J)
        # Shape: (..., n1, d, d), (..., n1, d), (..., n1, d), (n, n1), (n, n1), (n, n1), (n, d, d), (n, d, d) -> (..., n)
        result = contract(
            "...abc,...ij,...IJ,pa,pi,pI,pbj,pcJ->...p",
            A,
            Xc,
            Yc,
            U,
            U,
            U,
            Gamma,
            Gamma,
        )

        return self.dg._regularise(result)

    @cached_property
    def operator(self) -> "LinearOperator":
        """
        Return the operator form α^{op}: 𝔛(M) → 𝔛(M), defined by
            ⟨α^{op}(X), Y⟩ = ∫ α(X, Y) dμ.

        The weak matrix entries follow
            α^{op,weak}_{i j, I J}
            = Σ_{p,a,b,c} α_{a b c} U_{p a} U_{p i} U_{p I}
                             Γ_p(b, j) Γ_p(c, J) μ_p
        where
            - (a,b,c) index α coefficients over basis φ_a dx_b⊗dx_c
            - (i,j) index the first vector field φ_i ∇x_j
            - (I,J) index the second vector field φ_I ∇x_J
        """
        assert (
            not self.batch_shape
        ), "to_operator only supports unbatched Tensor02 objects."

        dg = self.dg
        n1, d = dg.n_coefficients, dg.dim

        # α coefficients as (a,b,c) ≡ (n1, d, d)
        alpha = self.coeffs.reshape(n1, d, d)

        # Backend components
        U = dg.function_basis[:, :n1]  # (p,a)
        Gamma = self.dg.cache.gamma_coords  # (p,b,j)
        mu = dg.measure  # (p,)

        # Compute weak 4-tensor:
        # Shape: (n1, d, d), (p, n1), (p, n1), (p, n1), (p, d, d), (p, d, d), (p) -> (n1, n1, d, d)
        W4 = contract(
            "abc,pi,pj,pa,pJb,pIc,p -> iIjJ", alpha, U, U, U, Gamma, Gamma, mu
        )

        # Reshape to (n1*d, n1*d)
        weak_matrix = W4.reshape(n1 * d, n1 * d)

        from diffusion_geometry.operators.types.linear import LinearOperator

        return LinearOperator(
            domain=dg.vector_field_space,
            codomain=dg.vector_field_space,
            weak_matrix=weak_matrix,
        )

    # -------------------------------------------------------------------------
    # Symmetrisation
    # -------------------------------------------------------------------------

    def symmetrise(self) -> "Tensor02Sym":
        """Return the symmetric part of this (0,2)-tensor."""
        coeffs_sym = symmetrise_tensor_coeffs(
            self.coeffs, self.dg.n_coefficients, self.dg.dim
        )
        return self.dg.tensor02sym_space.wrap(coeffs_sym)

    # -------------------------------------------------------------------------
    # Transpose
    # -------------------------------------------------------------------------

    def transpose(self) -> "Tensor02":
        """
        Return the transpose of this (0,2)-tensor.

        The transpose T^T is defined by T^T(X, Y) = T(Y, X).
        In coefficients, (T^T)_{ij} = T_{ji}.
        """
        n1, d = self.dg.n_coefficients, self.dg.dim

        coeffs = self.coeffs.reshape(self.batch_shape + (n1, d, d))
        coeffs_T = np.swapaxes(coeffs, -1, -2)
        coeffs_flat = coeffs_T.reshape(self.batch_shape + (n1 * d * d,))
        return self.dg.tensor02_space.wrap(coeffs_flat)

    @property
    def T(self) -> "Tensor02":
        """Alias for transpose()."""
        return self.transpose()
