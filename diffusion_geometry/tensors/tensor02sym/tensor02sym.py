"""
Symmetric (0,2)-tensor for diffusion geometry.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from diffusion_geometry.tensors.base_tensor.base_tensor import Tensor
from diffusion_geometry.utils.basis_conversions import _from_pointwise_basis
from diffusion_geometry.utils.basis_utils import expand_symmetric_tensor_coeffs
from diffusion_geometry.utils.batch_utils import _infer_batch_shape


if TYPE_CHECKING:
    from diffusion_geometry.core import DiffusionGeometry
    from diffusion_geometry import operators
    from .tensor02sym_space import Tensor02SymSpace
    from diffusion_geometry.tensors import Tensor02


class Tensor02Sym(Tensor):
    """
    A symmetric (0,2)-tensor S ∈ Sym²(Ω¹(M)).

    Symmetric (0,2)-tensors are represented as linear combinations of the basis
    of symmetric products {φ_i (dx_j ⊗ dx_k + dx_k ⊗ dx_j)}.
    """

    def __init__(self, space: "Tensor02SymSpace", coeffs: np.ndarray):
        from .tensor02sym_space import Tensor02SymSpace

        if not isinstance(space, Tensor02SymSpace):
            raise TypeError(
                "Tensor02Sym requires a Tensor02SymSpace; "
                f"got {type(space).__name__}."
            )

        dg = space.dg
        d = dg.dim
        d_sym = d * (d + 1) // 2
        expected_size = dg.n_coefficients * d_sym
        coeffs_arr, batch_shape = _infer_batch_shape(
            coeffs, (expected_size,), name="Tensor02Sym"
        )
        super().__init__(space, coeffs_arr, rank=2, batch_shape=batch_shape)

    def __repr__(self):
        return (
            f"Tensor02Sym(space={self.space!r}, shape={self.shape}, batch_shape={self.batch_shape}, "
            f"dg.n={self.dg.n}, dg.dim={self.dg.dim})"
        )

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: np.ndarray, dg: "DiffusionGeometry") -> "Tensor02Sym":
        """Create a symmetric (0,2)-tensor from coefficients."""
        return dg.tensor02sym_space.wrap(coeffs)

    @classmethod
    def from_pointwise_basis(
        cls, data: np.ndarray, dg: "DiffusionGeometry"
    ) -> "Tensor02Sym":
        """Construct a symmetric (0,2)-tensor from pointwise data."""
        coeffs = _from_pointwise_basis(data, dg.tensor02sym_space)
        return cls(dg.tensor02sym_space, coeffs)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @cached_property
    def full_tensor(self) -> "Tensor02":
        """Full (0,2)-tensor view of this symmetric tensor, cached."""
        coeffs_full = expand_symmetric_tensor_coeffs(
            self.coeffs, self.dg.n_coefficients, self.dg.dim
        )
        return self.dg.tensor02_space.wrap(coeffs_full)

    # -------------------------------------------------------------------------
    # Ambient space action
    # -------------------------------------------------------------------------

    def to_ambient(self) -> np.ndarray:
        """Embed symmetric tensor into full (0,2)-tensor space, then convert."""
        return self.full_tensor.to_ambient()

    # -------------------------------------------------------------------------
    # Action
    # -------------------------------------------------------------------------

    def __call__(self, X, Y=None):
        """Delegate to the cached full (0,2)-tensor representation."""
        return self.full_tensor(X, Y)

    @cached_property
    def operator(self) -> "operators.LinearOperator":
        """Operator form α^{op}: 𝔛(M) → 𝔛(M), computed via full tensor."""
        return self.full_tensor.operator

    # -------------------------------------------------------------------------
    # Transpose
    # -------------------------------------------------------------------------

    def transpose(self) -> "Tensor02Sym":
        """
        Return the transpose of this symmetric (0,2)-tensor.

        Since the tensor is symmetric, T^T = T, so this returns self.
        """
        return self

    @property
    def T(self) -> "Tensor02Sym":
        """Alias for transpose()."""
        return self.transpose()
