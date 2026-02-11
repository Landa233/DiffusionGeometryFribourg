"""
Vector field space descriptor for diffusion geometry.

The space of vector fields.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from diffusion_geometry.tensors.base_tensor.base_tensor_space import BaseTensorSpace


if TYPE_CHECKING:
    from .vector_field import VectorField
    import numpy as np


class VectorFieldSpace(BaseTensorSpace):
    """
    Space of vector fields 𝔛(M).

    Vector fields have dim components per point, stored as n1 * dim coefficients.
    """

    @property
    def cdc_components(self) -> np.ndarray:
        """
        Pointwise carré du champ components for vector fields, Γ(dxᵢ, dxⱼ).
        """
        return self.dg.cache.gamma_coords

    def wrap(self, coeffs: "np.ndarray") -> "VectorField":
        from .vector_field import VectorField

        return VectorField(self, coeffs)
