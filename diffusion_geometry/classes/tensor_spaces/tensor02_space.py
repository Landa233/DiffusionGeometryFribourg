"""
The space of general (0,2)-tensors for diffusion geometry.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .base import BaseTensorSpace

if TYPE_CHECKING:
    import numpy as np

    from ..tensors import Tensor02

from ...src import carre_du_champ


class Tensor02Space(BaseTensorSpace):
    """
    Space of general (0,2)-tensors Ω⁰²(M).

    General (0,2)-tensors have dim² independent components per point. Compare with Tensor02SymSpace
    for symmetric tensors which use fewer components.
    """

    @property
    def cdc_components(self) -> "np.ndarray":
        """
        Pointwise carré du champ components for (0,2)-tensors, Γ(dxᵢ ⊗ dxⱼ, dxₖ ⊗ dxₗ) = Γⁱᵏ(x) Γʲˡ(x).
        """
        return carre_du_champ.gamma_02(self.dg.cache.gamma_coords)

    def wrap(self, coeffs: "np.ndarray") -> "Tensor02":
        from ..tensors import Tensor02

        return Tensor02(self, coeffs)
