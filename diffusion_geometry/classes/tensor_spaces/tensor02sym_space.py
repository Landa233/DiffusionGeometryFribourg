"""
Space of symmetric (0,2)-tensors for diffusion geometry.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .base import BaseTensorSpace
from ...src import carre_du_champ

if TYPE_CHECKING:
    import numpy as np

    from ..tensors import Tensor02Sym


class Tensor02SymSpace(BaseTensorSpace):
    """
    Space of symmetric (0,2)-tensors Sym²(Ω¹(M)) ⊂ Ω⁰²(M).

    Symmetric tensors have dim*(dim+1)/2 independent components per point,
    fewer than the dim² used by general (0,2)-tensors.
    """

    @property
    def cdc_components(self) -> "np.ndarray":
        """
        Pointwise metric components for symmetric (0,2)-tensors.
        """
        return carre_du_champ.gamma_02_sym(self.dg.cache.gamma_coords)

    def wrap(self, coeffs: "np.ndarray") -> "Tensor02Sym":
        from ..tensors import Tensor02Sym

        return Tensor02Sym(self, coeffs)
