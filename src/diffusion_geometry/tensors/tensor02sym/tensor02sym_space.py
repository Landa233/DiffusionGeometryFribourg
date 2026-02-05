"""
Space of symmetric (0,2)-tensors for diffusion geometry.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from diffusion_geometry.core.diffusion.carre_du_champ import gamma_02_sym
from diffusion_geometry.tensors.base_tensor.base_tensor_space import BaseTensorSpace


if TYPE_CHECKING:
    import numpy as np

    from diffusion_geometry.tensors.tensor02sym.tensor02sym import Tensor02Sym


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
        return gamma_02_sym(self.dg.cache.gamma_coords)

    def wrap(self, coeffs: "np.ndarray") -> "Tensor02Sym":
        from diffusion_geometry.tensors.tensor02sym.tensor02sym import Tensor02Sym

        return Tensor02Sym(self, coeffs)
