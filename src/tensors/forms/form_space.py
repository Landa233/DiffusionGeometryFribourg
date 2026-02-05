"""
Form space descriptor for diffusion geometry.

The space of differential k-forms.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from tensors.base_tensor.base_tensor_space import BaseTensorSpace


if TYPE_CHECKING:
    import numpy as np
    from core.geometry.diffusion_geometry import DiffusionGeometry
    from tensors.forms.form import Form


class FormSpace(BaseTensorSpace):
    """
    Space of differential k-forms Ωᵏ(M).

    Forms of degree k have C(dim, k) components per point.
    """

    def __init__(self, dg: "DiffusionGeometry", degree: int) -> None:
        super().__init__(dg)
        assert (
            1 <= degree <= dg.dim
        ), f"Form degree k={degree} must be between 1 and {dg.dim}"
        self._degree = degree

    def __repr__(self) -> str:
        return (
            f"FormSpace(degree={self.degree}, coeff_dimension={self.coeff_dimension})"
        )

    @property
    def degree(self) -> int:
        """Degree of the differential forms."""
        return self._degree

    @property
    def cdc_components(self) -> np.ndarray:
        """
        Pointwise carré du champ components for k-forms, Γᵏ(x) = (Γ(x; dx_J1, dx_J2)) with J1, J2 in C(d, k).
        """
        return self.dg.cache.gamma_coords_compound(self.degree)[1]

    def wrap(self, coeffs: "np.ndarray") -> "Form":
        from tensors.forms.form import Form

        return Form(self, coeffs, degree=self.degree)
