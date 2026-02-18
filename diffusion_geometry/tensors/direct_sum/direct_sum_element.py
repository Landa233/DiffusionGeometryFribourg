"""
Direct sum element tensor for diffusion geometry.

Represents an element in the direct sum of multiple tensor spaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from diffusion_geometry.tensors.base_tensor.base_tensor import Tensor
from diffusion_geometry.utils.batch_utils import _infer_batch_shape


if TYPE_CHECKING:
    from .direct_sum_space import DirectSumSpace


class DirectSumElement(Tensor):
    """
    Tensor living in a direct sum of tensor spaces.

    Stores coefficients for multiple tensor types concatenated along the last axis.
    Use `unpack()` to extract individual tensor components.
    """

    def __init__(self, space: "DirectSumSpace", coeffs: np.ndarray):
        from .direct_sum_space import DirectSumSpace

        if not isinstance(space, DirectSumSpace):
            raise TypeError(
                "DirectSumElement requires a DirectSumSpace; "
                f"got {type(space).__name__}."
            )

        coeffs_arr, batch_shape = _infer_batch_shape(
            coeffs, (space.dim,), name="DirectSumElement"
        )
        super().__init__(space, coeffs_arr, rank=None, batch_shape=batch_shape)

    def unpack(self) -> tuple[Tensor, ...]:
        """Return wrapped components for each summand space.

        Returns
        -------
        components : tuple of Tensor
            The individual tensor components (e.g., Function, VectorField, etc.).
        """
        components = []
        for space, coeffs in zip(
            self.space.spaces, self.space.split_coeffs(self.coeffs)
        ):
            components.append(space.wrap(coeffs))
        return tuple(components)

    def coeff_blocks(self) -> tuple[np.ndarray, ...]:
        """Return a tuple of raw coefficient views corresponding to each summand."""
        return self.space.split_coeffs(self.coeffs)

    def to_pointwise_basis(self) -> np.ndarray:
        """Not implemented for direct sum elements."""
        raise NotImplementedError(
            "Convert each component to the pointwise basis individually after unpack()."
        )
