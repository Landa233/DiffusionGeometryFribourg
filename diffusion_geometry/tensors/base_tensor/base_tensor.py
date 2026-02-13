"""
Base tensor class and utilities for diffusion geometry.

This module provides the abstract Tensor base class and utility functions
for coefficient array handling that are shared across all tensor types.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from diffusion_geometry.utils.basis_conversions import (
    _from_pointwise_basis,
    _to_pointwise_basis,
)
from diffusion_geometry.utils.batch_utils import compatible_batches

if TYPE_CHECKING:
    from diffusion_geometry.core import DiffusionGeometry
    from .base_tensor_space import BaseTensorSpace


class Tensor:
    """
    Base class for tensor objects.

    Tensors are represented as linear combinations of the basis of
    coefficient functions {φ_i} and their derivatives.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        space: "BaseTensorSpace",
        coeffs: np.ndarray,
        rank: int | None = None,
        *,
        batch_shape: tuple[int, ...] | None = None,
    ):
        self._space = space
        self._coeffs = np.asarray(coeffs).copy()
        self._rank = rank
        self._batch_shape = (
            self._coeffs.shape[:-1] if batch_shape is None else batch_shape
        )
        self._data_cache = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def coeffs(self) -> np.ndarray:
        """Coefficients in the tesnor space basis."""
        return self._coeffs

    @property
    def rank(self) -> int | None:
        """Rank (if tensor), else None."""
        return self._rank

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch dimensions preceding the coefficient axis."""
        return self._batch_shape

    @property
    def shape(self) -> tuple:
        """Shape of the coefficients."""
        return self._coeffs.shape

    @property
    def dg(self) -> "DiffusionGeometry":
        """Parent DiffusionGeometry instance."""
        return self._space.dg

    @property
    def space(self) -> "BaseTensorSpace":
        """Tensor space this tensor lives in."""
        return self._space

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_pointwise_basis(
        cls, data: np.ndarray, dg: "DiffusionGeometry"
    ) -> np.ndarray:
        """Construct tensor from pointwise basis data."""
        return NotImplemented

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def __getitem__(self, item):
        """Index into the batch dimensions while preserving tensor semantics."""
        coeffs = self._coeffs[item]
        coeffs_arr = np.asarray(coeffs)
        expected = self._coeffs.shape[-1]

        if coeffs_arr.ndim == 0 or coeffs_arr.shape[-1] != expected:
            return coeffs

        return self.space.wrap(coeffs_arr)

    # -------------------------------------------------------------------------
    # Basis conversion
    # -------------------------------------------------------------------------

    def to_pointwise_basis(self) -> np.ndarray:
        """
        Convert to the pointwise basis at each sample point.

        Expects coeffs with trailing shape (dg.n_coefficients * component_dimension).
        Returns data with last dim (dg.n * component_dimension).

        This will work for all tensor types with num_components > 0, so not for functions
        (0-tensors with num_components = 0). It is overridden in the Function class.
        """
        if self._data_cache is None:
            self._data_cache = _to_pointwise_basis(self.coeffs, self.space)
        return self._data_cache

    # -------------------------------------------------------------------------
    # Complex accessors
    # -------------------------------------------------------------------------

    @property
    def real(self) -> "Tensor":
        """Return a tensor containing the real part of the coefficients."""
        return self.space.wrap(np.real(self.coeffs))

    @property
    def imag(self) -> "Tensor":
        """Return a tensor containing the imaginary part of the coefficients."""
        return self.space.wrap(np.imag(self.coeffs))

    # -------------------------------------------------------------------------
    # Ambient space action
    # -------------------------------------------------------------------------

    def to_ambient(self) -> np.ndarray:
        """Convert to ambient polyvector representation."""
        return NotImplemented

    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------

    def _check_arithmetic_compatibility(
        self, other, require_same_space: bool = True
    ) -> bool:
        """
        Check if `other` is compatible for arithmetic operations.

        Parameters
        ----------
        other : Any
            The object to check compatibility with.
        require_same_space : bool, optional
            If True, requires `other` to be in the exact same tensor space (e.g. for addition).
            If False, only requires `other` to be a Tensor from the same DiffusionGeometry
            instance (e.g. for multiplication). Defaults to True.

        Returns
        -------
        bool
            True if compatible.

        Raises
        ------
        ValueError
            If operands belong to different DiffusionGeometry instances or have mismatching shapes.
        """
        if not isinstance(other, Tensor):
            return False

        assert (
            self.dg is other.dg
        ), "Operands must belong to the same DiffusionGeometry."
        if require_same_space:
            assert (
                other.space is self.space
            ), f"Operands must belong to the same space: {self.space} vs {other.space}"

        assert compatible_batches(
            self.batch_shape, other.batch_shape
        ), f"Incompatible batch shapes: {self.batch_shape} vs {other.batch_shape}"

        return True

    def _broadcast_batch_scalars(
        self, other
    ) -> tuple[np.ndarray, tuple[int, ...]] | None:
        """
        Return ``other`` broadcast over this tensor's batch axes when numeric.

        This treats non-tensor array-likes as batch-wise scalar weights.
        The coefficient axis is never matched against ``other``.
        """
        if isinstance(other, Tensor):
            return None

        try:
            scalars = np.asarray(other)
        except Exception:
            return None

        if scalars.dtype == np.dtype("O"):
            return None

        if not (
            np.issubdtype(scalars.dtype, np.number)
            or np.issubdtype(scalars.dtype, np.bool_)
        ):
            return None

        try:
            target_batch_shape = np.broadcast_shapes(self.batch_shape, scalars.shape)
        except ValueError:
            return None

        return np.broadcast_to(scalars, target_batch_shape), target_batch_shape

    def _broadcast_coeffs_to_batch(self, batch_shape: tuple[int, ...]) -> np.ndarray:
        """Broadcast coefficients to a target batch shape, preserving coeff axis."""
        return np.broadcast_to(self.coeffs, batch_shape + (self.coeffs.shape[-1],))

    @staticmethod
    def _ufunc_where_is_supported(where) -> bool:
        """
        Support only trivial where=True semantics for arithmetic ufunc dispatch.
        """
        if where is True:
            return True
        if np.isscalar(where):
            return bool(where)
        return False

    def __neg__(self):
        return self.space.wrap(-self.coeffs)

    def __add__(self, other):
        if self._check_arithmetic_compatibility(other, require_same_space=True):
            return self.space.wrap(self.coeffs + other.coeffs)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract a tensor of the same space."""
        return self.__add__(-other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        """
        Handles:
        - Scalar multiplication: returns scaled tensor in same space.
        - Function multiplication: returns pointwise product in same space.
        """
        if np.isscalar(other):
            return self.space.wrap(other * self.coeffs)

        from diffusion_geometry.tensors import Function

        if isinstance(other, Function):
            self._check_arithmetic_compatibility(other, require_same_space=False)

            # Pointwise product
            # self: (..., n * C) -> (..., n, C)
            # func: (..., n) -> (..., n, 1)

            c_dim = self.space.component_dimension
            n = self.dg.n

            self_pw = self.to_pointwise_basis()
            func_pw = other.to_pointwise_basis()

            # Helper shapes
            self_pw_reshaped = self_pw.reshape(self.batch_shape + (n, c_dim))
            func_pw_reshaped = func_pw.reshape(other.batch_shape + (n, 1))

            # Broadcast batch shapes if needed (numpy handles this usually)
            product_pw = self_pw_reshaped * func_pw_reshaped

            coeffs = _from_pointwise_basis(product_pw, self.space)
            return self.space.wrap(coeffs)

        batch_scalars = self._broadcast_batch_scalars(other)
        if batch_scalars is not None:
            scalars, target_batch_shape = batch_scalars
            coeffs = self._broadcast_coeffs_to_batch(target_batch_shape)
            return self.space.wrap(coeffs * scalars[..., np.newaxis])

        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Delegate NumPy arithmetic ufuncs back to tensor dunder operators.

        Why this method matters:
        expressions like ``ndarray * Tensor`` are executed by NumPy's ufunc
        system (e.g. ``np.multiply``). Without this hook, NumPy may fall back to
        object-style elementwise behavior and produce ``dtype=object`` arrays of
        Tensor objects instead of one correctly batched Tensor.

        We intentionally support only a narrow subset of ufunc behavior so the
        rules stay readable and predictable:
        - method must be "__call__"
        - no out= buffers
        - where must be trivially true
        - only basic arithmetic ufuncs used in this codebase
        """
        if method != "__call__":
            return NotImplemented

        # We do not support ufunc writes into Tensor outputs via out=.
        out = kwargs.get("out", None)
        if out is not None:
            out_args = out if isinstance(out, tuple) else (out,)
            if any(arg is not None for arg in out_args):
                return NotImplemented

        # Keep semantics simple: masked updates are out of scope here.
        if not self._ufunc_where_is_supported(kwargs.get("where", True)):
            return NotImplemented

        if ufunc is np.negative and len(inputs) == 1:
            operand = inputs[0]
            if isinstance(operand, Tensor):
                return -operand
            return NotImplemented

        if len(inputs) != 2:
            return NotImplemented

        left, right = inputs
        dispatch = {
            np.add: ("__add__", "__radd__"),
            np.subtract: ("__sub__", "__rsub__"),
            np.multiply: ("__mul__", "__rmul__"),
            np.true_divide: ("__truediv__", "__rtruediv__"),
            np.divide: ("__truediv__", "__rtruediv__"),
        }
        # Explicit map keeps supported ufuncs obvious and easy to audit.
        methods = dispatch.get(ufunc)
        if methods is None:
            return NotImplemented

        left_method, right_method = methods
        if isinstance(left, Tensor):
            result = getattr(left, left_method)(right)
            if result is not NotImplemented:
                return result

        if isinstance(right, Tensor):
            reverse = getattr(right, right_method, None)
            if reverse is not None:
                result = reverse(left)
                if result is not NotImplemented:
                    return result

        return NotImplemented

    def __xor__(self, other):
        """
        Wedge product (^).

        For functions (0-forms) and scalars, this is equivalent to multiplication.
        Subclasses (like Form) should override this for higher-degree forms.
        """
        if np.isscalar(other):
            return self * other

        from diffusion_geometry.tensors import Function

        if isinstance(other, Function):
            return self * other

        return NotImplemented

    def __rxor__(self, other):
        """
        Reverse wedge product ∧.
        """
        if np.isscalar(other):
            return other * self

        from diffusion_geometry.tensors import Function

        if isinstance(other, Function):
            return other * self

        return NotImplemented

    def __truediv__(self, other):
        """
        Handles:
        - Scalar division.
        - Function division (pointwise with safe inverse).
        """
        if np.isscalar(other):
            return self.space.wrap(self.coeffs / other)

        from diffusion_geometry.tensors import Function

        if isinstance(other, Function):
            self._check_arithmetic_compatibility(other, require_same_space=False)

            # Pointwise division
            # self: (..., n * C) -> (..., n, C)
            # func: (..., n) -> (..., n, 1)
            c_dim = self.space.component_dimension
            n = self.dg.n

            self_pw = self.to_pointwise_basis()
            func_pw = other.to_pointwise_basis()

            # Safe division: apply standard epsilon regularization
            eps = 1e-12
            denom_safe = np.where(np.abs(func_pw) < eps, eps, func_pw)

            # Reshape for broadcasting
            self_pw_reshaped = self_pw.reshape(self.batch_shape + (n, c_dim))
            # func_pw is (..., n*1) from to_pointwise_basis for functions
            func_pw_reshaped = denom_safe.reshape(other.batch_shape + (n, 1))

            quotient_pw = self_pw_reshaped / func_pw_reshaped

            coeffs = _from_pointwise_basis(quotient_pw, self.space)
            return self.space.wrap(coeffs)

        batch_scalars = self._broadcast_batch_scalars(other)
        if batch_scalars is not None:
            scalars, target_batch_shape = batch_scalars
            coeffs = self._broadcast_coeffs_to_batch(target_batch_shape)
            return self.space.wrap(coeffs / scalars[..., np.newaxis])

        return NotImplemented

    # -------------------------------------------------------------------------
    # Global and pointwise norms
    # -------------------------------------------------------------------------

    def norm(self) -> float:
        """Compute the L² norm of this tensor."""
        return self.dg.norm(self)

    def pointwise_norm(self) -> np.ndarray:
        """Compute the pointwise norm at each sample point."""
        return self.dg.pointwise_norm(self)
