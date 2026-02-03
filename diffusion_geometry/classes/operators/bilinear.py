from functools import cached_property
from typing import TYPE_CHECKING, Optional

import numpy as np
from opt_einsum import contract

from ..tensors import _flatten_batch_dims, _restore_batch_dims, compatible_batches
from .linear import LinearOperator

if TYPE_CHECKING:
    from ..tensor_spaces import BaseTensorSpace


class BilinearOperator:
    """
    Bilinear operator B: V × U ↦ W.

    Can be partially applied on the left to yield a LinearOperator B(v, ⋅) : U ↦ W.
    """

    def __init__(
        self,
        domain_a: "BaseTensorSpace",
        domain_b: "BaseTensorSpace",
        codomain: "BaseTensorSpace",
        *,
        weak_tensor: np.ndarray | None = None,
        strong_tensor: np.ndarray | None = None,
    ):
        self._domain_a = domain_a
        self._domain_b = domain_b
        self._codomain = codomain
        self._weak = weak_tensor
        self._strong = strong_tensor

        if weak_tensor is None and strong_tensor is None:
            raise ValueError("Provide at least one of weak_tensor or strong_tensor")

        # Expected component shape: (codim, dim_a, dim_b)
        self._component_shape = (
            codomain.coeff_dimension,
            domain_a.coeff_dimension,
            domain_b.coeff_dimension,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def domain_a(self) -> "BaseTensorSpace":
        """The first input space V."""
        return self._domain_a

    @property
    def domain_b(self) -> "BaseTensorSpace":
        """The second input space U."""
        return self._domain_b

    @property
    def codomain(self) -> "BaseTensorSpace":
        """The output space W."""
        return self._codomain

    @property
    def component_shape(self) -> tuple[int, int, int]:
        """Shape of the operator components (codim, dim_a, dim_b)."""
        return self._component_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the strong tensor."""
        return self.strong.shape

    @cached_property
    def weak(self) -> np.ndarray:
        """Weak form tensor."""
        if self._weak is not None:
            return self._weak
        # weak_iAB = gram_ij * strong_jAB
        # Shape: (codim, codim), (..., codim, dim_a, dim_b) -> (..., codim, dim_a, dim_b)
        return contract("iC,...CAB->...iAB", self.codomain.gram, self.strong)

    @cached_property
    def strong(self) -> np.ndarray:
        """Strong form tensor."""
        if self._strong is not None:
            return self._strong
        # strong^iAB = gram_inv^ij * weak_jAB
        # Shape: (codim, codim), (..., codim, dim_a, dim_b) -> (..., codim, dim_a, dim_b)
        return contract("iC,...CAB->...iAB", self.codomain.gram_inv, self.weak)

    def __array__(self, dtype=None):
        # We allow dtype for compatibility with np.asarray() calls.
        return np.asarray(self.strong, dtype=dtype)

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    def __call__(self, x, y=None):
        """
        Apply the bilinear operator.

        If y is None, performs partial application A(x, .) -> LinearOperator.
        If y is provided, performs full application A(x, y) -> Tensor.
        """
        if not hasattr(x, "space"):
            raise TypeError(f"Expected tensor object, got {type(x)}")

        assert (
            x.space == self.domain_a
        ), f"Domain A mismatch: tensor in {x.space}, operator expects {self.domain_a}"

        if y is None:
            return self._partial_apply(x)

        return self._full_apply(x, y)

    def _partial_apply(self, x) -> LinearOperator:
        """Return LinearOperator L(y) = B(x, y)."""
        if x.coeffs.ndim > 1:
            raise ValueError("Batched partial application not supported.")

        # Shape: (codim, dim_a, dim_b), (dim_a) -> (codim, dim_b)
        weak_matrix = contract("iAB,A->iB", self.weak, x.coeffs)

        return LinearOperator(
            domain=self.domain_b, codomain=self.codomain, weak_matrix=weak_matrix
        )

    def _full_apply(self, x, y) -> object:  # Returns Tensor
        """
        Evaluate the bilinear form B(x, y).

        Parameters
        ----------
        x : Tensor
            Tensor in domain_a.
        y : Tensor
            Tensor in domain_b.

        Returns
        -------
        Tensor
            The resulting tensor in the codomain.
        """
        if not hasattr(y, "space"):
            raise TypeError(f"Expected tensor object, got {type(y)}")

        assert (
            y.space == self.domain_b
        ), f"Domain B mismatch: tensor in {y.space}, operator expects {self.domain_b}"

        assert compatible_batches(
            x.batch_shape, y.batch_shape
        ), f"Batch shape mismatch: x has {x.batch_shape}, y has {y.batch_shape}"

        # Handles batches via broadcasting if implemented carefully, but opt_einsum handles it well.
        # Shape: (..., codim, dim_a, dim_b), (..., dim_a), (..., dim_b) -> (..., codim)
        val = contract("...iAB,...A,...B->...i", self.strong, x.coeffs, y.coeffs)

        return self.codomain.wrap(val)

    # -------------------------------------------------------------------------
    # Transpose
    # -------------------------------------------------------------------------

    def transpose(self) -> "BilinearOperator":
        """
        Return the transpose of this bilinear operator.

        The transpose B^T: V × U → W is defined by B^T(y, x) = B(x, y).

        Returns
        -------
        B_T : BilinearOperator
            The transposed operator with swapped input domains.
        """

        if self._weak is not None:
            weak = np.swapaxes(self._weak, -1, -2)
            return BilinearOperator(
                domain_a=self.domain_b,
                domain_b=self.domain_a,
                codomain=self.codomain,
                weak_tensor=weak,
            )

        if self._strong is not None:
            strong = np.swapaxes(self._strong, -1, -2)
            return BilinearOperator(
                domain_a=self.domain_b,
                domain_b=self.domain_a,
                codomain=self.codomain,
                strong_tensor=strong,
            )

    @property
    def T(self) -> "BilinearOperator":
        """Alias for transpose()."""
        return self.transpose()
