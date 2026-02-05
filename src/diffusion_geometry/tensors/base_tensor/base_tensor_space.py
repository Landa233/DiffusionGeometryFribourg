"""
Base tensor space descriptor for diffusion geometry.

Provides the abstract base class for all tensor spaces, defining the common
interface for metric computation, Gram matrices, and coefficient operations.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
import numpy as np

from opt_einsum import contract
from diffusion_geometry.tensors.base_tensor.metric_gram import gram


if TYPE_CHECKING:
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry
    from diffusion_geometry.tensors.direct_sum.direct_sum_space import DirectSumSpace


class BaseTensorSpace:
    """
    Base descriptor for tensor spaces (coefficient representations) on a :class:`DiffusionGeometry`.

    The inner product is defined as:
    ⟨A, B⟩ = ∫ g(A, B) dμ

    The space is represented by a basis {φ_i ⊗ e_j} where {φ_i} are
    coefficient functions and {e_j} are local basis sections.

    Subclasses include FunctionSpace, VectorFieldSpace, etc.
    """

    def __init__(self, dg: "DiffusionGeometry") -> None:
        self.dg = dg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(coeff_dimension={self.coeff_dimension})"

    def __eq__(self, other: object) -> bool:
        """Spaces are equal when type, geometry, and degree (if any) match."""
        if self is other:
            return True
        if not isinstance(other, BaseTensorSpace):
            return NotImplemented
        if type(self) is not type(other):
            return False
        if self.dg != other.dg:
            return False
        if self.degree is None:
            return True
        return self.degree == other.degree

    def __hash__(self) -> int:
        return hash((type(self), self.dg, self.degree))

    def __contains__(self, item) -> bool:
        if hasattr(item, "space"):
            return item.space == self
        return False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def degree(self) -> int | None:
        """Differential degree for forms (``None`` otherwise)."""
        return None

    @property
    def cdc_components(self) -> np.ndarray:
        """
        Pointwise carré du champ components Γ(x) for this space, from which we
        can compute the metric tensor g and the Gram matrix.

        Returns
        -------
        (n, C, C) array
            The pointwise metric tensor mapping coefficients to pointwise inner products.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must define 'cdc_components'."
        )

    @property
    def component_dimension(self) -> int:
        """Number of pointwise components represented per spatial sample."""
        return self.cdc_components.shape[1]

    @property
    def coeff_dimension(self) -> int:
        """Number of coefficients required to represent an element."""
        return self.dg.n_coefficients * self.component_dimension

    # -------------------------------------------------------------------------
    # Direct sum space construction
    # -------------------------------------------------------------------------

    def __add__(self, other: object) -> "DirectSumSpace":
        from diffusion_geometry.tensors.direct_sum.direct_sum_space import DirectSumSpace

        if not isinstance(other, BaseTensorSpace):
            return NotImplemented
        assert (
            self.dg == other.dg
        ), "Cannot form direct sum of spaces from different DiffusionGeometry instances"
        left = self.spaces if isinstance(self, DirectSumSpace) else (self,)
        right = other.spaces if isinstance(other, DirectSumSpace) else (other,)
        return DirectSumSpace(self.dg, left + right)

    # -------------------------------------------------------------------------
    # Riemannian metric
    # -------------------------------------------------------------------------

    @property
    def metric(self) -> "np.ndarray":
        """
        Metric tensor (usually not needed, use metric_apply instead).

        Returns
        -------
        (n, n1*C, n1*C) array
            Metric tensor at each point.
        """
        from diffusion_geometry.tensors.base_tensor.metric_gram import metric

        u_coeffs = self.dg.function_basis[:, : self.dg.n_coefficients]
        return metric(u_coeffs, self.cdc_components)

    def metric_apply(
        self, a_coeffs: "np.ndarray", b_coeffs: "np.ndarray"
    ) -> "np.ndarray":
        """Apply the metric to two coefficient vectors."""
        from diffusion_geometry.tensors.base_tensor.metric_gram import _metric_apply

        return _metric_apply(
            self.dg.function_basis[:, : self.dg.n_coefficients],
            self.dg._regularise,
            a_coeffs,
            b_coeffs,
            self.cdc_components,
        )

    # -------------------------------------------------------------------------
    # Gram matrix and basis
    # -------------------------------------------------------------------------

    @property
    def gram(self) -> "np.ndarray":
        """
        Gram matrix.

        Returns
        -------
        (n1*C, n1*C) array
            Gram matrix.
        """
        u_coeffs = self.dg.function_basis[:, : self.dg.n_coefficients]
        return gram(u_coeffs, self.cdc_components, self.dg.triple.measure)

    @cached_property
    def _is_orthonormal(self) -> bool:
        """Check if the Gram matrix is approximately the identity matrix."""
        eye = np.eye(self.gram.shape[0], dtype=self.gram.dtype)
        return np.allclose(self.gram, eye, atol=1e-10)

    @cached_property
    def _gram_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors of the Gram matrix with spectral cutoff."""
        evals, evecs = np.linalg.eigh(self.gram)
        keep = evals > self.dg.rcond
        return evals[keep], evecs[:, keep]

    @cached_property
    def gram_inv(self) -> np.ndarray:
        """Moore-Penrose pseudoinverse of the Gram matrix with spectral cutoff."""
        evals, evecs = self._gram_spectrum
        if evecs.size == 0:
            size = self.gram.shape[0]
            return np.zeros((size, size))
        # Shape: (N, K), (K, N) -> (N, N)
        return contract("ik,kj->ij", evecs / evals, evecs.T)

    @cached_property
    def orthonormal_basis(self) -> np.ndarray:
        """Orthonormal basis of the spectrally cutoff subspace."""
        evals, evecs = self._gram_spectrum
        if evecs.size == 0:
            return np.zeros((self.gram.shape[0], 0))
        return evecs / np.sqrt(evals)

    # -------------------------------------------------------------------------
    # Create elements
    # -------------------------------------------------------------------------

    def wrap(self, coeffs: np.ndarray):
        raise NotImplementedError

    def zeros(self, shape: tuple[int, ...] = ()) -> np.ndarray:
        """
        Create an element with all zero coefficients.

        Parameters
        ----------
        shape : tuple of int, optional
            Batch shape for the tensor. Default is () for a single element.

        Returns
        -------
        Tensor
            A tensor element of this space with zero coefficients.
        """
        coeff_shape = shape + (self.coeff_dimension,)
        coeffs = np.zeros(coeff_shape)
        return self.wrap(coeffs)
