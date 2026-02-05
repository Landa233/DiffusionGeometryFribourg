from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from ..tensors import _flatten_batch_dims, _restore_batch_dims

if TYPE_CHECKING:
    from ..tensor_spaces import BaseTensorSpace, DiffusionGeometry


class LinearOperator:
    """
    Linear operator L: V ↦ W acting on coefficient spaces.

    Represented by a matrix relative to the bases of V and W.
    - Weak form: B(w, v) = ⟨w, L(v)⟩_W
    - Strong form: Strong matrix L(v)
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        domain: "BaseTensorSpace",
        codomain: "BaseTensorSpace",
        weak_matrix: np.ndarray | None = None,
        strong_matrix: np.ndarray | None = None,
    ):
        self.domain = domain
        self.codomain = codomain
        self._weak = weak_matrix
        self._strong = strong_matrix
        if weak_matrix is None and strong_matrix is None:
            raise ValueError("Provide at least one of weak_matrix or strong_matrix")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the operator matrix."""
        return self.matrix.shape

    def __array__(self, dtype=None):
        if dtype is None:
            return np.asarray(self.matrix)
        return np.asarray(self.matrix, dtype=dtype)

    @cached_property
    def weak(self) -> np.ndarray:
        """The operator matrix in the weak form (bilinear form) relative to the bases."""
        if self._weak is not None:
            return self._weak
        return self.codomain.gram @ self._strong

    @cached_property
    def matrix(self) -> np.ndarray:
        """The operator matrix in the strong form relative to the bases."""
        if self._strong is not None:
            return self._strong
        return self.codomain.gram_inv @ self._weak

    @cached_property
    def adjoint(self) -> "LinearOperator":
        """The adjoint linear operator L* : W ↦ V."""
        return LinearOperator(
            domain=self.codomain,
            codomain=self.domain,
            weak_matrix=self.weak.conj().T,
        )

    @cached_property
    def is_self_adjoint(self) -> bool | None:
        """True if the operator is self-adjoint (L = L*). Only defined for endomorphisms."""
        if self.domain != self.codomain:
            return None
        return np.allclose(self.weak, self.weak.conj().T, atol=1e-12)

    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------

    def _check_arithmetic_compatibility(self, other) -> bool:
        if not isinstance(other, LinearOperator):
            return False
        assert (
            self.domain == other.domain and self.codomain == other.codomain
        ), "Operators must share domain and codomain for arithmetic"
        return True

    def __add__(self, other: "LinearOperator") -> "LinearOperator":
        if self._check_arithmetic_compatibility(other):
            return LinearOperator(
                domain=self.domain,
                codomain=self.codomain,
                weak_matrix=self.weak + other.weak,
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: "LinearOperator") -> "LinearOperator":
        """Generic subtraction: self + (-other)."""
        return self + (-other)

    def __rsub__(self, other):
        if isinstance(other, LinearOperator):
            return other.__sub__(self)
        return NotImplemented

    def __mul__(self, scalar) -> "LinearOperator":
        if np.isscalar(scalar) or (
            isinstance(scalar, np.ndarray) and scalar.shape == ()
        ):
            return LinearOperator(
                domain=self.domain,
                codomain=self.codomain,
                weak_matrix=self.weak * scalar,
            )
        return NotImplemented

    def __rmul__(self, scalar) -> "LinearOperator":
        return self.__mul__(scalar)

    def __neg__(self) -> "LinearOperator":
        return (-1) * self

    def __matmul__(self, other: "LinearOperator") -> "LinearOperator":
        """Compose two linear operators (self after other)."""
        if not isinstance(other, LinearOperator):
            return NotImplemented
        assert (
            other.codomain == self.domain
        ), f"Incompatible spaces: {other.codomain!r} -> {self.domain!r}."

        # Compose in weak form to avoid inverting Gram matrices unnecessarily.
        return LinearOperator(
            domain=other.domain,
            codomain=self.codomain,
            weak_matrix=self.weak @ other.matrix,
        )

    # -------------------------------------------------------------------------
    # Spectral Theory
    # -------------------------------------------------------------------------

    def _spectral_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the eigenvalues and eigenvector coordinates in the orthonormal basis.

        Returns:
            eigvals: (K,) array of eigenvalues.
            eigvecs_coords: (K, K) matrix where columns are eigenvectors in the
                orthonormal basis coordinates.
        """
        assert (
            self.domain == self.codomain
        ), "Spectral decomposition is defined only for endomorphisms"

        # Use the truncated orthonormal basis of the domain to ensure numerical
        # stability and exploit self-adjointness.
        basis = self.domain.orthonormal_basis  # (N, K)
        if basis.size == 0:
            return np.zeros(0, dtype=self.matrix.dtype), np.zeros(
                (0, 0), dtype=self.matrix.dtype
            )

        # Restricted operator in the orthonormal basis: A = Φ* W Φ
        # where Φ is the orthonormal basis and W is the weak matrix.
        # Shape: (K, N) @ (N, N) @ (N, K) -> (K, K)
        weak_conjugated = basis.conj().T @ self.weak @ basis

        if self.is_self_adjoint:
            eigvals, eigvecs_coords = np.linalg.eigh(weak_conjugated)
        else:
            eigvals, eigvecs_coords = np.linalg.eig(weak_conjugated)

        # Sort eigenvalues and eigenvectors
        if eigvals.ndim == 1 and eigvals.size:
            if np.iscomplexobj(eigvals):
                magnitudes = np.abs(eigvals)
                order = np.lexsort((eigvals.imag, eigvals.real, magnitudes))
            else:
                order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs_coords = eigvecs_coords[:, order]

        return np.real_if_close(eigvals), eigvecs_coords

    def spectrum(self, *, eigvals_only: bool = False):
        """
        Compute the spectrum of the operator if it is an endomorphism.

        Parameters
        ----------
        eigvals_only : bool, optional
            If True, only return the eigenvalues. Default is False.

        Returns
        -------
        eigvals : ndarray
            The eigenvalues.
        eigenvectors : Tensor, optional
            A batched tensor object containing the eigenvectors as its batch.
        """
        eigvals, eigvecs_coords = self._spectral_decomposition()

        if eigvals_only:
            return eigvals

        if eigvals.size == 0:
            coeff_dim = self.domain.gram.shape[0]
            empty = np.zeros((0, coeff_dim), dtype=self.matrix.dtype)
            return eigvals, self.domain.wrap(empty)

        # Project coordinates back to the full coefficient space: V = Φ @ eigvecs_coords
        # Shape: (N, K) @ (K, K) -> (N, K)
        eigenvectors = self.domain.orthonormal_basis @ eigvecs_coords
        eigenvectors = np.real_if_close(eigenvectors.T)
        return eigvals, self.domain.wrap(eigenvectors)

    def inverse(self, *, rcond: float | None = None) -> "LinearOperator":
        """Return the spectral pseudo-inverse of the operator."""
        eigvals, eigvecs_coords = self._spectral_decomposition()

        if eigvals.size == 0:
            return LinearOperator(
                domain=self.domain,
                codomain=self.codomain,
                strong_matrix=np.zeros_like(self.matrix),
            )

        if rcond is None:
            rcond = getattr(self.domain.dg, "rcond", 0.0)

        mask = np.abs(eigvals) > rcond
        if not np.any(mask):
            return LinearOperator(
                domain=self.domain,
                codomain=self.codomain,
                strong_matrix=np.zeros_like(self.matrix),
            )

        # Compute pseudo-inverse in the orthonormal basis coordinates
        # Case 1: Self-adjoint (orthonormal eigvecs) -> A⁻¹ = Σ λᵢ⁻¹ vᵢ vᵢ*
        if self.is_self_adjoint:
            eigvecs_kept = eigvecs_coords[:, mask]  # (K, K_kept)
            inv_vals = 1.0 / eigvals[mask]  # (K_kept,)
            # (K, K_kept) * (K_kept,) @ (K_kept, K) -> (K, K)
            inv_coords = (eigvecs_kept * inv_vals) @ eigvecs_kept.conj().T
        # Case 2: Non-self-adjoint -> A⁻¹ = V Λ⁻¹ V⁻¹
        else:
            inv_vals = np.zeros_like(eigvals)
            inv_vals[mask] = 1.0 / eigvals[mask]
            # V Λ⁻¹ (K, K)
            weighted = eigvecs_coords * inv_vals
            # Matrix V⁻¹ via solving V* X = I
            eigvecs_inv = np.linalg.solve(
                eigvecs_coords.T, np.eye(eigvecs_coords.shape[0])
            ).T
            inv_coords = weighted @ eigvecs_inv

        # Lift the inverse back to the original coefficient space:
        # L⁻¹ = Φ @ inv_coords @ Φ* @ G
        # Shape: (N, K) @ (K, K) @ (K, N) @ (N, N) -> (N, N)
        basis = self.domain.orthonormal_basis
        strong_inverse = basis @ inv_coords @ (basis.conj().T @ self.domain.gram)
        strong_inverse = np.real_if_close(strong_inverse)

        return LinearOperator(
            domain=self.domain,
            codomain=self.codomain,
            strong_matrix=strong_inverse,
        )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    def __call__(self, tensor_obj):
        """
        Apply the operator to a tensor object.

        Parameters
        ----------
        tensor_obj : Tensor
            The tensor to act on. Must belong to the operator's domain.

        Returns
        -------
        Tensor
            The resulting tensor in the operator's codomain.
        """
        if not hasattr(tensor_obj, "space"):
            raise TypeError(
                f"Expected a tensor-like object with a 'space' attribute, "
                f"got {type(tensor_obj).__name__}"
            )
        assert (
            tensor_obj.space == self.domain
        ), f"Input tensor belongs to {tensor_obj.space}, expected {self.domain}."

        coeffs_flat, batch_shape = _flatten_batch_dims(tensor_obj.coeffs)
        result_flat = coeffs_flat @ self.matrix.T
        result = _restore_batch_dims(result_flat, batch_shape)
        return self.codomain.wrap(result)


def zero(
    domain: "BaseTensorSpace",
    codomain: "BaseTensorSpace" = None,
    *,
    dtype=None,
) -> LinearOperator:
    """Return the zero operator from ``domain`` to ``codomain``."""

    if codomain is None:
        codomain = domain
    if dtype is None:
        dtype = np.result_type(domain.gram.dtype, codomain.gram.dtype)

    weak_matrix = np.zeros(
        (codomain.coeff_dimension, domain.coeff_dimension), dtype=dtype
    )
    return LinearOperator(domain=domain, codomain=codomain, weak_matrix=weak_matrix)


def identity(
    space: "BaseTensorSpace",
    *,
    dtype=None,
) -> LinearOperator:
    """Return the identity operator on ``space``."""

    if dtype is None:
        dtype = space.gram.dtype

    strong_matrix = np.eye(space.coeff_dimension, dtype=dtype)
    return LinearOperator(domain=space, codomain=space, strong_matrix=strong_matrix)


id = identity
