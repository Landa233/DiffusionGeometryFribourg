"""
Differential form tensor for diffusion geometry.

Represents a differential k-form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from opt_einsum import contract
from scipy.special import comb

# from diffusion_geometry.src import basis_utils
from diffusion_geometry.tensors.base_tensor.base_tensor import Tensor
from diffusion_geometry.tensors.tensor02.tensor02 import Tensor02
from diffusion_geometry.utils import basis_utils
from diffusion_geometry.utils.basis_conversions import _from_pointwise_basis
from diffusion_geometry.utils.batch_utils import _infer_batch_shape, compatible_batches

# from .base import Tensor, compatible_batches, _infer_batch_shape, _from_pointwise_basis


if TYPE_CHECKING:
    from .form_space import FormSpace
    from diffusion_geometry.core import DiffusionGeometry
    from diffusion_geometry.tensors import Function, VectorField


def _wedge_product(a: "Form", b: "Form") -> "Form":
    """
    Compute wedge product coefficients of two forms.

    α ∧ β : Ωᵏ¹(M) × Ωᵏ²(M) ↦ Ωᵏ¹⁺ᵏ²(M)

    Parameters
    ----------
    a : Form
        A k1-form α.
    b : Form
        A k2-form β.

    Returns
    -------
    out : Form
        The wedge product α ∧ β.
    """
    k1, k2 = a.degree, b.degree

    assert a.dg == b.dg, "Forms must be defined on the same DG object."

    assert compatible_batches(
        a.batch_shape, b.batch_shape
    ), "Forms must have compatible batch shapes."

    d = a.dg.dim
    n1 = a.dg.n_coefficients
    n = a.dg.n

    k_total = k1 + k2
    if k_total > d:
        # Result is identically zero. Since we do not support "oversized forms"
        # (degree > dimension), we return a zero function (0-form).
        return a.dg.function_space.zeros(a.batch_shape)

    # 1. Retrieve cached combinatorial data:
    # target_idx: indices K in the basis of (k1+k2)-forms
    # left_idx: indices I in the basis of k1-forms
    # right_idx: indices J in the basis of k2-forms
    # parity_signs: signs s such that dx^I ^ dx^J = s * dx^K
    (
        target_idx,
        left_idx,
        right_idx,
        parity_signs,
    ) = basis_utils.get_wedge_product_indices(d, k1, k2)

    # If k_total <= d, target_idx will always contain at least one combination.
    # The get_wedge_product_indices function handles the combinatorial mapping.

    # 2. Compute pointwise values:
    # The spectral forms are converted to their pointwise representation at
    # each data point. This allows the wedge product to be computed pointwise
    # via the component aggregation formula: (α ∧ β)_K = Σ_IJ sgn(I,J) α_I β_J
    a_pw = a.to_pointwise_basis().reshape(*a.batch_shape, n, -1)
    b_pw = b.to_pointwise_basis().reshape(*b.batch_shape, n, -1)

    # Gather components: (..., n, L) where L = n_out * n_splits
    # a_vals corresponds to coefficients a_I
    # b_vals corresponds to coefficients b_J
    a_vals = a_pw[..., left_idx]
    b_vals = b_pw[..., right_idx]

    # Compute term contributions: s ⋅ α_I ⋅ β_J
    products = a_vals * b_vals * parity_signs

    # 3. Accumulate into result basis:
    # The summation is over all splits I ∪ J = K.
    # We reshape products to segregate terms belonging to the same K.
    # number of splits = comb(k_total, k1)
    # number of targets = comb(d, k_total)
    n_out = int(comb(d, k_total))
    n_splits = int(comb(k_total, k1))

    # Reshape to (..., n, n_out, n_splits)
    products_reshaped = products.reshape(*products.shape[:-1], n_out, n_splits)

    # Sum over the splits to get coefficients for each K
    # (..., n, n_out)
    out_pw = products_reshaped.sum(axis=-1)

    # 4. Wrap result:
    # Convert pointwise tensor values back to spectral coefficients
    res = a.dg.form(out_pw, k_total)

    return res


def wedge_operator(a: "Form", l: "int"):
    """
    Given a k-form 'a' and an integer l, compute the wedge product as a linear operator
    from the space of l-forms to the space of (k+l)-forms.
    b -> a ∧ b

    Parameters
    ----------
    a : Form
        k-form with coeffs shape (n1*C(d, k),).
    l : int
        Degree of the exterior form.

    Returns
    -------
    out : ndarray
        Coefficients of the wedge product as a linear operator.
    """

    d = a.dg.dim
    k = a.degree
    n1 = a.dg.n_coefficients

    a_coeff = a.coeffs.copy()
    b_coeff = np.eye(n1 * int(comb(d, l)))

    # broadcast a_coeff to match b_coeff shape
    a_coeff_broadcasted = np.broadcast_to(
        a_coeff[None, ...], (b_coeff.shape[0],) + a_coeff.shape
    )

    a_batched = Form.from_coeffs(a_coeff_broadcasted, a.dg, k)
    b_batched = Form.from_coeffs(b_coeff, a.dg, l)
    wedge_matrix = _wedge_product(a_batched, b_batched)

    return wedge_matrix.coeffs.T


def _tensor_product_1_forms(a: "Form", b: "Form") -> "Tensor02":
    """
    Compute tensor product of two 1-forms a and b.

    Parameters
    ----------
    a : Form
        1-form.
    b : Form
        1-form.

    Returns
    -------
    out : Tensor02
        Tensor product a * b, rank (0,2).
    """
    assert a.dg == b.dg, "Forms must be defined on the same DG object."

    assert (
        a.degree == 1 and b.degree == 1
    ), "Tensor product is only defined for 1-forms."

    dg = a.dg
    n, d = dg.n, dg.dim

    # batch-shape equality (no broadcasting for tensor product logic yet)
    a_shape = a.shape[:-1]
    b_shape = b.shape[:-1]
    assert compatible_batches(
        a_shape, b_shape
    ), "Forms must have compatible batch shapes for tensor product."

    # pointwise data come back as (..., n*d); reshape to (..., n, d)
    a_pw = a.to_pointwise_basis().reshape(*a_shape, n, d)
    b_pw = b.to_pointwise_basis().reshape(*b_shape, n, d)

    # α_i(x) ⊗ β_j(x) -> (..., n, d, d)
    tensor_pw = contract("...pi,...pj->...pij", a_pw, b_pw)

    # Use flattened component dimension for Tensor02 creation
    tensor_pw_flat = tensor_pw.reshape(*tensor_pw.shape[:-2], -1)

    # Construct as pointwise (0,2)-tensor
    # Shape: (..., n, d, d)
    return dg.tensor02(tensor_pw_flat)


class Form(Tensor):
    """
    A differential k-form ω ∈ Ωᵏ(M) for k ≥ 1.

    Forms are represented as linear combinations of the basis {φ_i dx_J}.
    Here J is a multi-index (ordered tuple of indices).
    """

    def __init__(self, space: "FormSpace", coeffs: np.ndarray, degree: int):
        from .form_space import FormSpace

        if not isinstance(space, FormSpace):
            raise TypeError(f"Form requires a FormSpace; got {type(space).__name__}.")
        assert (
            space.degree == degree
        ), f"FormSpace degree mismatch: expected {space.degree}, got {degree}."

        self._degree = degree
        dg = space.dg
        expected_size = dg.n_coefficients * int(comb(dg.dim, degree))
        coeffs_arr, batch_shape = _infer_batch_shape(
            coeffs, (expected_size,), name="Form"
        )
        super().__init__(space, coeffs_arr, rank=None, batch_shape=batch_shape)

    def __repr__(self):
        dg = self.dg
        n_components = int(comb(dg.dim, self._degree))
        return (
            f"Form(degree={self._degree}, shape={self.shape}, batch_shape={self.batch_shape}, "
            f"components={n_components}, dg.n={dg.n}, dg.dim={dg.dim})"
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def degree(self) -> int:
        """Degree of the differential form."""
        return self._degree

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_coeffs(
        cls, coeffs: np.ndarray, dg: "DiffusionGeometry", degree: int
    ) -> "Form":
        """
        Create a Form from its spectral expansion coefficients.

        Parameters
        ----------
        coeffs : ndarray
            Coefficients array with trailing shape (n1 ⋅ C(d, k)).
        dg : DiffusionGeometry
            The diffusion geometry instance.
        degree : int
            The degree k of the form.

        Returns
        -------
        Form
            The form object.
        """
        space = dg.form_space(degree)
        return space.wrap(coeffs)

    @classmethod
    def from_pointwise_basis(
        cls, data: np.ndarray, dg: "DiffusionGeometry", degree: int
    ) -> "Form":
        """
        Create a Form from pointwise data.

        Parameters
        ----------
        data : ndarray
            Form data with trailing shape (dg.n, component_dimension).
        dg : DiffusionGeometry
            The diffusion geometry instance.
        degree : int
            The degree of the form.
        """
        space = dg.form_space(degree)
        coeffs = _from_pointwise_basis(data, space)
        return space.wrap(coeffs)

    # -------------------------------------------------------------------------
    # Ambient space action
    # -------------------------------------------------------------------------

    def to_ambient(self) -> np.ndarray:
        """
        Convert to ambient polyvector representation.

        We visualise k-forms by considering their action on the ambient coordinate
        vector fields, yielding a skew-symmetric tensor of signed magnitudes.
        """
        assert not self.batch_shape, "to_ambient only supports unbatched Form objects."
        return basis_utils.form_to_ambient_polyvector(self)

    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------

    def __mul__(self, other):
        """
        Multiplication by a scalar, function, or another 1-form.

        Parameters
        ----------
        other : scalar, Function, or Form
            The object to multiply with.

        Returns
        -------
        Form or Tensor02
            Result of the multiplication.
        """
        # Represents the tensor product of 1-forms if other is a Form.
        if isinstance(other, Form):
            self._check_arithmetic_compatibility(other, require_same_space=False)
            return _tensor_product_1_forms(self, other)

        # Inherits scalar and function multiplication from base class.
        return super().__mul__(other)

    def __xor__(self, other):
        """
        Compute the wedge product α ∧ β.

        Parameters
        ----------
        other : Form
            The form to wedge with.

        Returns
        -------
        Form
            The wedge product α ∧ β.
        """
        if isinstance(other, Form):
            self._check_arithmetic_compatibility(other, require_same_space=False)
            return _wedge_product(self, other)

        return super().__xor__(other)

    # -------------------------------------------------------------------------
    # Duality
    # -------------------------------------------------------------------------

    def sharp(self) -> "VectorField":
        """Dual vector field of a 1-form (musical isomorphism ♯)."""
        assert self.degree == 1, "Sharp can only be applied to 1-forms."
        return self.dg.vector_field_space.wrap(self._coeffs)

    # -------------------------------------------------------------------------
    # Action
    # -------------------------------------------------------------------------

    def __call__(self, X: "VectorField") -> "Function":
        """
        Evaluate the action of a 1-form on a vector field (interior product).

        α(X) = g(α♯, X)

        Parameters
        ----------
        X : VectorField
            The vector field to act on.

        Returns
        -------
        f : Function
            The resulting scalar function α(X).
        """
        if self.degree != 1:
            raise TypeError("Only 1-forms can act on vector fields.")
        from diffusion_geometry.tensors import VectorField

        if not isinstance(X, VectorField):
            raise TypeError("Expected a VectorField as the argument.")
        assert (
            X.space is self.dg.vector_field_space
        ), "VectorField must live in the canonical vector field space of the same DG."

        return self.dg.g(self, X.flat())

    # -------------------------------------------------------------------------
    # Differential operators
    # -------------------------------------------------------------------------

    def d(self) -> "Form":
        """Apply the exterior derivative d : Ωᵏ(M) ↦ Ωᵏ⁺¹(M)."""
        return self.dg.d(self.degree)(self)

    def codifferential(self) -> "Form":
        """Apply the codifferential operator δ : Ωᵏ(M) ↦ Ωᵏ⁻¹(M)."""
        return self.dg.codifferential(self.degree)(self)

    def up_laplacian(self) -> "Form":
        """Apply the up-Laplacian Δᵤₚ : Ωᵏ(M) ↦ Ωᵏ(M)."""
        return self.dg.up_laplacian(self.degree)(self)

    def down_laplacian(self) -> "Form":
        """Apply the down-Laplacian Δᵈᵒʷⁿ : Ωᵏ(M) ↦ Ωᵏ(M)."""
        return self.dg.down_laplacian(self.degree)(self)

    def laplacian(self) -> "Form":
        """Apply the Hodge Laplacian Δ : Ωᵏ(M) ↦ Ωᵏ(M)."""
        return self.dg.laplacian(self.degree)(self)

    def hodge_decomposition(self):
        """
        Return the Hodge decomposition components of self.

        For a k-form α, returns (exact_potential, coexact_potential, harmonic_part)
        where:
        - α = d(exact_potential) + δ(coexact_potential) + harmonic_part
        """
        k = self.degree
        dg = self.dg

        exact_potential = dg.up_laplacian(k - 1).inverse()(self.codifferential())
        exact_part = exact_potential.d()

        if k < dg.dim:
            coexact_potential = dg.down_laplacian(k + 1).inverse()(self.d())
            coexact_part = coexact_potential.codifferential()
        else:
            coexact_potential = None
            coexact_part = self.space.wrap(np.zeros_like(self.coeffs))

        harmonic_part = self - exact_part - coexact_part
        return exact_potential, coexact_potential, harmonic_part
