"""
Tensor wrappers for diffusion geometry.

This package provides object-oriented wrappers for diffusion geometry tensors,
including functions, vector fields, differential forms, and (0,2)-tensors.

Classes
-------
Tensor
    Abstract base class for all tensors.
DirectSumElement
    Element in a direct sum of tensor spaces.
Function
    Scalar function (0-form).
VectorField
    Vector field.
Form
    Differential k-form.
Tensor02
    General (0,2)-tensor.
Tensor02Sym
    Symmetric (0,2)-tensor.
"""

from .base import (
    Tensor,
    _infer_batch_shape,
    _flatten_batch_dims,
    _restore_batch_dims,
    compatible_batches,
    _from_pointwise_basis,
)
from .direct_sum_element import DirectSumElement
from .function import Function
from .vector_field import VectorField
from .form import Form
from .tensor02 import Tensor02
from .tensor02sym import Tensor02Sym

__all__ = [
    "Tensor",
    "DirectSumElement",
    "Function",
    "VectorField",
    "Form",
    "Tensor02",
    "Tensor02Sym",
    "_infer_batch_shape",
    "_flatten_batch_dims",
    "_restore_batch_dims",
    "compatible_batches",
    "_from_pointwise_basis",
]
