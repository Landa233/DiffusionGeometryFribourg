"""
Tensor space descriptors for diffusion geometry.

This package provides space descriptors that define the vector spaces
for each tensor type: dimension, metric, Gram matrix, and coefficient handling.

Classes
-------
BaseTensorSpace
    Abstract base class for all tensor spaces.
FunctionSpace
    Space of scalar functions.
VectorFieldSpace
    Space of vector fields.
FormSpace
    Space of differential k-forms.
Tensor02Space
    Space of general (0,2)-tensors.
Tensor02SymSpace
    Space of symmetric (0,2)-tensors.
DirectSumSpace
    Direct sum of tensor spaces.
"""

from .base import BaseTensorSpace
from .direct_sum_space import DirectSumSpace
from .form_space import FormSpace
from .function_space import FunctionSpace
from .tensor02_space import Tensor02Space
from .tensor02sym_space import Tensor02SymSpace
from .vector_field_space import VectorFieldSpace

__all__ = [
    "BaseTensorSpace",
    "FunctionSpace",
    "VectorFieldSpace",
    "FormSpace",
    "Tensor02Space",
    "Tensor02SymSpace",
    "DirectSumSpace",
]
