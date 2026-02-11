from .core.geometry.diffusion_geometry import DiffusionGeometry
from .tensors.functions.function import Function
from .tensors.vector_fields.vector_field import VectorField
from .tensors.forms.form import Form
from .tensors.tensor02.tensor02 import Tensor02
from .tensors.tensor02sym.tensor02sym import Tensor02Sym

from .tensors.functions.function_space import FunctionSpace
from .tensors.vector_fields.vector_field_space import VectorFieldSpace
from .tensors.forms.form_space import FormSpace
from .tensors.tensor02.tensor02_space import Tensor02Space
from .tensors.tensor02sym.tensor02sym_space import Tensor02SymSpace

from .operators.types.linear import LinearOperator
from .operators.types.bilinear import BilinearOperator

__all__ = [
    "DiffusionGeometry",
    "Function",
    "VectorField",
    "Form",
    "Tensor02",
    "Tensor02Sym",
    "FunctionSpace",
    "VectorFieldSpace",
    "FormSpace",
    "Tensor02Space",
    "Tensor02SymSpace",
    "LinearOperator",
    "BilinearOperator",
]
