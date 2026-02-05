from diffusion_geometry.operators.types.bilinear import BilinearOperator
from diffusion_geometry.operators.types.direct_sum import (block, hstack, vstack)
from diffusion_geometry.operators.types.linear import (LinearOperator, zero, identity)


__all__ = ["BilinearOperator", "block", "hstack",
           "vstack", "LinearOperator", "zero", "identity"]
