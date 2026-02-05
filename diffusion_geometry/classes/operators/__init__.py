"""
Linear and bilinear operator classes and constructors for diffusion geometry.
"""

from .linear import LinearOperator, identity, id, zero
from .bilinear import BilinearOperator
from .direct_sum import block, hstack, vstack
