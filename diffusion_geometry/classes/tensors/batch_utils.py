"""
Utility functions for batch handling.
"""

from __future__ import annotations

import numpy as np


def compatible_batches(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> bool:
    """
    Check if two batch shapes are broadcast-compatible.

    Parameters
    ----------
    shape1 : tuple of int
        First batch shape.
    shape2 : tuple of int
        Second batch shape.

    Returns
    -------
    bool
        True if shapes can be broadcast together.
    """
    try:
        np.broadcast_shapes(shape1, shape2)
        return True
    except ValueError:
        return False


def _infer_batch_shape(
    array: np.ndarray, expected_tail: tuple[int, ...], *, name: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Validate trailing dimensions and infer the leading batch shape.

    Parameters
    ----------
    array : ndarray
        Input coefficient array.
    expected_tail : tuple of int
        Expected trailing dimensions (e.g., (n_function_basis,) for functions).
    name : str
        Name for error messages.

    Returns
    -------
    arr : ndarray
        The array as a numpy array.
    batch_shape : tuple of int
        The leading batch dimensions.
    """
    arr = np.asarray(array)
    assert arr.ndim >= len(
        expected_tail
    ), f"{name} coefficients must have at least one dimension"

    assert (
        arr.shape[-len(expected_tail) :] == expected_tail
    ), f"{name} coefficients must have trailing shape {expected_tail}, got {arr.shape}"

    batch_shape = arr.shape[: arr.ndim - len(expected_tail)]
    return arr, batch_shape


def _flatten_batch_dims(array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Flatten batch dimensions while preserving the last axis.

    Used to prepare data for batch operations that expect 2D arrays.

    Parameters
    ----------
    array : ndarray
        Input array with shape (..., n).

    Returns
    -------
    flat : ndarray
        Reshaped array with shape (prod(batch), n).
    batch_shape : tuple of int
        Original batch dimensions for later restoration.
    """
    assert array.ndim > 0, "Coefficient arrays must have at least one dimension"

    batch_shape = array.shape[:-1]
    flat = array.reshape((-1, array.shape[-1]))
    return flat, batch_shape


def _restore_batch_dims(flat: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    """
    Restore the batch dimensions that were flattened by `_flatten_batch_dims`.

    Parameters
    ----------
    flat : ndarray
        Flattened array with shape (prod(batch), n).
    batch_shape : tuple of int
        Original batch dimensions.

    Returns
    -------
    ndarray
        Array with shape batch_shape + (n,).
    """
    trailing = flat.shape[-1]
    if not batch_shape:
        return flat.reshape((trailing,))
    return flat.reshape(batch_shape + (trailing,))
