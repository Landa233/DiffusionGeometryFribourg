from typing import TYPE_CHECKING, Sequence

import numpy as np

from .linear import LinearOperator


if TYPE_CHECKING:
    from diffusion_geometry.tensors import BaseTensorSpace


def _sum_spaces(spaces: Sequence["BaseTensorSpace"]) -> "BaseTensorSpace":
    """Direct sum of multiple tensor spaces."""
    iterator = iter(spaces)
    result = next(iterator)
    for space in iterator:
        result = result + space
    return result


def block(block_rows: Sequence[Sequence[LinearOperator]]) -> LinearOperator:
    """
    Construct a block operator from a grid of LinearOperators.

    Expects a rectangular grid where:
    - All operators in column j share the same domain.
    - All operators in row i share the same codomain.

    Parameters
    ----------
    block_rows : Sequence[Sequence[LinearOperator]]
        Rows of operators [[A, B], [C, D]].

    Returns
    -------
    LinearOperator
        The block operator acting on the direct sum of domains to the direct sum of codomains.
    """
    assert block_rows, "Provide at least one row of operator blocks"

    rows = len(block_rows)
    assert block_rows[0], "Provide at least one column of operator blocks"
    cols = len(block_rows[0])

    # 1. Validate grid structure and types
    for row in block_rows:
        assert len(row) == cols, "Operator block matrix must be rectangular"
        for op in row:
            if not isinstance(op, LinearOperator):
                raise TypeError("All blocks must be LinearOperator instances")

    # 2. Validate consistency of domains (columns) and codomains (rows)
    domains = []
    for j in range(cols):
        dom = block_rows[0][j].domain
        for i in range(1, rows):
            assert (
                block_rows[i][j].domain == dom
            ), f"Column {j} has inconsistent domains"
        domains.append(dom)

    codomains = []
    for i in range(rows):
        cod = block_rows[i][0].codomain
        for j in range(1, cols):
            assert (
                block_rows[i][j].codomain == cod
            ), f"Row {i} has inconsistent codomains"
        codomains.append(cod)

    # 3. Construct combined spaces
    full_domain = domains[0] if len(domains) == 1 else _sum_spaces(domains)
    full_codomain = codomains[0] if len(codomains) == 1 else _sum_spaces(codomains)

    # 4. Assemble weak matrix
    # Concatenate weak matrices to form the global weak matrix.
    weak_blocks = [[op.weak for op in row] for row in block_rows]
    full_weak = np.block(weak_blocks)

    return LinearOperator(
        domain=full_domain, codomain=full_codomain, weak_matrix=full_weak
    )


def hstack(blocks: Sequence[LinearOperator]) -> LinearOperator:
    """Horizontally stack operators: [A, B]."""
    assert blocks, "hstack requires at least one operator"
    return block([list(blocks)])


def vstack(blocks: Sequence[LinearOperator]) -> LinearOperator:
    """Vertically stack operators: [[A], [B]]."""
    assert blocks, "vstack requires at least one operator"
    return block([[op] for op in blocks])
