from typing import Optional, Callable
import numpy as np


class MarkovTriple:
    """
    Data container for a Markov triple (M, μ, Γ) in coefficient form.

    This object stores:
    - coefficient functions {φ_i} sampled at n points,
    - a discrete measure μ on those points,
    - a carré du champ callback Γ(f, h) acting on pointwise function values.

    Shape notation:
    - n: number of sampled points.
    - n0: number of basis functions (n_function_basis).
    """

    def __init__(
        self,
        function_basis: np.ndarray,
        measure: np.ndarray,
        carre_du_champ: Callable[[np.ndarray, np.ndarray], np.ndarray],
        regularise: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialise a Markov triple.

        Args:
            function_basis: Array of shape (n, n0) with columns equal to
                coefficient functions φ_i evaluated at sample points.
            measure: Array of shape (n,) containing pointwise integration
                weights / stationary measure values.
            carre_du_champ: Callable implementing Γ(f, h) for arrays f
                and h with leading point axis n. The return should have
                the same broadcasted shape as f and h.
            regularise: Optional callable that maps an input array to a
                regularised array of the same shape.
        """
        self.function_basis = function_basis
        self.measure = measure
        self._cdc = carre_du_champ
        self._regularise = regularise
        self.n, self.n_function_basis = function_basis.shape

    def cdc(self, f: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Evaluate the carré du champ Γ(f, h).

        Args:
            f: Array with leading shape (n, ...).
            h: Array with leading shape (n, ...), broadcast-compatible with f.

        Returns:
            Array with the broadcasted shape of f and h (leading axis n).
        """
        return self._cdc(f, h)

    def regularise(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the optional regularisation map.

        Args:
            x: Input array of arbitrary shape.

        Returns:
            If no regulariser is set, returns x unchanged. Otherwise returns
            regularise(x), typically with the same shape as x.
        """
        if self._regularise is None:
            return x
        return self._regularise(x)


class ImmersedMarkovTriple(MarkovTriple):
    """
    Markov triple with an immersion into Euclidean ambient coordinates.

    In addition to :class:`MarkovTriple`, this class stores coordinates
    x : M → R^d. These immersion coordinates are required to generate the
    tensor algebra as an A-module via coordinate differentials and their
    tensor products over the function algebra A.

    Shape notation:
    - n: number of sampled points.
    - n0: number of function basis elements.
    - d: ambient coordinate dimension.
    """

    def __init__(
        self,
        function_basis: np.ndarray,
        measure: np.ndarray,
        carre_du_champ: Callable[[np.ndarray, np.ndarray], np.ndarray],
        immersion_coords: np.ndarray,
        data_matrix: Optional[np.ndarray] = None,
        regularise: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialise an immersed Markov triple.

        Args:
            function_basis: Array of shape (n, n0).
            measure: Array of shape (n,).
            carre_du_champ: Carré du champ callable as in :class:`MarkovTriple`.
            immersion_coords: Ambient coordinates of shape (n, d) used as
                the immersion / embedding; required to generate the tensor
                algebra as an A-module.
            data_matrix: Optional raw data matrix associated with the same
                points, typically shape (n, p) for some feature dimension p.
            regularise: Optional regularisation callable passed to the base class.
        """
        super().__init__(
            function_basis=function_basis,
            measure=measure,
            carre_du_champ=carre_du_champ,
            regularise=regularise,
        )
        self.immersion_coords = immersion_coords
        self.data_matrix = data_matrix
        self.dim = self.immersion_coords.shape[1]
