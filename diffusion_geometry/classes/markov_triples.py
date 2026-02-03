from typing import Optional, Callable
import numpy as np


class MarkovTriple:
    """
    Data container for the Markov Triple (M, mu, K).
    """

    def __init__(
        self,
        function_basis: np.ndarray,
        measure: np.ndarray,
        carre_du_champ: Callable[[np.ndarray, np.ndarray], np.ndarray],
        regularise: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.function_basis = function_basis
        self.measure = measure
        self._cdc = carre_du_champ
        self._regularise = regularise
        self.n, self.n_function_basis = function_basis.shape

    def cdc(self, f: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self._cdc(f, h)

    def regularise(self, x: np.ndarray) -> np.ndarray:
        if self._regularise is None:
            return x
        return self._regularise(x)


class EmbeddedMarkovTriple(MarkovTriple):
    """
    Markov Triple with an embedding into Euclidean space.
    """

    def __init__(
        self,
        function_basis: np.ndarray,
        measure: np.ndarray,
        carre_du_champ: Callable[[np.ndarray, np.ndarray], np.ndarray],
        embedding_coords: np.ndarray,
        data_matrix: Optional[np.ndarray] = None,
        regularise: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(
            function_basis=function_basis,
            measure=measure,
            carre_du_champ=carre_du_champ,
            regularise=regularise,
        )
        self.embedding_coords = embedding_coords
        self.data_matrix = data_matrix
        self.dim = self.embedding_coords.shape[1]
