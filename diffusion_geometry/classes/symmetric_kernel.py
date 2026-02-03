from typing import Optional, Literal, Tuple, Callable
import numpy as np
from diffusion_geometry.src import diffusion_process, carre_du_champ


class SymmetricKernelConstructor:
    """
    Resolves graph geometric data (kernel, bandwidths, indices) by filling in missing values
    using a symmetric kernel.

    resolve_measure: if μ is not given, we use the stationary distribution of the Markov chain.
    resolve_function_basis: if {φ_i} is not given, we compute the first n_function_basis
        coefficient functions of the Markov chain.
    resolve_immersion: if immersion_coords is not given, we compute them by regularising data_matrix.
    """

    def __init__(
        self,
        nbr_indices: np.ndarray,
        kernel: np.ndarray,
        bandwidths: Optional[np.ndarray] = None,
        use_mean_centres: bool = True,
    ):
        self.nbr_indices = np.asarray(nbr_indices)
        self.kernel = np.asarray(kernel)
        self.bandwidths = np.asarray(bandwidths) if bandwidths is not None else None
        self.use_mean_centres = use_mean_centres

        # Lazy caches
        self._K_sym: Optional[np.ndarray] = None
        self._row_sums: Optional[np.ndarray] = None

    @property
    def symmetric_kernel_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lazily computes (K_sym, row_sums).
        """
        if self._K_sym is None:
            self._K_sym, self._row_sums = (
                diffusion_process.build_symmetric_kernel_matrix(
                    self.kernel, self.nbr_indices
                )
            )
        return self._K_sym, self._row_sums

    def resolve_measure(self, mu: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Resolves the invariant measure mu.
        If mu is not given, we use the stationary distribution of the Markov chain.
        """
        if mu is None:
            K_sym, row_sums = self.symmetric_kernel_data
            mu = row_sums / row_sums.sum()
        return np.asarray(mu)

    def resolve_function_basis(
        self, n_function_basis: int, function_basis: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Resolves the function basis {φ_i}.
        If {φ_i} is not given, we compute the first n_function_basis
        coefficient functions of the Markov chain.
        """
        if function_basis is None:
            K_sym, row_sums = self.symmetric_kernel_data
            function_basis = diffusion_process.compute_eigenfunction_basis(
                K_sym, row_sums, n0=int(n_function_basis)
            )
        return np.asarray(function_basis)

    def resolve_immersion(
        self,
        regularise: Callable,
        data_matrix: Optional[np.ndarray] = None,
        immersion_coords: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Resolves immersion coordinates.
        If missing, computes them by regularising data_matrix.
        """
        if immersion_coords is not None:
            return np.asarray(immersion_coords)

        if data_matrix is None:
            # We have neither immersion_coords nor data_matrix
            raise ValueError("data_matrix and/or immersion_coords must be provided.")

        # Compute immersion coords by regularising data_matrix
        return regularise(np.asarray(data_matrix))
