import pytest
import numpy as np
from functools import cached_property
from diffusion_geometry.classes.main import DiffusionGeometry
from diffusion_geometry.classes.tensors.base import compatible_batches
from diffusion_geometry.classes.markov_triples import ImmersedMarkovTriple


class MockDiffusionGeometry(DiffusionGeometry):
    def __init__(self, n=10, d=3):
        n0 = 5
        n1 = 3

        # Create mock data for the triple
        function_basis = np.random.rand(n, n0)
        measure = np.ones(n) / n
        immersion_coords = np.random.rand(n, d)

        # Mock CdC: returns 0 (not used by overrides)
        def mock_cdc(f, h):
            return np.zeros_like(f)

        # Mock Regularisation
        def mock_reg(x, **kwargs):
            return x

        triple = ImmersedMarkovTriple(
            function_basis=function_basis,
            measure=measure,
            carre_du_champ=mock_cdc,
            immersion_coords=immersion_coords,
            regularisation=mock_reg,
        )

        # Initialize parent with mock data
        super().__init__(triple=triple, n_coefficients=n1)

        self.basis_count = 3  # Unused?
        self.dim_out = 3  # Unused?

        # Mock specific internal properties used by engines if accessed,
        # or override engine methods directly if simpler.

        # Since DiffusionGeometry delegates to engines, and we want to mock behavior,
        # we can specific attributes or override the methods on DiffusionGeometry itself.
        # Overriding methods on DiffusionGeometry is cleanest given the tests call methods on dg.

        self._n = n
        self._dim = d
        self._n_coefficients = n1
        self._n0 = n0

        # Mocking for VectorField metric
        self._gamma_mixed = np.zeros((n, self.n_coefficients, self.n_coefficients, d))
        self._levi_civita_02_weak = np.zeros(
            (self.n_coefficients * d * d, self.n_coefficients * d)
        )

    @property
    def gamma_mixed(self):
        return self._gamma_mixed

    @property
    def levi_civita_02_weak(self):
        return self._levi_civita_02_weak

    @cached_property
    def gamma_coords(self):
        # Mock result of gamma_coords: (n, dim, dim)
        return np.tile(np.eye(self.dim), (self.n, 1, 1))

    # Override standard engine accessors or the methods themselves
    # DiffusionGeometry methods call self.operators_engine.d_weak etc.
    # To mock them, we can either mock the engines OR override the properties that expose keys.
    # But current DiffusionGeometry accesses engines directly in methods (e.g. d(k)).
    # Wait, d(k) calls self.operators_engine.d_weak(k).
    # So to mock d_weak, we need self.operators_engine.d_weak to be our mock.

    # We can perform a surgery on self.operators_engine after super().__init__

    # Re-mock engines
    class MockEngine:
        def d_weak(self, k):
            return np.eye(5)

        def up_delta_weak(self, k):
            return np.eye(5)

        def down_delta_weak(self, k):
            return np.eye(5)

        def gram(self, k):
            if k == 0:
                return np.eye(5)
            # n1 * C(d, k)
            import math

            dim_k = 3 * math.comb(3, k)
            return np.eye(dim_k)

        @property
        def lie_bracket_weak(self):
            return None  # Not used in these tests?

        @property
        def hessian_02_sym_weak(self):
            return None

        @property
        def levi_civita_02_weak(self):
            return np.zeros((3 * 3 * 3, 3 * 3))  # n1*d*d, n1*d?

    # Actually, simpler to just inject mocks into the engines or overwrite engines.
    pass


# To properly mock without complex engine replacement, we can override the properties/methods
# on MockDiffusionGeometry if DiffusionGeometry exposes them OR if we want to intercept them.
# However, DiffusionGeometry methods like `d(k)` use `self.operators_engine`.
# So we should swap engines.


class MockDiffusionGeometryV2(DiffusionGeometry):
    def __init__(self, n=10, d=3):
        n0 = 5
        n1 = 3
        function_basis = np.random.rand(n, n0)
        measure = np.ones(n) / n
        immersion_coords = np.random.rand(n, d)
        triple = ImmersedMarkovTriple(
            function_basis=function_basis,
            measure=measure,
            carre_du_champ=lambda f, h: np.zeros_like(f),
            immersion_coords=immersion_coords,
            regularise=lambda x, **kw: x,
        )
        super().__init__(triple=triple, n_coefficients=n1)

        # Override engines with Mocks

        self.cache.gamma_coords = np.tile(np.eye(d), (n, 1, 1))
        # gamma_mixed? DiffusionGeometry exposes it?
        # main.py properties: gamma_coords, but not gamma_mixed directly (it was delegated).
        # But wait, VectorFieldSpace used gamma_mixed?
        # VectorFieldSpace used gamma_coords.

        # Check VectorFieldSpace.metric_apply in main.py?
        # It calls _metric_apply with gamma_coords.

        self.dim_val = d


@pytest.fixture
def dg():
    return MockDiffusionGeometryV2()


def test_compatible_batches():
    assert compatible_batches((5,), (5,))
    assert compatible_batches((5,), (1,))
    assert compatible_batches((1,), (5,))
    assert compatible_batches((3, 4), (4,))
    assert compatible_batches((), (5,))

    assert not compatible_batches((5,), (6,))
    assert not compatible_batches((2, 3), (3, 2))


def test_arithmetic_broadcasting(dg):
    # Setup tensors
    # Function 1 coeffs size n0=5
    f1 = dg.function(np.random.rand(5, dg.n))
    assert f1.batch_shape == (5,)

    # Batch (1,)
    f2 = dg.function(np.random.rand(1, dg.n))
    assert f2.batch_shape == (1,)

    # Addition: (5,) + (1,) -> (5,)
    f_sum = f1 + f2
    assert f_sum.batch_shape == (5,)

    # Multiplication: (5,) * (1,) -> (5,)
    f_prod = f1 * f2
    assert f_prod.batch_shape == (5,)

    # Scalar broadcasting
    f_scalar = f1 * 2.0
    assert f_scalar.batch_shape == (5,)


def test_metric_broadcasting(dg):
    # Test g(a, b) with broadcasting

    # Vector fields
    # data shape (batch, n, dim)
    v1_data = np.random.rand(5, dg.n, dg.dim)
    v2_data = np.random.rand(1, dg.n, dg.dim)

    v1 = dg.vector_field(v1_data)
    v2 = dg.vector_field(v2_data)

    assert v1.batch_shape == (5,)
    assert v2.batch_shape == (1,)

    # g returns (batch, n)
    metric_val = dg.g(v1, v2)
    assert metric_val.shape == (5, dg.n)

    # inner returns (batch,)
    inner_val = dg.inner(v1, v2)
    assert inner_val.shape == (5,)


def test_wedge_product_broadcasting(dg):
    # Test wedge product broadcasting

    # Form 1 coeffs
    # dg.form_space(1).dim is n1 * C(d, 1) = 3 * 3 = 9
    coeff_dim = 9

    c1 = np.random.rand(5, coeff_dim)
    c2 = np.random.rand(1, coeff_dim)

    form1 = dg.form_space(1).wrap(c1)
    form2 = dg.form_space(1).wrap(c2)

    assert form1.batch_shape == (5,)
    assert form2.batch_shape == (1,)

    # Wedge product: 1-form ^ 1-form = 2-form
    joined = form1 ^ form2
    assert joined.batch_shape == (5,)
    assert joined.degree == 2


def test_failure_cases(dg):
    f1 = dg.function(np.random.rand(5, dg.n))
    f2 = dg.function(np.random.rand(6, dg.n))

    # Incompatible shapes should raise AssertionError
    with pytest.raises(AssertionError, match="Incompatible batch shapes"):
        f1 + f2

    with pytest.raises(AssertionError, match="not compatible"):
        dg.g(f1, f2)
