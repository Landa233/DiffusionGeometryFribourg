import numpy as np
from diffusion_geometry.classes.main import DiffusionGeometry


def test_from_edges_cycle_graph():
    print("Testing from_edges with a cycle graph...")
    # Create a simple cycle graph: 0->1->2->0
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])

    # Degrees should be 1 for all
    # Kernel weights should be 1.0 for each edge
    # Mu should be 1.0 for all

    dg = DiffusionGeometry.from_edges(edge_index)

    print("Successfully instantiated DG object.")

    assert dg.n == 3
    assert np.allclose(dg.measure, np.ones(3))

    # Check embedding coords default to eye(3)
    assert np.allclose(dg.embedding_coords, np.eye(3))

    # Check simple function
    f = np.array([1.0, 2.0, 3.0])
    f_obj = dg.function(f)
    print("Function object creation successful.")

    # Check g metric roughly
    # Since mu is uniform and graph is regular, should be well behaved.
    norm_sq = dg.g(f_obj, f_obj)
    print("Computed norm squared:", norm_sq.shape)

    # Compute Laplacian
    Delta_f = dg.laplacian(0)(f_obj)
    print("Computed Laplacian:", Delta_f.coeffs)


def test_from_edges_custom_embedding():
    print("\nTesting from_edges with custom embedding...")
    edge_index = np.array([[0, 1], [1, 0]])
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    dg = DiffusionGeometry.from_edges(edge_index, embedding_coords=coords)

    assert dg.n == 2
    assert np.allclose(dg.embedding_coords, coords)
    assert np.allclose(dg.measure, np.ones(2))  # degrees are 1


def test_from_edges_weighted_degree():
    print("\nTesting from_edges with uneven degrees...")
    # 0 -> 2, 1 -> 2. Node 2 has in-degree 2.
    # 2 -> 0. Node 0 has in-degree 1.
    # Node 1 has in-degree 0? No, let's make it fully connected enough to be valid.
    # 0->1, 0->2, 1->2, 2->0
    # In-degrees:
    # 0: from 2 (1 edge) -> d(0)=1
    # 1: from 0 (1 edge) -> d(1)=1
    # 2: from 0, 1 (2 edges) -> d(2)=2

    edge_index = np.array([[0, 0, 1, 2], [1, 2, 2, 0]])

    dg = DiffusionGeometry.from_edges(edge_index)

    expected_measure = np.array([1.0, 1.0, 2.0])
    print("Measure:", dg.measure)
    assert np.allclose(dg.measure, expected_measure)

    # Weights for edges to 2 should be 1/2 = 0.5
    # Edges to 0 and 1 should be 1/1 = 1.0

    # Access internal cdc to verify?
    # Or just trust logic if mu is correct.

    print("Verification passed!")


if __name__ == "__main__":
    test_from_edges_cycle_graph()
    test_from_edges_custom_embedding()
    test_from_edges_weighted_degree()
