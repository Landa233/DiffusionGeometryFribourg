import numpy as np

from diffusion_geometry import DiffusionGeometry
from methods.circular_coordinates import circular_coordinates
from TDA.image_patches.patch_data import (
    build_knn_diffusion_kernel,
    cut_kernel_by_angle,
    maxmin_landmarks,
    normalise_rows,
)
from TDA.synthetic.run_circular_coordinates import klein_bottle


def angle_alignment_score(recovered: np.ndarray, truth: np.ndarray) -> float:
    forward = abs(np.mean(np.exp(1j * (recovered - truth))))
    backward = abs(np.mean(np.exp(1j * (-recovered - truth))))
    return float(max(forward, backward))


def test_normalise_rows_drops_zero_rows():
    data = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])

    result = normalise_rows(data)

    assert result.shape == (2, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_maxmin_landmarks_is_deterministic():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 3))

    first = maxmin_landmarks(data, 8, seed=10)
    second = maxmin_landmarks(data, 8, seed=10)

    assert first.shape == (8, 3)
    assert np.allclose(first, second)


def test_cut_kernel_by_angle_removes_seam_edges_and_renormalises():
    theta = np.array([0.05, 0.15, 2.0 * np.pi - 0.15, 2.0 * np.pi - 0.05])
    nbr_indices = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
            [3, 2, 1],
        ]
    )
    kernel = np.full_like(nbr_indices, 1.0 / 3.0, dtype=float)

    cut_kernel, diagnostics = cut_kernel_by_angle(nbr_indices, kernel, theta)

    assert np.allclose(cut_kernel.sum(axis=1), 1.0)
    assert diagnostics.edges_after < diagnostics.edges_before
    assert diagnostics.fraction_removed > 0.0
    assert cut_kernel[0, 2] == 0.0
    assert cut_kernel[2, 0] > 0.0


def test_synthetic_klein_cut_coordinate_smoke():
    rng = np.random.default_rng(4)
    data, truth = klein_bottle(180, ambient_dim=4, rng=rng)
    dg = DiffusionGeometry.from_point_cloud(
        data,
        n_function_basis=24,
        n_coefficients=12,
        knn_kernel=24,
        knn_bandwidth=8,
    )

    base_result = circular_coordinates(dg, epsilon=1.0, k=8)
    base_candidate = max(
        base_result.candidates,
        key=lambda candidate: angle_alignment_score(candidate.angle, truth["base"]),
    )
    theta_base = base_candidate.angle
    alignment = angle_alignment_score(theta_base, truth["base"])

    nbr_indices, kernel, bandwidths = build_knn_diffusion_kernel(
        data, knn_kernel=24, knn_bandwidth=8
    )
    cut_kernel, diagnostics = cut_kernel_by_angle(nbr_indices, kernel, theta_base)
    cut_dg = DiffusionGeometry.from_knn_kernel(
        nbr_indices=nbr_indices,
        kernel=cut_kernel,
        bandwidths=bandwidths,
        immersion_coords=data,
        data_matrix=data,
        n_function_basis=24,
        n_coefficients=12,
    )
    fibre_result = circular_coordinates(cut_dg, epsilon=1.0, k=8)

    assert alignment > 0.2
    assert diagnostics.edges_after < diagnostics.edges_before
    assert diagnostics.components_after <= 3
    assert fibre_result.angle.shape == (data.shape[0],)
    assert np.all(np.isfinite(fibre_result.angle))
