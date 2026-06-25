"""Physical-location scores for decoded Gardner circular coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class CoordinatePhysicalScore:
    coordinate_index: int
    physical_smoothness: float
    circular_variance: float
    local_energy: float
    decoded_angle: np.ndarray


def circular_variance(angle: np.ndarray) -> float:
    return float(1.0 - abs(np.mean(np.exp(1j * angle))))


def wrapped_angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (first - second)))


def physical_smoothness_score(
    physical_xy: np.ndarray,
    angle: np.ndarray,
    *,
    n_neighbors: int,
    stride: int,
) -> tuple[float, float, float]:
    """Score local angle variation over physical space.

    Lower ``physical_smoothness`` is better. The numerator is the weighted
    nearest-neighbour wrapped angle energy; dividing by circular variance avoids
    rewarding nearly constant decoded angles.
    """

    if stride <= 0:
        raise ValueError("stride must be positive.")
    if physical_xy.shape[0] != angle.shape[0]:
        raise ValueError("physical coordinates and angles must have the same length.")

    physical_xy = physical_xy[::stride]
    angle = angle[::stride]
    finite = np.isfinite(physical_xy).all(axis=1) & np.isfinite(angle)
    physical_xy = physical_xy[finite]
    angle = angle[finite]
    if physical_xy.shape[0] <= n_neighbors:
        raise ValueError("Need more decoded samples than physical neighbours.")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(physical_xy)
    distances, indices = nbrs.kneighbors(physical_xy)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    diffs = wrapped_angle_difference(angle[:, None], angle[indices])
    scale = np.median(distances[distances > 0])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    weights = np.exp(-((distances / scale) ** 2))
    local_energy = float(np.sum(weights * diffs**2) / np.sum(weights))
    variance = circular_variance(angle)
    return float(local_energy / max(variance, 1e-3)), variance, local_energy


def score_decoded_coordinates(
    coordsbox: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    times_box: np.ndarray,
    *,
    physical_neighbors: int,
    stride: int,
) -> list[CoordinatePhysicalScore]:
    physical_xy = np.column_stack((xx[times_box], yy[times_box]))
    scores = []
    for coordinate_index in range(coordsbox.shape[1]):
        decoded_angle = coordsbox[:, coordinate_index]
        smoothness, variance, local_energy = physical_smoothness_score(
            physical_xy,
            decoded_angle,
            n_neighbors=physical_neighbors,
            stride=stride,
        )
        scores.append(
            CoordinatePhysicalScore(
                coordinate_index=coordinate_index,
                physical_smoothness=smoothness,
                circular_variance=variance,
                local_energy=local_energy,
                decoded_angle=decoded_angle,
            )
        )
    return scores
