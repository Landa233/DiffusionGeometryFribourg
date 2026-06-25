import numpy as np
import pytest

from TDA.gardner2022.compare_decodings import build_comparison_summary
from TDA.gardner2022.physical_coordinate_scores import (
    physical_smoothness_score,
    score_decoded_coordinates,
)


def test_physical_smoothness_prefers_location_smooth_angle():
    grid_x, grid_y = np.meshgrid(np.linspace(-1.0, 1.0, 20), np.linspace(-1.0, 1.0, 20))
    physical_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    smooth_angle = np.mod(np.arctan2(physical_xy[:, 1], physical_xy[:, 0]), 2.0 * np.pi)
    checker_angle = np.where((grid_x.ravel() > 0) ^ (grid_y.ravel() > 0), 0.0, np.pi)

    smooth_score, smooth_variance, smooth_energy = physical_smoothness_score(
        physical_xy,
        smooth_angle,
        n_neighbors=8,
        stride=1,
    )
    checker_score, _, checker_energy = physical_smoothness_score(
        physical_xy,
        checker_angle,
        n_neighbors=8,
        stride=1,
    )

    assert smooth_variance > 0.0
    assert smooth_energy < checker_energy
    assert smooth_score < checker_score


def test_score_decoded_coordinates_matches_times_box():
    xx = np.linspace(-1.0, 1.0, 60)
    yy = np.sin(xx)
    times_box = np.arange(5, 55)
    angles = np.column_stack(
        (
            np.mod(np.linspace(0.0, 2.0 * np.pi, len(times_box)), 2.0 * np.pi),
            np.mod(np.linspace(2.0 * np.pi, 0.0, len(times_box)), 2.0 * np.pi),
        )
    )

    scores = score_decoded_coordinates(
        angles,
        xx,
        yy,
        times_box,
        physical_neighbors=5,
        stride=1,
    )

    assert [score.coordinate_index for score in scores] == [0, 1]
    assert all(np.isfinite(score.physical_smoothness) for score in scores)


def test_physical_smoothness_rejects_nonpositive_stride():
    with pytest.raises(ValueError, match="stride"):
        physical_smoothness_score(
            np.zeros((10, 2)),
            np.zeros(10),
            n_neighbors=2,
            stride=0,
        )


def test_comparison_summary_accepts_hodge_when_no_worse_than_persistent():
    scores = {
        "persistent": {"mean_physical_smoothness": 0.5},
        "hodge": {"mean_physical_smoothness": 0.5},
    }

    summary = build_comparison_summary(
        scores,
        rat="R",
        module="1",
        session="OF",
        day="day2",
        physical_neighbors=12,
        stride=10,
    )

    assert summary["hodge_at_least_persistent"] is True
    assert summary["mean_physical_smoothness_delta_hodge_minus_persistent"] == 0.0


def test_comparison_summary_rejects_hodge_when_smoother_baseline_wins():
    scores = {
        "persistent": {"mean_physical_smoothness": 0.4},
        "hodge": {"mean_physical_smoothness": 0.6},
    }

    summary = build_comparison_summary(
        scores,
        rat="R",
        module="1",
        session="OF",
        day="day2",
        physical_neighbors=12,
        stride=10,
    )

    assert summary["ranking"] == ["persistent", "hodge"]
    assert summary["hodge_at_least_persistent"] is False


def test_comparison_summary_can_apply_small_baseline_tolerance():
    scores = {
        "persistent": {"mean_physical_smoothness": 0.4},
        "hodge": {"mean_physical_smoothness": 0.400001},
    }

    summary = build_comparison_summary(
        scores,
        rat="R",
        module="1",
        session="OF",
        day="day2",
        physical_neighbors=12,
        stride=10,
        baseline_tolerance=1e-5,
    )

    assert summary["hodge_at_least_persistent"] is True
