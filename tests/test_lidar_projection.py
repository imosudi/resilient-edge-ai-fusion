import math

import numpy as np

from preprocessing.lidar_projection import (
    polar_to_cartesian,
    project_scan,
    scan_angles,
    valid_range_mask,
)


def test_polar_to_cartesian_scalar():
    x, y = polar_to_cartesian(90, 1000)

    assert abs(x) < 1e-6
    assert math.isclose(y, 1000.0)


def test_scan_angles_default_span():
    angles = scan_angles(3)

    assert np.allclose(angles, [-120.0, 0.0, 120.0])


def test_project_scan_filters_invalid_ranges():
    points = project_scan([0, 1000, float("nan"), 5000, 2000])

    assert points.shape == (2, 2)


def test_valid_range_mask():
    mask = valid_range_mask([10, 20, 1000, 5000, float("inf")])

    assert mask.tolist() == [False, True, True, False, False]
