import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.confidence_fusion import adaptive_fusion


def test_normal_conditions():

    result = adaptive_fusion(
        camera_conf=0.9,
        lidar_conf=0.8
    )

    expected = (0.9 * 0.7) + (0.8 * 0.3)

    assert abs(result - expected) < 1e-6


def test_camera_degradation():

    result = adaptive_fusion(
        camera_conf=0.2,
        lidar_conf=0.9
    )

    expected = (0.2 * 0.3) + (0.9 * 0.7)

    assert abs(result - expected) < 1e-6