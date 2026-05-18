import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.confidence_fusion import adaptive_fusion


def test_normal_conditions():

    result = adaptive_fusion(
        camera_conf=0.9,
        lidar_conf=0.8
    )

    assert "fused_confidence" in result
    assert "camera_weight" in result
    assert "lidar_weight" in result
    assert result["camera_healthy"] is True
    assert result["lidar_healthy"] is True
    assert result["dual_degraded"] is False
    assert 0.0 <= result["fused_confidence"] <= 1.0
    assert abs(result["camera_weight"] + result["lidar_weight"] - 1.0) < 1e-6


def test_camera_degradation():

    result = adaptive_fusion(
        camera_conf=0.2,
        lidar_conf=0.9
    )

    assert result["camera_healthy"] is False
    assert result["lidar_healthy"] is True
    assert result["dual_degraded"] is False
    assert result["lidar_weight"] > 0.5
    assert 0.0 <= result["fused_confidence"] <= 1.0
