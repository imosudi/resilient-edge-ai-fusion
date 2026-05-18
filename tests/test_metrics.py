import importlib.util

import pytest

from metrics import build_metric_record
from metrics.baseline_metrics import measure_yolo_baseline


pytestmark = pytest.mark.skipif(
    any(
        importlib.util.find_spec(package) is None
        for package in ("pandas", "psutil", "ultralytics")
    ),
    reason="baseline inference metrics require pandas, psutil, and ultralytics",
)


def test_latency_positive(tmp_path):
    metrics = measure_yolo_baseline(csv_path=tmp_path / "results.csv")

    assert metrics["latency_ms"] > 0


def test_fps_positive(tmp_path):
    metrics = measure_yolo_baseline(csv_path=tmp_path / "results.csv")

    assert metrics["fps"] > 0


def test_reasonable_latency(tmp_path):
    metrics = measure_yolo_baseline(csv_path=tmp_path / "results.csv")

    assert metrics["latency_ms"] < 5000


def test_build_metric_record_has_standard_fields():
    record = build_metric_record(
        run_id="run_001",
        timestamp=1779125682.05,
        degradation="low_light",
        severity=0.6,
        inference_target="cpu",
        camera_health=False,
        lidar_health=True,
        stale_sync=False,
        latency_ms=87.3,
        fps=11.4,
        cpu_percent=73.1,
        memory_mb=812,
        temperature_c=64.2,
        camera_confidence=0.46,
        lidar_confidence=0.66,
        fused_confidence=0.52,
        fallback_state="lidar_assisted",
        robustness_score=81.4,
    )

    assert record["run_id"] == "run_001"
    assert record["degradation"] == "low_light"
    assert record["severity"] == 0.6
    assert record["fusion_confidence"] == 0.52
    assert record["fused_confidence"] == 0.52
    assert record["camera_health"] == "degraded"
    assert record["lidar_health"] == "normal"
    assert record["fallback_state"] == "lidar_assisted"
