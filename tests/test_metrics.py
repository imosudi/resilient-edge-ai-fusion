import importlib.util

import pytest

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
