from scripts.evaluate_resilience import (
    HARDWARE_BENCHMARK_MATRIX,
    METRIC_FIELDS,
)


def test_unified_metric_schema_contains_required_fields():
    required_fields = {
        "run_id",
        "timestamp",
        "degradation",
        "severity",
        "inference_target",
        "camera_health",
        "lidar_health",
        "stale_sync",
        "latency_ms",
        "fps",
        "cpu_percent",
        "memory_mb",
        "temperature_c",
        "camera_confidence",
        "lidar_confidence",
        "fusion_confidence",
        "fallback_state",
        "robustness_score",
        "object_count",
        "detected_objects",
        "detection_confidences",
        "object_distances_mm",
        "nearest_object_distance_mm",
        "annotation_path",
    }

    assert required_fields.issubset(set(METRIC_FIELDS))


def test_hardware_benchmark_matrix_is_separate_from_metric_schema():
    matrix = {
        item["inference_path"]: item["runtime"]
        for item in HARDWARE_BENCHMARK_MATRIX
    }

    assert matrix == {
        "CPU FP32": "ONNX Runtime",
        "GPU FP32": "CUDA",
        "NPU INT8": "Hailo Runtime",
    }
