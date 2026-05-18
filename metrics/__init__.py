"""metrics/__init__.py

Standardised metric record utilities for the resilient edge AI framework.
"""

import time
from typing import Any, Dict, List

STANDARD_METRIC_FIELDS: List[str] = [
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
    "recovery_state",
    "robustness_score",
]


def _normalise_health(value: Any) -> str:
    if isinstance(value, bool):
        return "normal" if value else "degraded"

    if value is None:
        return "unknown"

    normalized = str(value).strip().lower()
    if normalized in {"healthy", "normal", "ok", "good"}:
        return "normal"
    if normalized in {"degraded", "unhealthy", "bad"}:
        return "degraded"
    return str(value)


def build_metric_record(**kwargs: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {field: kwargs.get(field) for field in STANDARD_METRIC_FIELDS}
    record["timestamp"] = float(kwargs.get("timestamp", time.time()))
    record["severity"] = float(kwargs.get("severity", 0.0))
    record["stale_sync"] = bool(kwargs.get("stale_sync", False))
    record["camera_health"] = _normalise_health(
        kwargs.get("camera_health", kwargs.get("camera_healthy"))
    )
    record["lidar_health"] = _normalise_health(
        kwargs.get("lidar_health", kwargs.get("lidar_healthy"))
    )

    fusion_confidence = kwargs.get("fusion_confidence", kwargs.get("fused_confidence"))
    if fusion_confidence is not None:
        record["fusion_confidence"] = float(fusion_confidence)
    if kwargs.get("fused_confidence") is not None:
        record["fused_confidence"] = float(kwargs["fused_confidence"])
    elif record["fusion_confidence"] is not None:
        record["fused_confidence"] = record["fusion_confidence"]

    for numeric in [
        "latency_ms",
        "fps",
        "cpu_percent",
        "memory_mb",
        "temperature_c",
        "camera_confidence",
        "lidar_confidence",
        "fusion_confidence",
        "robustness_score",
    ]:
        value = record.get(numeric)
        if value is not None:
            record[numeric] = float(value)

    record.update(kwargs)
    return record
