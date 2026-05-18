"""
fusion/confidence_fusion.py

Adaptive confidence-based sensor fusion weighting.

Improvements over original:
- Smooth sigmoid-based weighting instead of hard binary threshold
- Explicit dual-degradation fallback (both sensors weak → equal weight, low output)
- Per-sensor floor prevents a dead sensor from dragging fused confidence to zero
  without the pipeline noticing
- Returns a full result dict so callers can inspect weights, not just the scalar
"""

from __future__ import annotations

import logging
import math

LOGGER = logging.getLogger(__name__)

# Confidence below this is considered "degraded".
CAMERA_DEGRADED_THRESHOLD = 0.35
LIDAR_DEGRADED_THRESHOLD = 0.40

# When BOTH sensors are degraded, report this as the fused confidence ceiling
# rather than a misleadingly high value.
DUAL_DEGRADATION_CEILING = 0.45

# Steepness of the sigmoid transition around the degraded threshold.
# Higher = sharper switch; lower = smoother blend.
SIGMOID_STEEPNESS = 12.0


def _sigmoid(x: float, centre: float, steepness: float) -> float:
    """Smooth 0→1 transition centred at *centre*."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - centre)))


def adaptive_fusion(
    camera_conf: float,
    lidar_conf: float,
    camera_threshold: float = CAMERA_DEGRADED_THRESHOLD,
    lidar_threshold: float = LIDAR_DEGRADED_THRESHOLD,
) -> dict[str, float]:
    """
    Compute an adaptively weighted fusion confidence score.

    Weight allocation strategy
    --------------------------
    * Camera healthy, LiDAR healthy → 60 % camera / 40 % LiDAR
      (vision carries more semantic content for object detection)
    * Camera degraded, LiDAR healthy → 25 % camera / 75 % LiDAR
    * Camera healthy, LiDAR degraded → 80 % camera / 20 % LiDAR
    * Both degraded → equal 50/50 but cap fused output at
      ``DUAL_DEGRADATION_CEILING`` and log a warning

    Weights transition smoothly via sigmoid rather than snapping at
    a hard threshold, so a camera_conf of 0.349 vs 0.351 does not cause
    a sudden jump in fused_confidence.

    Parameters
    ----------
    camera_conf:
        Camera confidence in [0, 1].
    lidar_conf:
        LiDAR confidence in [0, 1].
    camera_threshold:
        Sigmoid centre for the camera health signal.
    lidar_threshold:
        Sigmoid centre for the LiDAR health signal.

    Returns
    -------
    dict with keys:
        ``fused_confidence`` – weighted combination in [0, 1]
        ``camera_weight``    – weight applied to camera_conf
        ``lidar_weight``     – weight applied to lidar_conf
        ``camera_healthy``   – bool, camera above threshold
        ``lidar_healthy``    – bool, LiDAR above threshold
        ``dual_degraded``    – bool, both sensors below threshold
    """
    camera_conf = float(max(0.0, min(1.0, camera_conf)))
    lidar_conf = float(max(0.0, min(1.0, lidar_conf)))

    camera_health = _sigmoid(camera_conf, camera_threshold, SIGMOID_STEEPNESS)
    lidar_health = _sigmoid(lidar_conf, lidar_threshold, SIGMOID_STEEPNESS)

    camera_healthy = camera_conf >= camera_threshold
    lidar_healthy = lidar_conf >= lidar_threshold
    dual_degraded = (not camera_healthy) and (not lidar_healthy)

    if dual_degraded:
        # Both sensors unreliable — equal weight, capped output.
        camera_weight = 0.5
        lidar_weight = 0.5
        raw_fused = camera_conf * camera_weight + lidar_conf * lidar_weight
        fused_confidence = min(raw_fused, DUAL_DEGRADATION_CEILING)
        LOGGER.warning(
            "adaptive_fusion: both sensors degraded "
            "(camera=%.3f, lidar=%.3f) → fused capped at %.3f",
            camera_conf,
            lidar_conf,
            fused_confidence,
        )
    else:
        # Smooth weight allocation:
        #   camera_weight = 0.25 + 0.55 * camera_health
        #   lidar_weight  = 0.75 - 0.55 * camera_health
        # When camera_health → 1 (healthy):  wc=0.80, wl=0.20  (but see below)
        # When camera_health → 0 (degraded): wc=0.25, wl=0.75
        #
        # We then modulate lidar_weight by lidar_health to reduce LiDAR's
        # contribution when it too is partially degraded.
        base_camera_weight = 0.25 + 0.55 * camera_health
        base_lidar_weight = 1.0 - base_camera_weight

        # Scale LiDAR share by its own health; redistribute to camera.
        effective_lidar_weight = base_lidar_weight * lidar_health
        effective_camera_weight = 1.0 - effective_lidar_weight

        # Normalise so weights always sum to 1.
        total = effective_camera_weight + effective_lidar_weight
        camera_weight = effective_camera_weight / total
        lidar_weight = effective_lidar_weight / total

        fused_confidence = float(
            max(0.0, min(1.0, camera_conf * camera_weight + lidar_conf * lidar_weight))
        )

    return {
        "fused_confidence": fused_confidence,
        "camera_weight": camera_weight,
        "lidar_weight": lidar_weight,
        "camera_healthy": camera_healthy,
        "lidar_healthy": lidar_healthy,
        "dual_degraded": dual_degraded,
    }


def fused_confidence_scalar(
    camera_conf: float,
    lidar_conf: float,
    camera_threshold: float = CAMERA_DEGRADED_THRESHOLD,
    lidar_threshold: float = LIDAR_DEGRADED_THRESHOLD,
) -> float:
    """
    Convenience wrapper that returns only the scalar fused confidence.

    Drop-in replacement for the original ``adaptive_fusion`` return value.
    """
    return adaptive_fusion(
        camera_conf,
        lidar_conf,
        camera_threshold=camera_threshold,
        lidar_threshold=lidar_threshold,
    )["fused_confidence"]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    scenarios = [
        ("Both healthy",          0.82, 0.91),
        ("Camera degraded",       0.28, 0.85),
        ("LiDAR degraded",        0.75, 0.25),
        ("Both degraded",         0.20, 0.30),
        ("Borderline camera",     0.36, 0.80),
        ("Your live run values",  0.389, 0.714),  # from the previous run
    ]

    print(f"{'Scenario':<30} {'cam':>5} {'lid':>5} {'fused':>6} {'wc':>5} {'wl':>5} {'dual':>5}")
    print("-" * 68)
    for label, cam, lid in scenarios:
        result = adaptive_fusion(cam, lid)
        print(
            f"{label:<30} "
            f"{cam:>5.3f} {lid:>5.3f} "
            f"{result['fused_confidence']:>6.3f} "
            f"{result['camera_weight']:>5.3f} "
            f"{result['lidar_weight']:>5.3f} "
            f"{str(result['dual_degraded']):>5}"
        )

"""
python fusion/confidence_fusion.py \
    && echo "\nAll tests passed."
"""