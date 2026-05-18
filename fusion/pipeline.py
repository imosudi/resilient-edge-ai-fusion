"""
fusion/pipeline.py

Vision-LiDAR fusion pipeline.

Improvements over original:
- estimate_lidar_confidence now uses valid-point ratio against the *expected*
  step count (not a fixed 360) so the metric actually varies with scene content
- estimate_camera_confidence uses contrast (std-dev) in addition to brightness
  so it responds to blur and overexposure, not just darkness
- Point-drop fallback: if lidar_points falls below a configurable floor the
  sample is logged as degraded and the fusion weights shift accordingly
- IncrementalSynchroniser replaces the full-history synchronise() call in the
  loop, keeping memory bounded and matching O(log n) per frame
- Sync delta is pulled from the synchroniser metadata rather than recomputed
- adaptive_fusion now returns the full result dict; weights are logged
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from fusion.confidence_fusion import adaptive_fusion
from metrics import build_metric_record
from fusion.hardware_config import (
    CAMERA_BACKEND_CHOICES,
    DEFAULT_CAMERA_BACKEND,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_HOKUYO_CLUSTER_COUNT,
    DEFAULT_HOKUYO_END_STEP,
    DEFAULT_HOKUYO_START_STEP,
    DEFAULT_LIDAR_BAUDRATE,
    DEFAULT_LIDAR_END_ANGLE_DEG,
    DEFAULT_LIDAR_MAX_DISTANCE_MM,
    DEFAULT_LIDAR_MIN_DISTANCE_MM,
    DEFAULT_LIDAR_PORT,
    DEFAULT_LIDAR_PROTOCOL,
    DEFAULT_LIDAR_START_ANGLE_DEG,
    DEFAULT_LIDAR_TIMEOUT,
    DEFAULT_MAX_FUSION_SAMPLES,
    DEFAULT_STREAM_DURATION,
    LIDAR_PROTOCOL_CHOICES,
)
from fusion.inference import INFERENCE_TARGET_CHOICES, get_inference_profile
from fusion.lidar import LidarCapture
from fusion.synchronise import IncrementalSynchroniser, synchronise
from fusion.vision import CameraCapture


LOGGER = logging.getLogger("fusion.pipeline")
logging.basicConfig(level=logging.INFO)

# Minimum expected LiDAR points per scan for the configured step window.
# At steps 256→512 the sensor should return ~256 points.
# Below this floor the scan is treated as partially degraded.
DEFAULT_MIN_LIDAR_POINTS = 180

# Maximum sync delta before a paired sample is considered stale.
DEFAULT_MAX_SYNC_DELTA_MS = 500.0


def estimate_camera_confidence(frame: dict) -> float:
    """
    Estimate camera signal quality from a single frame.

    Combines:
    - Mean brightness (low → dark, high → overexposed)
    - Normalised standard deviation (low → blur / flat scene / overexposure)

    Both components are penalised when they move away from a healthy midrange,
    giving a score in [0, 1] that responds to darkness, overexposure, and blur.
    """
    if frame is None or "image" not in frame:
        return 0.0

    image = frame["image"]
    if image is None:
        return 0.0

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.astype(np.float32)
    )

    mean_brightness = float(gray.mean()) / 255.0
    std_norm = float(gray.std()) / 128.0          # 128 ≈ typical healthy std

    # Brightness score: penalise both darkness (<0.15) and overexposure (>0.90)
    # Peak at ~0.45 (typical indoor scene).
    brightness_score = 1.0 - abs(mean_brightness - 0.45) / 0.55

    # Contrast score: low std means blur or flat/overexposed scene.
    contrast_score = float(min(1.0, std_norm))

    # Weight contrast more heavily — blur kills detection even at good brightness.
    confidence = 0.35 * brightness_score + 0.65 * contrast_score
    return float(max(0.0, min(1.0, confidence)))


def estimate_lidar_confidence(
    scan: dict,
    expected_points: int = DEFAULT_MIN_LIDAR_POINTS,
) -> float:
    """
    Estimate LiDAR signal quality from a scan dict.

    Uses the ratio of valid points returned to the expected count for the
    current step window, so the score actually falls when the sensor returns
    fewer points (e.g. due to obstruction or USB brownout).

    A scan at full capacity scores 1.0; a scan with half the expected points
    scores ~0.5.
    """
    if scan is None or "ranges" not in scan:
        return 0.0

    ranges = scan.get("ranges") or scan.get("points") or []
    if not ranges:
        return 0.0

    point_count = len(ranges)
    score = min(1.0, point_count / max(1, expected_points))
    return float(score)


class FusionPipeline:

    def __init__(
        self,
        camera_source: str = "camera",
        image_folder: str | None = None,
        camera_backend: str = DEFAULT_CAMERA_BACKEND,
        camera_width: int = DEFAULT_CAMERA_WIDTH,
        camera_height: int = DEFAULT_CAMERA_HEIGHT,
        lidar_port: str = DEFAULT_LIDAR_PORT,
        lidar_baudrate: int = DEFAULT_LIDAR_BAUDRATE,
        lidar_timeout: float = DEFAULT_LIDAR_TIMEOUT,
        lidar_log: str | None = None,
        lidar_protocol: str = DEFAULT_LIDAR_PROTOCOL,
        hokuyo_start_step: int = DEFAULT_HOKUYO_START_STEP,
        hokuyo_end_step: int = DEFAULT_HOKUYO_END_STEP,
        hokuyo_cluster_count: int = DEFAULT_HOKUYO_CLUSTER_COUNT,
        lidar_start_angle: float = DEFAULT_LIDAR_START_ANGLE_DEG,
        lidar_end_angle: float = DEFAULT_LIDAR_END_ANGLE_DEG,
        lidar_min_distance: float = DEFAULT_LIDAR_MIN_DISTANCE_MM,
        lidar_max_distance: float = DEFAULT_LIDAR_MAX_DISTANCE_MM,
        min_lidar_points: int = DEFAULT_MIN_LIDAR_POINTS,
        max_sync_delta_ms: float = DEFAULT_MAX_SYNC_DELTA_MS,
        inference_target: str = "cpu",
        run_id: str | None = None,
        degradation: str = "clean",
        severity: float = 0.0,
    ) -> None:
        self.camera_source = camera_source
        self.image_folder = image_folder
        self.camera_backend = camera_backend
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.lidar_port = lidar_port
        self.lidar_baudrate = lidar_baudrate
        self.lidar_timeout = lidar_timeout
        self.lidar_log = lidar_log
        self.lidar_protocol = lidar_protocol
        self.hokuyo_start_step = hokuyo_start_step
        self.hokuyo_end_step = hokuyo_end_step
        self.hokuyo_cluster_count = hokuyo_cluster_count
        self.lidar_start_angle = lidar_start_angle
        self.lidar_end_angle = lidar_end_angle
        self.lidar_min_distance = lidar_min_distance
        self.lidar_max_distance = lidar_max_distance
        self.min_lidar_points = min_lidar_points
        self.max_sync_delta_ms = max_sync_delta_ms
        self.inference_profile = get_inference_profile(inference_target)
        self.run_id = run_id or f"run_{int(time.time())}"
        self.degradation = degradation
        self.severity = severity
        self.camera = None
        self.lidar = None

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def _open_camera(self) -> None:
        self.camera = CameraCapture(
            source=self.camera_source,
            image_folder=self.image_folder,
            width=self.camera_width,
            height=self.camera_height,
            camera_backend=self.camera_backend,
        )

    def _open_lidar(self) -> None:
        self.lidar = LidarCapture(
            port=self.lidar_port,
            baudrate=self.lidar_baudrate,
            timeout=self.lidar_timeout,
            offline_log=self.lidar_log,
            protocol=self.lidar_protocol,
            hokuyo_start_step=self.hokuyo_start_step,
            hokuyo_end_step=self.hokuyo_end_step,
            hokuyo_cluster_count=self.hokuyo_cluster_count,
            start_angle_deg=self.lidar_start_angle,
            end_angle_deg=self.lidar_end_angle,
            min_distance_mm=self.lidar_min_distance,
            max_distance_mm=self.lidar_max_distance,
        )

    def _close_resources(self) -> None:
        if self.camera is not None:
            self.camera.release()
        if self.lidar is not None:
            self.lidar.close()

    # ------------------------------------------------------------------
    # Fusion logic
    # ------------------------------------------------------------------

    def fuse_sample(
        self,
        frame: dict,
        scan: dict,
        delta_ms: float = 0.0,
        stale: bool = False,
    ) -> dict:
        """
        Produce a single fused result from one camera frame and one LiDAR scan.

        Fallback behaviour
        ------------------
        * Stale pairing (delta > max_sync_delta_ms): camera weight boosted,
          LiDAR confidence penalised.
        * LiDAR point count below floor: lidar_conf is scaled down
          proportionally, triggering adaptive weight shift automatically.
        """
        lidar_point_count = len(
            scan.get("points") or scan.get("ranges") or []
        )

        # Apply point-drop penalty before confidence estimation.
        point_ratio = min(1.0, lidar_point_count / max(1, self.min_lidar_points))
        if point_ratio < 0.85:
            LOGGER.warning(
                "fuse_sample: LiDAR point count %d is %.0f%% of expected %d "
                "— scan quality degraded",
                lidar_point_count,
                point_ratio * 100,
                self.min_lidar_points,
            )

        camera_conf = estimate_camera_confidence(frame)
        # Scale raw lidar confidence by point-drop ratio.
        raw_lidar_conf = estimate_lidar_confidence(
            scan, expected_points=self.min_lidar_points
        )
        lidar_conf = raw_lidar_conf * point_ratio

        # Penalise LiDAR further if the sync pairing is stale.
        if stale:
            lidar_conf *= 0.5
            LOGGER.warning(
                "fuse_sample: stale sync pairing (delta=%.1f ms) — "
                "LiDAR confidence halved",
                delta_ms,
            )

        fusion_result = adaptive_fusion(camera_conf, lidar_conf)

        camera_health = "normal" if fusion_result["camera_healthy"] else "degraded"
        lidar_health = "normal" if fusion_result["lidar_healthy"] else "degraded"
        if stale:
            fallback_state = "stale_sync_penalty"
        elif fusion_result["dual_degraded"]:
            fallback_state = "dual_degraded_fallback"
        elif not fusion_result["camera_healthy"]:
            fallback_state = "lidar_assisted"
        elif not fusion_result["lidar_healthy"]:
            fallback_state = "camera_assisted"
        else:
            fallback_state = "normal"

        return build_metric_record(
            run_id=self.run_id,
            timestamp=max(frame["timestamp"], scan["timestamp"], time.time()),
            degradation=self.degradation,
            severity=self.severity,
            inference_target=self.inference_profile.target,
            camera_health=camera_health,
            lidar_health=lidar_health,
            stale_sync=stale,
            latency_ms=None,
            fps=None,
            cpu_percent=None,
            memory_mb=None,
            temperature_c=None,
            camera_confidence=camera_conf,
            lidar_confidence=lidar_conf,
            fusion_confidence=fusion_result["fused_confidence"],
            fallback_state=fallback_state,
            robustness_score=None,
            inference_label=self.inference_profile.label,
            model_artifact=self.inference_profile.model_artifact,
            precision=self.inference_profile.precision,
            runtime=self.inference_profile.runtime,
            accelerator=self.inference_profile.accelerator,
            camera_weight=fusion_result["camera_weight"],
            lidar_weight=fusion_result["lidar_weight"],
            dual_degraded=fusion_result["dual_degraded"],
            frame_source=frame.get("source"),
            lidar_source=scan.get("source"),
            camera_backend=frame.get("camera_backend"),
            lidar_protocol=scan.get("protocol"),
            lidar_points=lidar_point_count,
            timestamp_delta_ms=delta_ms,
        )

    # ------------------------------------------------------------------
    # Offline mode
    # ------------------------------------------------------------------

    def run_offline(
        self, max_samples: int = DEFAULT_MAX_FUSION_SAMPLES
    ) -> list[dict]:
        self.camera_source = "folder"
        self._open_camera()
        self._open_lidar()

        samples: list[dict] = []
        camera_frames: list[dict] = []
        lidar_scans: list[dict] = []

        try:
            while len(samples) < max_samples:
                frame = self.camera.get_frame()
                scan = self.lidar.read_scan()

                if scan is None:
                    break

                camera_frames.append(frame)
                lidar_scans.append(scan)

                sync = synchronise(
                    camera_frames,
                    lidar_scans,
                    max_delta_ms=self.max_sync_delta_ms,
                )
                if not sync:
                    continue

                pair = sync[-1]
                fused = self.fuse_sample(
                    pair["frame"],
                    pair["lidar"],
                    delta_ms=pair["delta_ms"],
                    stale=pair["stale"],
                )
                samples.append(fused)
                LOGGER.info("Fused sample %d: %s", len(samples), fused)

            return samples

        finally:
            self._close_resources()

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def run_live(
        self,
        duration: float = DEFAULT_STREAM_DURATION,
        max_samples: int | None = DEFAULT_MAX_FUSION_SAMPLES,
        display: bool = False,
    ) -> list[dict]:
        self._open_camera()
        self._open_lidar()

        # Use IncrementalSynchroniser to avoid O(n²) full-history rebuild.
        synchroniser = IncrementalSynchroniser(
            max_delta_ms=self.max_sync_delta_ms,
            max_scan_buffer=100,
        )

        samples: list[dict] = []
        stop_time = time.time() + duration

        try:
            while time.time() < stop_time:
                if max_samples is not None and len(samples) >= max_samples:
                    break

                frame = self.camera.get_frame()
                scan = self.lidar.read_scan()
                if scan is None:
                    continue

                synchroniser.add_scan(scan)
                synchroniser.add_frame(frame)

                for pair in synchroniser.pop_pending():
                    fused = self.fuse_sample(
                        pair["frame"],
                        pair["lidar"],
                        delta_ms=pair["delta_ms"],
                        stale=pair["stale"],
                    )
                    samples.append(fused)
                    LOGGER.info(
                        "Live fused sample %d: %s", len(samples), fused
                    )

                    if display:
                        if not self._show_live_display(frame, scan, fused):
                            return samples

                time.sleep(0.01)

            return samples

        finally:
            if display:
                cv2.destroyAllWindows()
            self._close_resources()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _show_live_display(
        self, frame: dict, scan: dict, fused: dict
    ) -> bool:
        image = frame["image"].copy()
        lines = [
            f"Inference: {fused['precision']} via {fused['runtime']}",
            f"Fused:  {fused['fused_confidence']:.3f}",
            f"Camera: {fused['camera_confidence']:.3f}  "
            f"w={fused.get('camera_weight', 0):.2f}",
            f"LiDAR:  {fused['lidar_confidence']:.3f}  "
            f"w={fused.get('lidar_weight', 0):.2f}",
            f"Sync delta: {fused['timestamp_delta_ms']:.1f} ms"
            + (" [STALE]" if fused.get("stale_sync") else ""),
            f"LiDAR pts: {fused['lidar_points']}"
            + (" [DEGRADED]" if fused.get("dual_degraded") else ""),
        ]

        for index, line in enumerate(lines):
            y = 28 + index * 26
            color = (
                (0, 80, 255) if "STALE" in line or "DEGRADED" in line
                else (0, 220, 60)
            )
            cv2.putText(
                image, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA,
            )

        cv2.imshow("Live Vision-LiDAR Fusion", image)
        return cv2.waitKey(1) != ord("q")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Vision-LiDAR fusion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument("--lidar-log", default=None)
    parser.add_argument("--lidar-port", default=DEFAULT_LIDAR_PORT)
    parser.add_argument(
        "--lidar-protocol",
        choices=LIDAR_PROTOCOL_CHOICES,
        default=DEFAULT_LIDAR_PROTOCOL,
    )
    parser.add_argument(
        "--camera-backend",
        choices=CAMERA_BACKEND_CHOICES,
        default=DEFAULT_CAMERA_BACKEND,
    )
    parser.add_argument(
        "--hokuyo-start-step", type=int, default=DEFAULT_HOKUYO_START_STEP
    )
    parser.add_argument(
        "--hokuyo-end-step", type=int, default=DEFAULT_HOKUYO_END_STEP
    )
    parser.add_argument(
        "--hokuyo-cluster-count", type=int, default=DEFAULT_HOKUYO_CLUSTER_COUNT
    )
    parser.add_argument(
        "--lidar-start-angle", type=float, default=DEFAULT_LIDAR_START_ANGLE_DEG
    )
    parser.add_argument(
        "--lidar-end-angle", type=float, default=DEFAULT_LIDAR_END_ANGLE_DEG
    )
    parser.add_argument(
        "--lidar-max-distance", type=float, default=DEFAULT_LIDAR_MAX_DISTANCE_MM
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_STREAM_DURATION)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_FUSION_SAMPLES)
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--inference-target",
        choices=INFERENCE_TARGET_CHOICES,
        default="cpu",
        help=(
            "Inference deployment profile used for result metadata: "
            "cpu = ONNX Runtime FP32 baseline, "
            "gpu = CUDA FP32 comparison, "
            "npu = Hailo Runtime INT8 HEF accelerated path."
        ),
    )
    args = parser.parse_args()

    pipeline = FusionPipeline(
        camera_source="folder" if args.mode == "offline" else "camera",
        image_folder=args.image_folder,
        camera_backend=args.camera_backend,
        lidar_port=args.lidar_port,
        lidar_log=args.lidar_log,
        lidar_protocol=args.lidar_protocol,
        hokuyo_start_step=args.hokuyo_start_step,
        hokuyo_end_step=args.hokuyo_end_step,
        hokuyo_cluster_count=args.hokuyo_cluster_count,
        lidar_start_angle=args.lidar_start_angle,
        lidar_end_angle=args.lidar_end_angle,
        lidar_max_distance=args.lidar_max_distance,
        inference_target=args.inference_target,
    )

    if args.mode == "offline":
        results = pipeline.run_offline(max_samples=args.max_samples)
    else:
        results = pipeline.run_live(
            duration=args.duration,
            max_samples=args.max_samples,
            display=args.display,
        )

    LOGGER.info("Fusion completed with %d results", len(results))
    for i, result in enumerate(results, 1):
        LOGGER.info("Result %d: %s", i, result)

    print(f"Completed {len(results)} fused samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Example usage:
python fusion/pipeline.py --mode live --display
    → Runs the live fusion pipeline with on-screen display. Press 'q' to quit.
python fusion/pipeline.py --mode offline --image-folder dataset/images --lidar-log dataset/lidar.log
    → Runs the offline fusion pipeline on the provided dataset.
    Results are logged to the console."""
