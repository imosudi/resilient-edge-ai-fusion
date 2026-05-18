import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from fusion.inference import INFERENCE_TARGET_CHOICES

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("scripts.run_fusion")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the resilient Vision-LiDAR fusion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["live", "offline"],
        default="offline",
        help="Pipeline mode: live sensor capture or offline playback.",
    )
    parser.add_argument(
        "--image-folder",
        default="dataset/images",
        help="Image folder for offline mode.",
    )
    parser.add_argument(
        "--lidar-log",
        default=None,
        help="LiDAR log file for offline mode.",
    )
    parser.add_argument(
        "--lidar-port",
        default=DEFAULT_LIDAR_PORT,
        help="Serial port for live LiDAR capture.",
    )
    parser.add_argument(
        "--lidar-baudrate",
        type=int,
        default=DEFAULT_LIDAR_BAUDRATE,
        help="Serial baudrate for live LiDAR capture.",
    )
    parser.add_argument(
        "--lidar-timeout",
        type=float,
        default=DEFAULT_LIDAR_TIMEOUT,
        help="Serial timeout in seconds for live LiDAR capture.",
    )
    parser.add_argument(
        "--lidar-protocol",
        choices=LIDAR_PROTOCOL_CHOICES,
        default=DEFAULT_LIDAR_PROTOCOL,
        help="LiDAR input protocol. Use hokuyo for Hokuyo URG SCIP polling.",
    )
    parser.add_argument(
        "--hokuyo-start-step",
        type=int,
        default=DEFAULT_HOKUYO_START_STEP,
        help="Hokuyo SCIP start step for live scans.",
    )
    parser.add_argument(
        "--hokuyo-end-step",
        type=int,
        default=DEFAULT_HOKUYO_END_STEP,
        help="Hokuyo SCIP end step for live scans.",
    )
    parser.add_argument(
        "--hokuyo-cluster-count",
        type=int,
        default=DEFAULT_HOKUYO_CLUSTER_COUNT,
        help="Hokuyo SCIP cluster count for live scans.",
    )
    parser.add_argument(
        "--lidar-start-angle",
        type=float,
        default=DEFAULT_LIDAR_START_ANGLE_DEG,
        help="Start angle in degrees for projecting LiDAR samples.",
    )
    parser.add_argument(
        "--lidar-end-angle",
        type=float,
        default=DEFAULT_LIDAR_END_ANGLE_DEG,
        help="End angle in degrees for projecting LiDAR samples.",
    )
    parser.add_argument(
        "--lidar-min-distance",
        type=float,
        default=DEFAULT_LIDAR_MIN_DISTANCE_MM,
        help="Minimum valid LiDAR distance in millimetres.",
    )
    parser.add_argument(
        "--lidar-max-distance",
        type=float,
        default=DEFAULT_LIDAR_MAX_DISTANCE_MM,
        help="Maximum valid LiDAR distance in millimetres.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=CAMERA_BACKEND_CHOICES,
        default=DEFAULT_CAMERA_BACKEND,
        help="Camera backend for live mode.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Live camera frame width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Live camera frame height.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_STREAM_DURATION,
        help="Live capture duration in seconds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_FUSION_SAMPLES,
        help="Maximum number of fused samples.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live camera overlay with fusion confidence.",
    )
    parser.add_argument(
        "--inference-target",
        choices=INFERENCE_TARGET_CHOICES,
        default="cpu",
        help=(
            "Inference deployment profile to record: "
            "cpu for ONNX Runtime FP32, npu for Hailo Runtime INT8 HEF."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from fusion.pipeline import FusionPipeline

    if args.mode == "offline" and not args.lidar_log:
        raise SystemExit(
            "Offline mode requires --lidar-log to be set to a valid LiDAR logfile."
        )

    pipeline = FusionPipeline(
        camera_source="folder" if args.mode == "offline" else "camera",
        image_folder=args.image_folder,
        camera_backend=args.camera_backend,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        lidar_port=args.lidar_port,
        lidar_baudrate=args.lidar_baudrate,
        lidar_timeout=args.lidar_timeout,
        lidar_log=args.lidar_log,
        lidar_protocol=args.lidar_protocol,
        hokuyo_start_step=args.hokuyo_start_step,
        hokuyo_end_step=args.hokuyo_end_step,
        hokuyo_cluster_count=args.hokuyo_cluster_count,
        lidar_start_angle=args.lidar_start_angle,
        lidar_end_angle=args.lidar_end_angle,
        lidar_min_distance=args.lidar_min_distance,
        lidar_max_distance=args.lidar_max_distance,
        inference_target=args.inference_target,
    )

    if args.mode == "offline":
        LOGGER.info(
            "Starting offline fusion with images=%s lidar_log=%s inference_target=%s",
            args.image_folder,
            args.lidar_log,
            args.inference_target,
        )
        results = pipeline.run_offline(max_samples=args.max_samples)
    else:
        LOGGER.info(
            "Starting live fusion with camera_backend=%s lidar_port=%s "
            "lidar_protocol=%s inference_target=%s",
            args.camera_backend,
            args.lidar_port,
            args.lidar_protocol,
            args.inference_target,
        )
        results = pipeline.run_live(
            duration=args.duration,
            max_samples=args.max_samples,
            display=args.display,
        )

    LOGGER.info("Fusion completed with %d results", len(results))
    for index, item in enumerate(results, start=1):
        LOGGER.info("Result %d: %s", index, item)

    print(f"Completed {len(results)} fused samples")


if __name__ == "__main__":
    main()
