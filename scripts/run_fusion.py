import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.pipeline import FusionPipeline

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("scripts.run_fusion")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the resilient Vision-LiDAR fusion pipeline."
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
        default="/dev/ttyACM0",
        help="Serial port for live LiDAR capture.",
    )
    parser.add_argument(
        "--lidar-baudrate",
        type=int,
        default=115200,
        help="Serial baudrate for live LiDAR capture.",
    )
    parser.add_argument(
        "--lidar-timeout",
        type=float,
        default=1,
        help="Serial timeout in seconds for live LiDAR capture.",
    )
    parser.add_argument(
        "--lidar-protocol",
        choices=["raw", "hokuyo"],
        default="raw",
        help="LiDAR input protocol. Use hokuyo for Hokuyo URG SCIP polling.",
    )
    parser.add_argument(
        "--hokuyo-start-step",
        type=int,
        default=44,
        help="Hokuyo SCIP start step for live scans.",
    )
    parser.add_argument(
        "--hokuyo-end-step",
        type=int,
        default=725,
        help="Hokuyo SCIP end step for live scans.",
    )
    parser.add_argument(
        "--hokuyo-cluster-count",
        type=int,
        default=0,
        help="Hokuyo SCIP cluster count for live scans.",
    )
    parser.add_argument(
        "--lidar-start-angle",
        type=float,
        default=-120.0,
        help="Start angle in degrees for projecting LiDAR samples.",
    )
    parser.add_argument(
        "--lidar-end-angle",
        type=float,
        default=120.0,
        help="End angle in degrees for projecting LiDAR samples.",
    )
    parser.add_argument(
        "--lidar-min-distance",
        type=float,
        default=20.0,
        help="Minimum valid LiDAR distance in millimetres.",
    )
    parser.add_argument(
        "--lidar-max-distance",
        type=float,
        default=4000.0,
        help="Maximum valid LiDAR distance in millimetres.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "opencv", "picamera2", "libcamera"],
        default="auto",
        help="Camera backend for live mode.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Live camera frame width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Live camera frame height.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Live capture duration in seconds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of fused samples.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live camera overlay with fusion confidence.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
    )

    if args.mode == "offline":
        LOGGER.info("Starting offline fusion with images=%s lidar_log=%s",
                    args.image_folder, args.lidar_log)
        results = pipeline.run_offline(max_samples=args.max_samples)
    else:
        LOGGER.info(
            "Starting live fusion with camera_backend=%s lidar_port=%s lidar_protocol=%s",
            args.camera_backend,
            args.lidar_port,
            args.lidar_protocol,
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
