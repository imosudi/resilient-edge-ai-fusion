import argparse
import logging

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
        "--duration",
        type=int,
        default=10,
        help="Live capture duration in seconds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of fused samples for offline mode.",
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
        lidar_port=args.lidar_port,
        lidar_log=args.lidar_log,
    )

    if args.mode == "offline":
        LOGGER.info("Starting offline fusion with images=%s lidar_log=%s",
                    args.image_folder, args.lidar_log)
        results = pipeline.run_offline(max_samples=args.max_samples)
    else:
        LOGGER.info("Starting live fusion with lidar_port=%s", args.lidar_port)
        results = pipeline.run_live(duration=args.duration)

    LOGGER.info("Fusion completed with %d results", len(results))
    for index, item in enumerate(results, start=1):
        LOGGER.info("Result %d: %s", index, item)

    print(f"Completed {len(results)} fused samples")


if __name__ == "__main__":
    main()
