import argparse
import logging
import time

from fusion.confidence_fusion import adaptive_fusion
from fusion.lidar import LidarCapture
from fusion.synchronise import synchronise
from fusion.vision import CameraCapture


LOGGER = logging.getLogger("fusion.pipeline")
logging.basicConfig(level=logging.INFO)


def estimate_camera_confidence(frame):
    if frame is None or "image" not in frame:
        return 0.0

    image = frame["image"]
    if image is None:
        return 0.0

    mean_brightness = image.mean() / 255.0
    return float(max(0.0, min(1.0, mean_brightness)))


def estimate_lidar_confidence(scan):
    if scan is None or "ranges" not in scan:
        return 0.0

    ranges = scan["ranges"]
    if not ranges:
        return 0.0

    score = min(1.0, len(ranges) / 360.0)
    return float(score)


class FusionPipeline:

    def __init__(
        self,
        camera_source="camera",
        image_folder=None,
        lidar_port="/dev/ttyACM0",
        lidar_log=None,
    ):
        self.camera_source = camera_source
        self.image_folder = image_folder
        self.lidar_port = lidar_port
        self.lidar_log = lidar_log
        self.camera = None
        self.lidar = None

    def _open_camera(self):
        self.camera = CameraCapture(
            source=self.camera_source,
            image_folder=self.image_folder,
        )

    def _open_lidar(self):
        self.lidar = LidarCapture(
            port=self.lidar_port,
            offline_log=self.lidar_log,
        )

    def _close_resources(self):
        if self.camera is not None:
            self.camera.release()

        if self.lidar is not None:
            self.lidar.close()

    def fuse_sample(self, frame, scan):
        camera_conf = estimate_camera_confidence(frame)
        lidar_conf = estimate_lidar_confidence(scan)
        fused_conf = adaptive_fusion(
            camera_conf=camera_conf,
            lidar_conf=lidar_conf,
        )

        return {
            "timestamp": max(frame["timestamp"], scan["timestamp"]),
            "camera_confidence": camera_conf,
            "lidar_confidence": lidar_conf,
            "fused_confidence": fused_conf,
            "frame_source": frame.get("source"),
            "lidar_source": scan.get("source"),
        }

    def run_offline(self, max_samples=20):
        self.camera_source = "folder"
        self._open_camera()
        self._open_lidar()

        samples = []
        camera_frames = []
        lidar_scans = []

        try:
            while len(samples) < max_samples:
                frame = self.camera.get_frame()
                scan = self.lidar.read_scan()

                if scan is None:
                    break

                camera_frames.append(frame)
                lidar_scans.append(scan)

                sync = synchronise(camera_frames, lidar_scans)
                if not sync:
                    continue

                fused = self.fuse_sample(sync[-1]["frame"], sync[-1]["lidar"])
                samples.append(fused)
                LOGGER.info("Fused sample %d: %s", len(samples), fused)

            return samples

        finally:
            self._close_resources()

    def run_live(self, duration=10):
        self._open_camera()
        self._open_lidar()

        samples = []
        camera_frames = []
        lidar_scans = []
        stop_time = time.time() + duration

        try:
            while time.time() < stop_time:
                frame = self.camera.get_frame()
                scan = self.lidar.read_scan()
                if scan is None:
                    continue

                camera_frames.append(frame)
                lidar_scans.append(scan)
                sync = synchronise(camera_frames, lidar_scans)
                if not sync:
                    continue

                fused = self.fuse_sample(sync[-1]["frame"], sync[-1]["lidar"])
                samples.append(fused)
                LOGGER.info("Live fused sample %d: %s", len(samples), fused)
                time.sleep(0.01)

            return samples

        finally:
            self._close_resources()


def main():
    parser = argparse.ArgumentParser(description="Run the Vision-LiDAR fusion pipeline.")
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument("--lidar-log", default=None)
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    pipeline = FusionPipeline(
        camera_source="folder" if args.mode == "offline" else "camera",
        image_folder=args.image_folder,
        lidar_port=args.lidar_port,
        lidar_log=args.lidar_log,
    )

    if args.mode == "offline":
        results = pipeline.run_offline(max_samples=args.max_samples)
    else:
        results = pipeline.run_live(duration=args.duration)

    print(f"Completed {len(results)} fused samples")


if __name__ == "__main__":
    main()
