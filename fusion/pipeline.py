import argparse
import logging
import time

import cv2

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
        camera_backend="auto",
        camera_width=640,
        camera_height=480,
        lidar_port="/dev/ttyACM0",
        lidar_baudrate=115200,
        lidar_timeout=1,
        lidar_log=None,
        lidar_protocol="raw",
        hokuyo_start_step=44,
        hokuyo_end_step=725,
        hokuyo_cluster_count=0,
        lidar_start_angle=-120.0,
        lidar_end_angle=120.0,
        lidar_min_distance=20.0,
        lidar_max_distance=4000.0,
    ):
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
        self.camera = None
        self.lidar = None

    def _open_camera(self):
        self.camera = CameraCapture(
            source=self.camera_source,
            image_folder=self.image_folder,
            width=self.camera_width,
            height=self.camera_height,
            camera_backend=self.camera_backend,
        )

    def _open_lidar(self):
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
            "camera_backend": frame.get("camera_backend"),
            "lidar_protocol": scan.get("protocol"),
            "lidar_points": len(scan.get("points", [])),
            "timestamp_delta_ms": abs(
                frame["timestamp"] - scan["timestamp"]
            ) * 1000.0,
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

    def run_live(self, duration=10, max_samples=None, display=False):
        self._open_camera()
        self._open_lidar()

        samples = []
        camera_frames = []
        lidar_scans = []
        stop_time = time.time() + duration

        try:
            while time.time() < stop_time:
                if max_samples is not None and len(samples) >= max_samples:
                    break

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

                if display:
                    if not self._show_live_display(frame, scan, fused):
                        break

                time.sleep(0.01)

            return samples

        finally:
            if display:
                cv2.destroyAllWindows()
            self._close_resources()

    def _show_live_display(self, frame, scan, fused):
        image = frame["image"].copy()
        lines = [
            f"Fused confidence: {fused['fused_confidence']:.3f}",
            f"Camera confidence: {fused['camera_confidence']:.3f}",
            f"LiDAR confidence: {fused['lidar_confidence']:.3f}",
            f"Sync delta: {fused['timestamp_delta_ms']:.1f} ms",
            f"LiDAR points: {fused['lidar_points']}",
        ]

        for index, line in enumerate(lines):
            y = 28 + index * 24
            cv2.putText(
                image,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Live Vision-LiDAR Fusion", image)
        return cv2.waitKey(1) != ord("q")


def main():
    parser = argparse.ArgumentParser(description="Run the Vision-LiDAR fusion pipeline.")
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument("--lidar-log", default=None)
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--lidar-protocol", choices=["raw", "hokuyo"], default="raw")
    parser.add_argument("--camera-backend", default="auto")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    pipeline = FusionPipeline(
        camera_source="folder" if args.mode == "offline" else "camera",
        image_folder=args.image_folder,
        camera_backend=args.camera_backend,
        lidar_port=args.lidar_port,
        lidar_log=args.lidar_log,
        lidar_protocol=args.lidar_protocol,
    )

    if args.mode == "offline":
        results = pipeline.run_offline(max_samples=args.max_samples)
    else:
        results = pipeline.run_live(
            duration=args.duration,
            max_samples=args.max_samples,
            display=args.display,
        )

    print(f"Completed {len(results)} fused samples")


if __name__ == "__main__":
    main()
