import cv2
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

from fusion.hardware_config import (
    DEFAULT_CAMERA_BACKEND,
    DEFAULT_CAMERA_DEVICE_INDEX,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
)


class CameraCapture:

    def __init__(
        self,
        source="camera",
        image_folder=None,
        device_index=DEFAULT_CAMERA_DEVICE_INDEX,
        width=DEFAULT_CAMERA_WIDTH,
        height=DEFAULT_CAMERA_HEIGHT,
        camera_backend=DEFAULT_CAMERA_BACKEND,
    ):

        self.source = source
        self.device_index = device_index
        self.width = width
        self.height = height
        self.camera_backend = camera_backend
        self._requested_camera_backend = camera_backend
        self.image_folder = image_folder
        self._image_paths = []
        self._current_index = 0
        self.cap = None
        self.picam2 = None
        self.libcamera_command = None
        self._libcamera_frame_path = Path("/tmp/resilient_edge_ai_camera_frame.jpg")

        if self.source == "camera":
            self._open_camera()

        elif self.source == "folder":
            if not self.image_folder:
                raise ValueError("image_folder must be provided for folder source")

            if not os.path.isdir(self.image_folder):
                raise FileNotFoundError(
                    f"Image folder not found: {self.image_folder}"
                )

            self._image_paths = sorted(
                glob.glob(os.path.join(self.image_folder, "*.jpg"))
                + glob.glob(os.path.join(self.image_folder, "*.png"))
            )

            if not self._image_paths:
                raise FileNotFoundError(
                    f"No images found in {self.image_folder}"
                )

        else:
            raise ValueError(f"Unsupported source type: {self.source}")

    def _open_camera(self):
        errors = []

        if self.camera_backend in ("auto", "opencv"):
            try:
                self._open_opencv_camera()
                return
            except RuntimeError as exc:
                errors.append(str(exc))
                if self.camera_backend == "opencv":
                    raise

        if self.camera_backend in ("auto", "picamera2"):
            try:
                self._open_picamera2_camera()
                return
            except RuntimeError as exc:
                errors.append(str(exc))
                if self.camera_backend == "picamera2":
                    raise

        if self.camera_backend in ("auto", "libcamera"):
            try:
                self._open_libcamera_camera()
                return
            except RuntimeError as exc:
                errors.append(str(exc))
                if self.camera_backend == "libcamera":
                    raise

        raise RuntimeError("Unable to open camera. " + " | ".join(errors))

    def _open_opencv_camera(self):
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"Unable to open OpenCV camera device {self.device_index}"
            )

        self.camera_backend = "opencv"

    def _open_picamera2_camera(self):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("Picamera2 is not installed") from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)
        self.camera_backend = "picamera2"

    def _open_libcamera_camera(self):
        self.libcamera_command = (
            shutil.which("rpicam-still")
            or shutil.which("libcamera-still")
        )
        if self.libcamera_command is None:
            raise RuntimeError("Neither rpicam-still nor libcamera-still was found")

        self.camera_backend = "libcamera"

    def get_frame(self):

        timestamp = time.time()

        if self.source == "camera":
            if self.camera_backend == "opencv":
                ret, image = self.cap.read()
                if not ret or image is None:
                    if self._requested_camera_backend == "auto":
                        self.cap.release()
                        self.cap = None
                        self._open_picamera2_camera()
                    else:
                        raise RuntimeError("Failed to read frame from OpenCV camera")
                else:
                    return {
                        "timestamp": timestamp,
                        "image": image,
                        "source": "camera",
                        "device_index": self.device_index,
                        "camera_backend": self.camera_backend,
                    }

            if self.camera_backend == "picamera2":
                image = self.picam2.capture_array()
                if image is None:
                    raise RuntimeError("Failed to read frame from Picamera2")

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return {
                    "timestamp": timestamp,
                    "image": image,
                    "source": "camera",
                    "device_index": self.device_index,
                    "camera_backend": self.camera_backend,
                }

            if self.camera_backend == "libcamera":
                command = [
                    self.libcamera_command,
                    "-n",
                    "--width",
                    str(self.width),
                    "--height",
                    str(self.height),
                    "--timeout",
                    "1",
                    "-o",
                    str(self._libcamera_frame_path),
                ]
                try:
                    subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as exc:
                    details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
                    raise RuntimeError(f"libcamera frame capture failed: {details}") from exc

                image = cv2.imread(str(self._libcamera_frame_path))
                if image is None:
                    raise RuntimeError("Failed to read libcamera frame output")

                return {
                    "timestamp": timestamp,
                    "image": image,
                    "source": "camera",
                    "device_index": self.device_index,
                    "camera_backend": self.camera_backend,
                }

            raise RuntimeError(f"Unsupported camera backend: {self.camera_backend}")

        image_path = self._image_paths[self._current_index]
        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        result = {
            "timestamp": timestamp,
            "image": image,
            "source": "folder",
            "path": image_path,
        }

        self._current_index = min(
            self._current_index + 1,
            len(self._image_paths) - 1,
        )

        return result

    def release(self):
        if self.cap is not None:
            self.cap.release()

        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()


def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")

    return {
        "timestamp": time.time(),
        "image": image,
        "source": "file",
        "path": path,
    }
