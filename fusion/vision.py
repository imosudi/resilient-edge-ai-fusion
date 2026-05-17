import cv2
import glob
import os
import time


class CameraCapture:

    def __init__(
        self,
        source="camera",
        image_folder=None,
        device_index=0,
        width=640,
        height=480,
    ):

        self.source = source
        self.device_index = device_index
        self.width = width
        self.height = height
        self.image_folder = image_folder
        self._image_paths = []
        self._current_index = 0
        self.cap = None

        if self.source == "camera":
            self.cap = cv2.VideoCapture(self.device_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Unable to open camera device {self.device_index}"
                )

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

    def get_frame(self):

        timestamp = time.time()

        if self.source == "camera":
            ret, image = self.cap.read()
            if not ret or image is None:
                raise RuntimeError("Failed to read frame from camera")

            return {
                "timestamp": timestamp,
                "image": image,
                "source": "camera",
                "device_index": self.device_index,
            }

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
