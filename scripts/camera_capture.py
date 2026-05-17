import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.vision import CameraCapture


def capture_camera_frame(
    device_index=0,
    width=640,
    height=480,
    output_path="captures/camera_test.jpg",
    show_window=False,
):
    camera = CameraCapture(
        source="camera",
        device_index=device_index,
        width=width,
        height=height,
    )

    try:
        frame = camera.get_frame()
        image = frame["image"]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"Failed to save camera frame: {output}")

        if show_window:
            cv2.imshow("Camera Test", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "timestamp": frame["timestamp"],
            "path": str(output),
            "shape": image.shape,
        }

    finally:
        camera.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Capture one test frame from camera.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", default="captures/camera_test.jpg")
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Display the captured frame in an OpenCV window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = capture_camera_frame(
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        output_path=args.output,
        show_window=args.show_window,
    )

    print("Camera connected successfully")
    print(f"Captured frame: {result['path']}")
    print(f"Image shape: {result['shape']}")
    print(f"Timestamp: {result['timestamp']}")


if __name__ == "__main__":
    main()
