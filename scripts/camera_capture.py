import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.hardware_config import (
    CAMERA_BACKEND_CHOICES,
    DEFAULT_CAMERA_BACKEND,
    DEFAULT_CAMERA_DEVICE_INDEX,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_OUTPUT,
    DEFAULT_CAMERA_STREAM_FPS,
    DEFAULT_CAMERA_STREAM_OUTPUT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_STREAM_DURATION,
)
from fusion.vision import CameraCapture


def capture_camera_frame(
    device_index=DEFAULT_CAMERA_DEVICE_INDEX,
    width=DEFAULT_CAMERA_WIDTH,
    height=DEFAULT_CAMERA_HEIGHT,
    output_path=DEFAULT_CAMERA_OUTPUT,
    show_window=False,
    backend=DEFAULT_CAMERA_BACKEND,
):
    if backend == "libcamera":
        return capture_with_libcamera(
            width=width,
            height=height,
            output_path=output_path,
            show_window=show_window,
        )

    try:
        return capture_with_camera_capture(
            device_index=device_index,
            width=width,
            height=height,
            output_path=output_path,
            show_window=show_window,
            backend=backend,
        )
    except RuntimeError as camera_error:
        if backend != "auto":
            raise

        try:
            return capture_with_libcamera(
                width=width,
                height=height,
                output_path=output_path,
                show_window=show_window,
            )
        except RuntimeError as libcamera_error:
            raise RuntimeError(
                "All camera backends failed. "
                f"OpenCV/Picamera2: {camera_error}. "
                f"libcamera/rpicam: {libcamera_error}."
            ) from libcamera_error


def stream_camera_frames(
    device_index=DEFAULT_CAMERA_DEVICE_INDEX,
    width=DEFAULT_CAMERA_WIDTH,
    height=DEFAULT_CAMERA_HEIGHT,
    backend=DEFAULT_CAMERA_BACKEND,
    duration=DEFAULT_STREAM_DURATION,
    fps=DEFAULT_CAMERA_STREAM_FPS,
    show_window=False,
    output_path=DEFAULT_CAMERA_STREAM_OUTPUT,
):
    if backend == "libcamera":
        return stream_with_libcamera_stills(
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            show_window=show_window,
            output_path=output_path,
        )

    try:
        camera = CameraCapture(
            source="camera",
            device_index=device_index,
            width=width,
            height=height,
            camera_backend=backend,
        )
    except RuntimeError:
        if backend != "auto":
            raise

        return stream_with_libcamera_stills(
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            show_window=show_window,
            output_path=output_path,
        )

    frame_count = 0
    interval = 1.0 / fps if fps > 0 else 0.0
    started_at = time.time()
    next_frame_at = started_at
    last_shape = None
    last_image = None
    active_backend = backend

    try:
        print("Camera stream connected successfully")
        while time.time() - started_at < duration:
            now = time.time()
            if now < next_frame_at:
                time.sleep(next_frame_at - now)

            frame = camera.get_frame()
            image = frame["image"]
            last_image = image
            active_backend = frame.get("camera_backend", backend)
            last_shape = image.shape
            frame_count += 1
            print(
                f"Frame {frame_count}: shape={last_shape}, "
                f"backend={active_backend}, timestamp={frame['timestamp']}"
            )

            if show_window:
                cv2.imshow("Camera Stream Test", image)
                if cv2.waitKey(1) == ord("q"):
                    break

            next_frame_at += interval

        elapsed = max(time.time() - started_at, 1e-9)
        if last_image is None:
            raise RuntimeError("Camera stream ended before any frames were captured")

        save_last_stream_frame(last_image, output_path)
        return {
            "frames": frame_count,
            "duration": elapsed,
            "fps": frame_count / elapsed,
            "shape": last_shape,
            "backend": active_backend,
            "path": output_path,
        }

    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()


def stream_with_libcamera_stills(
    width=DEFAULT_CAMERA_WIDTH,
    height=DEFAULT_CAMERA_HEIGHT,
    duration=DEFAULT_STREAM_DURATION,
    fps=1,
    show_window=False,
    output_path=DEFAULT_CAMERA_STREAM_OUTPUT,
):
    frame_count = 0
    interval = 1.0 / fps if fps > 0 else 0.0
    started_at = time.time()
    last_result = None

    print("Camera stream connected successfully")
    while time.time() - started_at < duration:
        last_result = capture_with_libcamera(
            width=width,
            height=height,
            output_path=output_path,
            show_window=False,
        )
        frame_count += 1
        print(
            f"Frame {frame_count}: shape={last_result['shape']}, "
            f"backend={last_result['backend']}, timestamp={last_result['timestamp']}"
        )
        if show_window:
            image = cv2.imread(output_path)
            if image is not None:
                cv2.imshow("Camera Stream Test", image)
                if cv2.waitKey(1) == ord("q"):
                    break

        if interval > 0:
            time.sleep(interval)

    elapsed = max(time.time() - started_at, 1e-9)
    if show_window:
        cv2.destroyAllWindows()

    if frame_count == 0:
        raise RuntimeError("Camera stream ended before any frames were captured")

    return {
        "frames": frame_count,
        "duration": elapsed,
        "fps": frame_count / elapsed,
        "shape": None if last_result is None else last_result["shape"],
        "backend": "libcamera",
        "path": output_path,
    }


def save_last_stream_frame(image, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to save camera frame: {output}")


def capture_with_camera_capture(
    device_index=DEFAULT_CAMERA_DEVICE_INDEX,
    width=DEFAULT_CAMERA_WIDTH,
    height=DEFAULT_CAMERA_HEIGHT,
    output_path=DEFAULT_CAMERA_OUTPUT,
    show_window=False,
    backend=DEFAULT_CAMERA_BACKEND,
):
    camera = CameraCapture(
        source="camera",
        device_index=device_index,
        width=width,
        height=height,
        camera_backend=backend,
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
            "backend": frame.get("camera_backend", backend),
        }

    finally:
        camera.release()


def capture_with_libcamera(
    width=DEFAULT_CAMERA_WIDTH,
    height=DEFAULT_CAMERA_HEIGHT,
    output_path=DEFAULT_CAMERA_OUTPUT,
    show_window=False,
):
    camera_command = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if camera_command is None:
        raise RuntimeError("Neither rpicam-still nor libcamera-still was found")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        camera_command,
        "-n",
        "--width",
        str(width),
        "--height",
        str(height),
        "--timeout",
        "1000",
        "-o",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"{Path(camera_command).name} failed: {details}") from exc

    image = cv2.imread(str(output))
    if image is None:
        raise RuntimeError(f"Camera command ran, but output could not be read: {output}")

    if show_window:
        cv2.imshow("Camera Test", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "timestamp": time.time(),
        "path": str(output),
        "shape": image.shape,
        "backend": Path(camera_command).name,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one test frame from camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=DEFAULT_CAMERA_DEVICE_INDEX,
        help="OpenCV camera device index.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Camera frame width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Camera frame height.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_CAMERA_OUTPUT,
        help="Output image path.",
    )
    parser.add_argument(
        "--backend",
        choices=CAMERA_BACKEND_CHOICES,
        default=DEFAULT_CAMERA_BACKEND,
        help="Camera backend. Use picamera2 or libcamera for Raspberry Pi Camera Module.",
    )
    parser.add_argument(
        "--show-window",
        "--display",
        dest="show_window",
        action="store_true",
        help="Display the camera feed in an OpenCV window.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Run a live camera stream test.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_STREAM_DURATION,
        help="Live stream test duration in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_CAMERA_STREAM_FPS,
        help="Target stream test frame rate.",
    )
    return parser.parse_args()


def print_camera_error(error):
    print("\nCamera test failed")
    print(f"Reason: {error}")
    print("\nSuggested checks:")
    print("- Raspberry Pi Camera Module: try --backend picamera2 or --backend libcamera")
    print("- Check detection with: rpicam-hello --list-cameras")
    print("- Older Raspberry Pi OS: libcamera-hello --list-cameras")
    print("- Make sure the ribbon cable is seated and the camera is enabled")
    print("- If using Picamera2 in a venv, install/enable the system Picamera2 package")
    print("- USB webcam: try --backend opencv --device-index 0 or --device-index 1")
    print("- Use --stream --duration 10 for a bounded live stream test")


def main():
    args = parse_args()
    output_path = args.output
    if args.stream and output_path == DEFAULT_CAMERA_OUTPUT:
        output_path = DEFAULT_CAMERA_STREAM_OUTPUT

    try:
        if args.stream:
            result = stream_camera_frames(
                device_index=args.device_index,
                width=args.width,
                height=args.height,
                output_path=output_path,
                show_window=args.show_window,
                backend=args.backend,
                duration=args.duration,
                fps=args.fps,
            )
        else:
            result = capture_camera_frame(
                device_index=args.device_index,
                width=args.width,
                height=args.height,
                output_path=output_path,
                show_window=args.show_window,
                backend=args.backend,
            )
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        print_camera_error(exc)
        return 1

    if args.stream:
        print("Camera stream test completed")
        print(f"Frames captured: {result['frames']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Average FPS: {result['fps']:.2f}")
        print(f"Last frame: {result['path']}")
        print(f"Image shape: {result['shape']}")
        print(f"Backend: {result['backend']}")
    else:
        print("Camera connected successfully")
        print(f"Captured frame: {result['path']}")
        print(f"Image shape: {result['shape']}")
        print(f"Backend: {result['backend']}")
        print(f"Timestamp: {result['timestamp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
