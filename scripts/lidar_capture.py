import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.lidar import parse_range_line
from preprocessing.lidar_projection import (
    DEFAULT_END_ANGLE_DEG,
    DEFAULT_MAX_DISTANCE_MM,
    DEFAULT_MIN_DISTANCE_MM,
    DEFAULT_START_ANGLE_DEG,
    scan_angles,
    valid_range_mask,
)


def capture_lidar_lines(
    port="/dev/ttyACM0",
    baudrate=115200,
    timeout=1,
    max_empty_reads=10,
    max_lines=1,
    start_angle_deg=DEFAULT_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_MAX_DISTANCE_MM,
):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed. Install it with: pip install pyserial"
        ) from exc

    try:
        serial_port = serial.Serial(port, baudrate, timeout=timeout)
    except serial.SerialException as exc:
        raise RuntimeError(f"Unable to open LiDAR serial port {port}: {exc}") from exc

    empty_reads = 0
    captured_lines = 0

    with serial_port:
        print("LiDAR connected successfully")

        while True:
            line = serial_port.readline().decode(errors="ignore").strip()
            if line:
                empty_reads = 0
                captured_lines += 1
                timestamp = time.time()
                print(f"[{timestamp}] {line}")
                print_lidar_sample_summary(
                    line=line,
                    start_angle_deg=start_angle_deg,
                    end_angle_deg=end_angle_deg,
                    min_distance_mm=min_distance_mm,
                    max_distance_mm=max_distance_mm,
                )

                if max_lines is not None and captured_lines >= max_lines:
                    print(f"LiDAR test completed: captured {captured_lines} line(s)")
                    return captured_lines

                continue

            empty_reads += 1
            if max_empty_reads is not None and empty_reads >= max_empty_reads:
                wait_time = timeout * max_empty_reads
                raise TimeoutError(
                    f"No LiDAR data received from {port} after about "
                    f"{wait_time:.1f} seconds"
                )


def summarise_lidar_line(
    line,
    start_angle_deg=DEFAULT_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_MAX_DISTANCE_MM,
):
    ranges = parse_range_line(line)
    angles = scan_angles(
        scan_size=len(ranges),
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
    )
    mask = valid_range_mask(
        ranges,
        min_distance_mm=min_distance_mm,
        max_distance_mm=max_distance_mm,
    )

    valid_indices = [index for index, is_valid in enumerate(mask.tolist()) if is_valid]
    if not valid_indices:
        return {
            "angle_range": (start_angle_deg, end_angle_deg),
            "sample_count": len(ranges),
            "valid_count": 0,
            "object_in_range": False,
            "nearest_distance_mm": None,
            "nearest_angle_deg": None,
        }

    nearest_index = min(valid_indices, key=lambda index: ranges[index])
    return {
        "angle_range": (start_angle_deg, end_angle_deg),
        "sample_count": len(ranges),
        "valid_count": len(valid_indices),
        "object_in_range": True,
        "nearest_distance_mm": ranges[nearest_index],
        "nearest_angle_deg": float(angles[nearest_index]),
    }


def print_lidar_sample_summary(
    line,
    start_angle_deg=DEFAULT_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_MAX_DISTANCE_MM,
):
    summary = summarise_lidar_line(
        line=line,
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
        min_distance_mm=min_distance_mm,
        max_distance_mm=max_distance_mm,
    )

    print(
        "Angle range: "
        f"{summary['angle_range'][0]:.1f} deg to "
        f"{summary['angle_range'][1]:.1f} deg"
    )
    print(
        "Range window: "
        f"{min_distance_mm:.0f} mm to {max_distance_mm:.0f} mm"
    )
    print(
        "Parsed ranges: "
        f"{summary['sample_count']} total, {summary['valid_count']} valid"
    )

    if summary["object_in_range"]:
        print("Object within range: yes")
        print(
            "Nearest object: "
            f"{summary['nearest_distance_mm']:.1f} mm at "
            f"{summary['nearest_angle_deg']:.1f} deg"
        )
    else:
        print("Object within range: no")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture raw LiDAR serial lines.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1)
    parser.add_argument("--start-angle", type=float, default=DEFAULT_START_ANGLE_DEG)
    parser.add_argument("--end-angle", type=float, default=DEFAULT_END_ANGLE_DEG)
    parser.add_argument("--min-distance-mm", type=float, default=DEFAULT_MIN_DISTANCE_MM)
    parser.add_argument("--max-distance-mm", type=float, default=DEFAULT_MAX_DISTANCE_MM)
    parser.add_argument(
        "--max-empty-reads",
        type=int,
        default=10,
        help="Stop with a clear error after this many empty reads.",
    )
    parser.add_argument(
        "--wait-forever",
        action="store_true",
        help="Keep waiting when no LiDAR lines are received.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of LiDAR lines to capture before exiting.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream continuously until Ctrl+C.",
    )
    return parser.parse_args()


def print_lidar_error(error):
    print("\nLiDAR test failed")
    print(f"Reason: {error}")
    print("\nSuggested checks:")
    print("- Confirm the device path: ls /dev/ttyACM* /dev/ttyUSB*")
    print("- Try: python scripts/lidar_capture.py --port /dev/ttyUSB0")
    print("- Check permissions: sudo usermod -a -G dialout $USER")
    print("- Log out and back in after changing the dialout group")
    print("- Check cable, power, and baudrate for the Hokuyo serial interface")
    print("- Default mode captures one line and exits; use --stream for continuous output")
    print("- Use --wait-forever if the sensor needs longer before streaming")


def main():
    args = parse_args()
    max_empty_reads = None if args.wait_forever else args.max_empty_reads
    max_lines = None if args.stream else args.samples

    try:
        capture_lidar_lines(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
            max_empty_reads=max_empty_reads,
            max_lines=max_lines,
            start_angle_deg=args.start_angle,
            end_angle_deg=args.end_angle,
            min_distance_mm=args.min_distance_mm,
            max_distance_mm=args.max_distance_mm,
        )
    except KeyboardInterrupt:
        print("\nLiDAR capture stopped")
        return 0
    except (RuntimeError, TimeoutError, OSError) as exc:
        print_lidar_error(exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
