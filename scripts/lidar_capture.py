import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.hardware_config import (
    DEFAULT_HOKUYO_CLUSTER_COUNT,
    DEFAULT_HOKUYO_END_STEP,
    DEFAULT_HOKUYO_START_STEP,
    DEFAULT_LIDAR_BAUDRATE,
    DEFAULT_LIDAR_END_ANGLE_DEG,
    DEFAULT_LIDAR_MAX_DISTANCE_MM,
    DEFAULT_LIDAR_MAX_EMPTY_READS,
    DEFAULT_LIDAR_MIN_DISTANCE_MM,
    DEFAULT_LIDAR_PORT,
    DEFAULT_LIDAR_PROTOCOL,
    DEFAULT_LIDAR_START_ANGLE_DEG,
    DEFAULT_LIDAR_TIMEOUT,
    LIDAR_PROTOCOL_CHOICES,
)
from fusion.lidar import (
    decode_scip_distances,
    decode_scip_value,
    initialise_hokuyo,
    parse_range_line,
    request_hokuyo_scan,
)
from preprocessing.lidar_projection import (
    polar_to_cartesian,
    scan_angles,
    valid_range_mask,
)


def open_serial_port(
    port=DEFAULT_LIDAR_PORT,
    baudrate=DEFAULT_LIDAR_BAUDRATE,
    timeout=DEFAULT_LIDAR_TIMEOUT,
):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed. Install it with: pip install pyserial"
        ) from exc

    try:
        return serial.Serial(port, baudrate, timeout=timeout)
    except serial.SerialException as exc:
        raise RuntimeError(f"Unable to open LiDAR serial port {port}: {exc}") from exc


def capture_lidar_lines(
    port=DEFAULT_LIDAR_PORT,
    baudrate=DEFAULT_LIDAR_BAUDRATE,
    timeout=DEFAULT_LIDAR_TIMEOUT,
    max_empty_reads=DEFAULT_LIDAR_MAX_EMPTY_READS,
    max_lines=1,
    duration=None,
    start_angle_deg=DEFAULT_LIDAR_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_LIDAR_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_LIDAR_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_LIDAR_MAX_DISTANCE_MM,
    show_window=False,
    output_path=None,
    protocol=DEFAULT_LIDAR_PROTOCOL,
    hokuyo_start_step=DEFAULT_HOKUYO_START_STEP,
    hokuyo_end_step=DEFAULT_HOKUYO_END_STEP,
    hokuyo_cluster_count=DEFAULT_HOKUYO_CLUSTER_COUNT,
):
    serial_port = open_serial_port(port=port, baudrate=baudrate, timeout=timeout)
    output_file = None
    if output_path is not None:
        output_file = open(output_path, 'w', encoding='utf-8')

    empty_reads = 0
    captured_lines = 0
    start_time = time.time()

    try:
        with serial_port:
            print('LiDAR connected successfully')
            if protocol == 'hokuyo':
                initialise_hokuyo(serial_port)

            while True:
                if duration is not None and time.time() - start_time >= duration:
                    print(
                        'LiDAR stream test completed: '
                        f'captured {captured_lines} line(s) in {duration:.1f} second(s)'
                    )
                    return captured_lines

                if protocol == 'hokuyo':
                    ranges = request_hokuyo_scan(
                        serial_port,
                        start_step=hokuyo_start_step,
                        end_step=hokuyo_end_step,
                        cluster_count=hokuyo_cluster_count,
                    )
                    line = '' if ranges is None else ','.join(str(value) for value in ranges)
                else:
                    line = serial_port.readline().decode(errors='ignore').strip()

                if line:
                    empty_reads = 0
                    captured_lines += 1
                    timestamp = time.time()
                    print(f'[{timestamp}] {line}')
                    print_lidar_sample_summary(
                        line=line,
                        start_angle_deg=start_angle_deg,
                        end_angle_deg=end_angle_deg,
                        min_distance_mm=min_distance_mm,
                        max_distance_mm=max_distance_mm,
                    )
                    if output_file is not None:
                        output_file.write(line + '\n')
                        output_file.flush()
                        output_file.flush()

                    if show_window:
                        should_continue = show_lidar_window(
                            line=line,
                            start_angle_deg=start_angle_deg,
                            end_angle_deg=end_angle_deg,
                            min_distance_mm=min_distance_mm,
                            max_distance_mm=max_distance_mm,
                            wait_ms=1 if max_lines is None else 0,
                        )
                        if not should_continue:
                            print('LiDAR display stopped')
                            return captured_lines

                    if max_lines is not None and captured_lines >= max_lines:
                        print(f'LiDAR test completed: captured {captured_lines} line(s)')
                        return captured_lines

                    continue

                empty_reads += 1
                if max_empty_reads is not None and empty_reads >= max_empty_reads:
                    wait_time = timeout * max_empty_reads
                    raise TimeoutError(
                        f'No LiDAR data received from {port} after about '
                        f'{wait_time:.1f} seconds'
                    )
    finally:
        if output_file is not None:
            output_file.close()


def render_lidar_view(
    line,
    start_angle_deg=DEFAULT_LIDAR_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_LIDAR_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_LIDAR_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_LIDAR_MAX_DISTANCE_MM,
    canvas_size=700,
):
    import cv2
    import numpy as np

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

    canvas = np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)
    origin = (canvas_size // 2, canvas_size - 70)
    radius_px = int(canvas_size * 0.42)
    scale = radius_px / max_distance_mm

    def to_screen(angle_deg, distance_mm):
        forward_mm, lateral_mm = polar_to_cartesian(angle_deg, distance_mm)
        screen_x = int(origin[0] + lateral_mm * scale)
        screen_y = int(origin[1] - forward_mm * scale)
        return screen_x, screen_y

    cv2.circle(canvas, origin, 5, (20, 20, 20), -1)
    for ratio in (0.25, 0.5, 0.75, 1.0):
        radius = int(radius_px * ratio)
        distance = int(max_distance_mm * ratio)
        cv2.circle(canvas, origin, radius, (220, 220, 220), 1)
        cv2.putText(
            canvas,
            f"{distance}mm",
            (origin[0] + 8, origin[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

    for angle in (start_angle_deg, end_angle_deg):
        cv2.line(canvas, origin, to_screen(angle, max_distance_mm), (180, 180, 180), 1)

    valid_indices = [index for index, is_valid in enumerate(mask.tolist()) if is_valid]
    nearest_index = None
    if valid_indices:
        nearest_index = min(valid_indices, key=lambda index: ranges[index])

    for index in valid_indices:
        point = to_screen(float(angles[index]), ranges[index])
        color = (40, 120, 255) if index == nearest_index else (60, 160, 60)
        size = 6 if index == nearest_index else 3
        cv2.circle(canvas, point, size, color, -1)

    cv2.putText(
        canvas,
        f"LiDAR live view | {start_angle_deg:.1f} to {end_angle_deg:.1f} deg",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Valid range: {min_distance_mm:.0f}-{max_distance_mm:.0f} mm",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    if nearest_index is not None:
        cv2.putText(
            canvas,
            f"Nearest: {ranges[nearest_index]:.1f} mm at {angles[nearest_index]:.1f} deg",
            (18, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (40, 80, 220),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "Object within range: no",
            (18, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Press q to close",
        (18, canvas_size - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return canvas


def show_lidar_window(
    line,
    start_angle_deg=DEFAULT_LIDAR_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_LIDAR_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_LIDAR_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_LIDAR_MAX_DISTANCE_MM,
    wait_ms=1,
):
    import cv2

    canvas = render_lidar_view(
        line=line,
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
        min_distance_mm=min_distance_mm,
        max_distance_mm=max_distance_mm,
    )
    cv2.imshow("LiDAR Live View", canvas)
    key = cv2.waitKey(wait_ms)
    if key == ord("q"):
        cv2.destroyWindow("LiDAR Live View")
        return False
    return True


def summarise_lidar_line(
    line,
    start_angle_deg=DEFAULT_LIDAR_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_LIDAR_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_LIDAR_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_LIDAR_MAX_DISTANCE_MM,
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
    start_angle_deg=DEFAULT_LIDAR_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_LIDAR_END_ANGLE_DEG,
    min_distance_mm=DEFAULT_LIDAR_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_LIDAR_MAX_DISTANCE_MM,
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
    parser = argparse.ArgumentParser(
        description="Capture raw LiDAR serial lines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=DEFAULT_LIDAR_PORT, help="LiDAR serial port.")
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_LIDAR_BAUDRATE,
        help="LiDAR serial baudrate.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_LIDAR_TIMEOUT,
        help="LiDAR serial timeout in seconds.",
    )
    parser.add_argument(
        "--protocol",
        choices=LIDAR_PROTOCOL_CHOICES,
        default=DEFAULT_LIDAR_PROTOCOL,
        help="LiDAR protocol. Hokuyo mode polls SCIP scans; raw mode reads text lines.",
    )
    parser.add_argument(
        "--hokuyo-start-step",
        type=int,
        default=DEFAULT_HOKUYO_START_STEP,
        help="Hokuyo SCIP start step for GD scan command.",
    )
    parser.add_argument(
        "--hokuyo-end-step",
        type=int,
        default=DEFAULT_HOKUYO_END_STEP,
        help="Hokuyo SCIP end step for GD scan command.",
    )
    parser.add_argument(
        "--hokuyo-cluster-count",
        type=int,
        default=DEFAULT_HOKUYO_CLUSTER_COUNT,
        help="Hokuyo SCIP cluster count for GD scan command.",
    )
    parser.add_argument(
        "--start-angle",
        type=float,
        default=DEFAULT_LIDAR_START_ANGLE_DEG,
        help="Start angle in degrees for projecting LiDAR samples.",
    )
    parser.add_argument(
        "--end-angle",
        type=float,
        default=DEFAULT_LIDAR_END_ANGLE_DEG,
        help="End angle in degrees for projecting LiDAR samples.",
    )
    parser.add_argument(
        "--min-distance-mm",
        type=float,
        default=DEFAULT_LIDAR_MIN_DISTANCE_MM,
        help="Minimum valid LiDAR distance in millimetres.",
    )
    parser.add_argument(
        "--max-distance-mm",
        type=float,
        default=DEFAULT_LIDAR_MAX_DISTANCE_MM,
        help="Maximum valid LiDAR distance in millimetres.",
    )
    parser.add_argument(
        "--max-empty-reads",
        type=int,
        default=DEFAULT_LIDAR_MAX_EMPTY_READS,
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
        "--duration",
        type=float,
        default=None,
        help="Maximum stream duration in seconds.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream continuously until Ctrl+C or --duration expires.",
    )
    parser.add_argument(
        "--show-window",
        "--display",
        dest="show_window",
        action="store_true",
        help="Display a live top-down LiDAR view in an OpenCV window.",
    )
    return parser.parse_args()


def print_lidar_error(error):
    print("\nLiDAR test failed")
    print(f"Reason: {error}")
    print("\nSuggested checks:")
    print("- Confirm the device path: ls /dev/ttyACM* /dev/ttyUSB*")
    print("- Try: python scripts/lidar_capture.py --port /dev/ttyUSB0")
    print("- Hokuyo URG sensors normally need SCIP polling; default protocol is hokuyo")
    print("- For devices that already print CSV/text lines, use --protocol raw")
    print("- Check permissions: sudo usermod -a -G dialout $USER")
    print("- Log out and back in after changing the dialout group")
    print("- Check cable, power, and baudrate for the Hokuyo serial interface")
    print("- Default mode captures one line and exits; use --stream for continuous output")
    print("- Use --stream --duration 10 for a bounded live stream test")
    print("- Use --stream --display for the live LiDAR view")
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
            duration=args.duration,
            start_angle_deg=args.start_angle,
            end_angle_deg=args.end_angle,
            min_distance_mm=args.min_distance_mm,
            max_distance_mm=args.max_distance_mm,
            show_window=args.show_window,
            protocol=args.protocol,
            hokuyo_start_step=args.hokuyo_start_step,
            hokuyo_end_step=args.hokuyo_end_step,
            hokuyo_cluster_count=args.hokuyo_cluster_count,
        )
    except KeyboardInterrupt:
        print("\nLiDAR capture stopped")
        return 0
    except (RuntimeError, TimeoutError, OSError) as exc:
        print_lidar_error(exc)
        return 1
    finally:
        if args.show_window:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
