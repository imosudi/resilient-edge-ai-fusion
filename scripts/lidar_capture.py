import argparse
import time


def capture_lidar_lines(
    port="/dev/ttyACM0",
    baudrate=115200,
    timeout=1,
    max_empty_reads=10,
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

    with serial_port:
        print("LiDAR connected successfully")

        while True:
            line = serial_port.readline().decode(errors="ignore").strip()
            if line:
                empty_reads = 0
                timestamp = time.time()
                print(f"[{timestamp}] {line}")
                continue

            empty_reads += 1
            if max_empty_reads is not None and empty_reads >= max_empty_reads:
                wait_time = timeout * max_empty_reads
                raise TimeoutError(
                    f"No LiDAR data received from {port} after about "
                    f"{wait_time:.1f} seconds"
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Capture raw LiDAR serial lines.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1)
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
    print("- Use --wait-forever if the sensor needs longer before streaming")


def main():
    args = parse_args()
    max_empty_reads = None if args.wait_forever else args.max_empty_reads

    try:
        capture_lidar_lines(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
            max_empty_reads=max_empty_reads,
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
