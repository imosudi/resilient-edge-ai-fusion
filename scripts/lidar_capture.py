#sudo usermod -a -G dialout $USER
#python scripts/lidar_capture.py --port /dev/ttyACM0 --baudrate 115200

import argparse
import time


def capture_lidar_lines(port="/dev/ttyACM0", baudrate=115200, timeout=1):
    import serial

    with serial.Serial(port, baudrate, timeout=timeout) as serial_port:
        print("LiDAR connected successfully")

        while True:
            line = serial_port.readline().decode(errors="ignore").strip()
            if line:
                timestamp = time.time()
                print(f"[{timestamp}] {line}")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture raw LiDAR serial lines.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        capture_lidar_lines(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        print("\nLiDAR capture stopped")
    except Exception as exc:
        print(f"LiDAR capture error: {exc}")
        raise


if __name__ == "__main__":
    main()
