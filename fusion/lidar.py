import os
import time

from preprocessing.lidar_projection import polar_to_cartesian, project_scan

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


def parse_range_line(line):
    """Parse a comma-separated LiDAR range line into a list of values."""
    cleaned = line.strip()
    if not cleaned:
        return []

    parts = cleaned.split(",")
    ranges = []
    for value in parts:
        try:
            ranges.append(float(value))
        except ValueError:
            continue

    return ranges


class LidarCapture:

    def __init__(
        self,
        port="/dev/ttyACM0",
        baudrate=115200,
        timeout=1,
        offline_log=None,
    ):

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.offline_log = offline_log
        self._lines = []
        self._index = 0
        self._ser = None

        if self.offline_log:
            if not os.path.isfile(self.offline_log):
                raise FileNotFoundError(
                    f"LiDAR log not found: {self.offline_log}"
                )

            with open(self.offline_log, "r", encoding="utf-8") as stream:
                self._lines = [line.strip() for line in stream if line.strip()]

            if not self._lines:
                raise RuntimeError("LiDAR log file is empty")

        else:
            if serial is None:
                raise RuntimeError("pyserial is required for live LiDAR capture")

            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout,
            )

    def read_scan(self):
        timestamp = time.time()

        if self.offline_log:
            if self._index >= len(self._lines):
                return None

            line = self._lines[self._index]
            self._index += 1
            ranges = parse_range_line(line)
            points = project_scan(ranges).tolist()
            return {
                "timestamp": timestamp,
                "ranges": ranges,
                "points": points,
                "raw": line,
                "source": "offline",
                "path": self.offline_log,
            }

        raw_line = self._ser.readline().decode(errors="ignore").strip()
        if not raw_line:
            return None

        ranges = parse_range_line(raw_line)
        points = project_scan(ranges).tolist()
        return {
            "timestamp": timestamp,
            "ranges": ranges,
            "points": points,
            "raw": raw_line,
            "source": "serial",
            "port": self.port,
        }

    def close(self):
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
