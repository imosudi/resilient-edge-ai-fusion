import os
import time

from fusion.hardware_config import (
    DEFAULT_HOKUYO_CLUSTER_COUNT,
    DEFAULT_HOKUYO_END_STEP,
    DEFAULT_HOKUYO_START_STEP,
    DEFAULT_LIDAR_BAUDRATE,
    DEFAULT_LIDAR_END_ANGLE_DEG,
    DEFAULT_LIDAR_MAX_DISTANCE_MM,
    DEFAULT_LIDAR_MIN_DISTANCE_MM,
    DEFAULT_LIDAR_PORT,
    DEFAULT_LIDAR_PROTOCOL,
    DEFAULT_LIDAR_START_ANGLE_DEG,
    DEFAULT_LIDAR_TIMEOUT,
)
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


def read_scip_response(serial_port):
    lines = []
    while True:
        raw = serial_port.readline()
        if not raw:
            break

        line = raw.decode(errors="ignore").rstrip("\r\n")
        if line == "":
            break

        lines.append(line)

    return lines


def send_scip_command(serial_port, command):
    serial_port.write(f"{command}\n".encode("ascii"))
    serial_port.flush()
    return read_scip_response(serial_port)


def initialise_hokuyo(serial_port):
    serial_port.reset_input_buffer()
    send_scip_command(serial_port, "SCIP2.0")
    response = send_scip_command(serial_port, "BM")
    if response and len(response) >= 2 and not response[1].startswith(("00", "02")):
        raise RuntimeError(f"Hokuyo laser start failed: {response}")


def decode_scip_value(chars):
    value = 0
    for char in chars:
        value = (value << 6) + (ord(char) - 0x30)
    return value


def decode_scip_distances(response_lines):
    if len(response_lines) < 4:
        return []

    payload = ""
    for line in response_lines[3:]:
        if len(line) <= 1:
            continue

        payload += line[:-1]

    distances = []
    chunk_size = 3
    for index in range(0, len(payload) - chunk_size + 1, chunk_size):
        chunk = payload[index:index + chunk_size]
        distances.append(decode_scip_value(chunk))

    return distances


def request_hokuyo_scan(
    serial_port,
    start_step=DEFAULT_HOKUYO_START_STEP,
    end_step=DEFAULT_HOKUYO_END_STEP,
    cluster_count=DEFAULT_HOKUYO_CLUSTER_COUNT,
):
    command = f"GD{start_step:04d}{end_step:04d}{cluster_count:02d}"
    response = send_scip_command(serial_port, command)

    if len(response) >= 2 and not response[1].startswith("00"):
        raise RuntimeError(f"Hokuyo scan request failed: {response}")

    distances = decode_scip_distances(response)
    if not distances:
        return None

    return distances


class LidarCapture:

    def __init__(
        self,
        port=DEFAULT_LIDAR_PORT,
        baudrate=DEFAULT_LIDAR_BAUDRATE,
        timeout=DEFAULT_LIDAR_TIMEOUT,
        offline_log=None,
        protocol=DEFAULT_LIDAR_PROTOCOL,
        hokuyo_start_step=DEFAULT_HOKUYO_START_STEP,
        hokuyo_end_step=DEFAULT_HOKUYO_END_STEP,
        hokuyo_cluster_count=DEFAULT_HOKUYO_CLUSTER_COUNT,
        start_angle_deg=DEFAULT_LIDAR_START_ANGLE_DEG,
        end_angle_deg=DEFAULT_LIDAR_END_ANGLE_DEG,
        min_distance_mm=DEFAULT_LIDAR_MIN_DISTANCE_MM,
        max_distance_mm=DEFAULT_LIDAR_MAX_DISTANCE_MM,
    ):

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.offline_log = offline_log
        self.protocol = protocol
        self.hokuyo_start_step = hokuyo_start_step
        self.hokuyo_end_step = hokuyo_end_step
        self.hokuyo_cluster_count = hokuyo_cluster_count
        self.start_angle_deg = start_angle_deg
        self.end_angle_deg = end_angle_deg
        self.min_distance_mm = min_distance_mm
        self.max_distance_mm = max_distance_mm
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
            if self.protocol == "hokuyo":
                initialise_hokuyo(self._ser)

    def read_scan(self):
        timestamp = time.time()

        if self.offline_log:
            if self._index >= len(self._lines):
                return None

            line = self._lines[self._index]
            self._index += 1
            ranges = parse_range_line(line)
            points = project_scan(
                ranges,
                start_angle_deg=self.start_angle_deg,
                end_angle_deg=self.end_angle_deg,
                min_distance_mm=self.min_distance_mm,
                max_distance_mm=self.max_distance_mm,
            ).tolist()
            return {
                "timestamp": timestamp,
                "ranges": ranges,
                "points": points,
                "raw": line,
                "source": "offline",
                "path": self.offline_log,
                "protocol": "offline",
            }

        if self.protocol == "hokuyo":
            ranges = request_hokuyo_scan(
                self._ser,
                start_step=self.hokuyo_start_step,
                end_step=self.hokuyo_end_step,
                cluster_count=self.hokuyo_cluster_count,
            )
            if ranges is None:
                return None

            raw_line = ",".join(str(value) for value in ranges)
            source = "hokuyo"
        else:
            raw_line = self._ser.readline().decode(errors="ignore").strip()
            if not raw_line:
                return None

            ranges = parse_range_line(raw_line)
            source = "serial"

        if not ranges:
            return None

        points = project_scan(
            ranges,
            start_angle_deg=self.start_angle_deg,
            end_angle_deg=self.end_angle_deg,
            min_distance_mm=self.min_distance_mm,
            max_distance_mm=self.max_distance_mm,
        ).tolist()
        return {
            "timestamp": timestamp,
            "ranges": ranges,
            "points": points,
            "raw": raw_line,
            "source": source,
            "port": self.port,
            "protocol": self.protocol,
        }

    def close(self):
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
