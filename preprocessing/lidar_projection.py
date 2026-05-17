import numpy as np


DEFAULT_START_ANGLE_DEG = -120.0
DEFAULT_END_ANGLE_DEG = 120.0
DEFAULT_MIN_DISTANCE_MM = 20.0
DEFAULT_MAX_DISTANCE_MM = 4000.0


def polar_to_cartesian(angle_deg, distance_mm):
    """Convert polar LiDAR measurements to Cartesian coordinates in millimetres."""
    theta = np.radians(angle_deg)
    x = np.asarray(distance_mm, dtype=float) * np.cos(theta)
    y = np.asarray(distance_mm, dtype=float) * np.sin(theta)

    if np.isscalar(angle_deg) and np.isscalar(distance_mm):
        return float(x), float(y)

    return x, y


def scan_angles(
    scan_size,
    start_angle_deg=DEFAULT_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_END_ANGLE_DEG,
    angle_increment_deg=None,
):
    """Build one angle per LiDAR range sample."""
    if scan_size <= 0:
        return np.array([], dtype=float)

    if angle_increment_deg is not None:
        return start_angle_deg + np.arange(scan_size, dtype=float) * angle_increment_deg

    if scan_size == 1:
        return np.array([start_angle_deg], dtype=float)

    return np.linspace(start_angle_deg, end_angle_deg, scan_size, dtype=float)


def valid_range_mask(
    ranges_mm,
    min_distance_mm=DEFAULT_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_MAX_DISTANCE_MM,
):
    """Return a boolean mask for finite, physically plausible LiDAR ranges."""
    ranges = np.asarray(ranges_mm, dtype=float)
    return (
        np.isfinite(ranges)
        & (ranges >= min_distance_mm)
        & (ranges <= max_distance_mm)
    )


def project_scan(
    ranges_mm,
    start_angle_deg=DEFAULT_START_ANGLE_DEG,
    end_angle_deg=DEFAULT_END_ANGLE_DEG,
    angle_increment_deg=None,
    min_distance_mm=DEFAULT_MIN_DISTANCE_MM,
    max_distance_mm=DEFAULT_MAX_DISTANCE_MM,
    include_invalid=False,
):
    """Project a 1D LiDAR range scan into Cartesian points.

    Returns an ``N x 2`` NumPy array of ``x, y`` coordinates in millimetres.
    Invalid range samples are dropped by default.
    """
    ranges = np.asarray(ranges_mm, dtype=float)
    if ranges.size == 0:
        return np.empty((0, 2), dtype=float)

    angles = scan_angles(
        scan_size=ranges.size,
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
        angle_increment_deg=angle_increment_deg,
    )

    if not include_invalid:
        mask = valid_range_mask(
            ranges,
            min_distance_mm=min_distance_mm,
            max_distance_mm=max_distance_mm,
        )
        ranges = ranges[mask]
        angles = angles[mask]

    x, y = polar_to_cartesian(angles, ranges)
    return np.column_stack((x, y))


if __name__ == "__main__":
    sample_ranges = [1000, 1200, 1500]
    points = project_scan(sample_ranges)
    for x, y in points:
        print(f"X: {x:.2f} mm, Y: {y:.2f} mm")
