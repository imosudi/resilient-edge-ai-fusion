from scripts.lidar_capture import summarise_lidar_line


def test_summarise_lidar_line_detects_nearest_object():
    summary = summarise_lidar_line(
        "0,1000,500,5000",
        start_angle_deg=-90,
        end_angle_deg=90,
        min_distance_mm=20,
        max_distance_mm=4000,
    )

    assert summary["angle_range"] == (-90, 90)
    assert summary["sample_count"] == 4
    assert summary["valid_count"] == 2
    assert summary["object_in_range"] is True
    assert summary["nearest_distance_mm"] == 500
    assert summary["nearest_angle_deg"] == 30


def test_summarise_lidar_line_reports_no_object_for_invalid_ranges():
    summary = summarise_lidar_line(
        "0,10,5000",
        min_distance_mm=20,
        max_distance_mm=4000,
    )

    assert summary["object_in_range"] is False
    assert summary["valid_count"] == 0
    assert summary["nearest_distance_mm"] is None
    assert summary["nearest_angle_deg"] is None
