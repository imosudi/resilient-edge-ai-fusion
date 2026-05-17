from scripts.lidar_capture import (
    decode_scip_distances,
    decode_scip_value,
    render_lidar_view,
    summarise_lidar_line,
)


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


def test_render_lidar_view_returns_canvas():
    canvas = render_lidar_view(
        "1000,500,2000",
        start_angle_deg=-45,
        end_angle_deg=45,
        canvas_size=300,
    )

    assert canvas.shape == (300, 300, 3)


def test_decode_scip_value():
    assert decode_scip_value("000") == 0
    assert decode_scip_value("00A") == 17


def test_decode_scip_distances_skips_protocol_header():
    response = [
        "GD0044072500",
        "00P",
        "123456789",
        "00000AX",
    ]

    assert decode_scip_distances(response) == [0, 17]
