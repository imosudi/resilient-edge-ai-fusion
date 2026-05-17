from scripts.lidar_capture import (
    capture_lidar_lines,
    decode_scip_distances,
    decode_scip_value,
    render_lidar_view,
    summarise_lidar_line,
)
from scripts.camera_capture import capture_camera_frame, stream_camera_frames
from fusion.hardware_config import (
    DEFAULT_CAMERA_BACKEND,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_STREAM_FPS,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_HOKUYO_END_STEP,
    DEFAULT_HOKUYO_START_STEP,
    DEFAULT_LIDAR_END_ANGLE_DEG,
    DEFAULT_LIDAR_MAX_DISTANCE_MM,
    DEFAULT_LIDAR_START_ANGLE_DEG,
)
from fusion.pipeline import FusionPipeline


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


def test_lidar_capture_and_fusion_defaults_stay_aligned():
    pipeline = FusionPipeline()

    assert capture_lidar_lines.__defaults__[6] == DEFAULT_LIDAR_START_ANGLE_DEG
    assert capture_lidar_lines.__defaults__[7] == DEFAULT_LIDAR_END_ANGLE_DEG
    assert capture_lidar_lines.__defaults__[9] == DEFAULT_LIDAR_MAX_DISTANCE_MM
    assert capture_lidar_lines.__defaults__[12] == DEFAULT_HOKUYO_START_STEP
    assert capture_lidar_lines.__defaults__[13] == DEFAULT_HOKUYO_END_STEP

    assert pipeline.lidar_start_angle == DEFAULT_LIDAR_START_ANGLE_DEG
    assert pipeline.lidar_end_angle == DEFAULT_LIDAR_END_ANGLE_DEG
    assert pipeline.lidar_max_distance == DEFAULT_LIDAR_MAX_DISTANCE_MM
    assert pipeline.hokuyo_start_step == DEFAULT_HOKUYO_START_STEP
    assert pipeline.hokuyo_end_step == DEFAULT_HOKUYO_END_STEP


def test_camera_capture_and_fusion_defaults_stay_aligned():
    pipeline = FusionPipeline()

    assert capture_camera_frame.__defaults__[1] == DEFAULT_CAMERA_WIDTH
    assert capture_camera_frame.__defaults__[2] == DEFAULT_CAMERA_HEIGHT
    assert capture_camera_frame.__defaults__[5] == DEFAULT_CAMERA_BACKEND
    assert stream_camera_frames.__defaults__[1] == DEFAULT_CAMERA_WIDTH
    assert stream_camera_frames.__defaults__[2] == DEFAULT_CAMERA_HEIGHT
    assert stream_camera_frames.__defaults__[3] == DEFAULT_CAMERA_BACKEND
    assert stream_camera_frames.__defaults__[5] == DEFAULT_CAMERA_STREAM_FPS

    assert pipeline.camera_width == DEFAULT_CAMERA_WIDTH
    assert pipeline.camera_height == DEFAULT_CAMERA_HEIGHT
    assert pipeline.camera_backend == DEFAULT_CAMERA_BACKEND


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
