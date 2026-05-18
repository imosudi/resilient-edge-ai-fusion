import pytest

from fusion.inference import get_inference_profile


def test_cpu_inference_profile_describes_onnx_fp32_baseline():
    profile = get_inference_profile("cpu")

    assert profile.target == "cpu"
    assert profile.model_artifact == "onnx"
    assert profile.precision == "FP32"
    assert profile.runtime == "ONNX Runtime"


def test_npu_inference_profile_describes_hailo_int8_hef_path():
    profile = get_inference_profile("npu")

    assert profile.target == "npu"
    assert profile.model_artifact == "hef"
    assert profile.precision == "INT8"
    assert profile.runtime == "Hailo Runtime"


def test_hailo_alias_maps_to_npu_profile():
    profile = get_inference_profile("hailo")

    assert profile.target == "npu"


def test_unknown_inference_profile_is_rejected():
    with pytest.raises(ValueError):
        get_inference_profile("gpu")
