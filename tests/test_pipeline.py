import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from fusion.pipeline import FusionPipeline


def test_offline_pipeline_fuses_synthetic_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "frame.jpg")
        lidar_log_path = os.path.join(tmpdir, "lidar.log")

        image = 255 * np.ones((64, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(image_path, image)

        with open(lidar_log_path, "w", encoding="utf-8") as lidar_file:
            lidar_file.write("1,2,3\n")
            lidar_file.write("4,5,6\n")

        pipeline = FusionPipeline(
            camera_source="folder",
            image_folder=tmpdir,
            lidar_log=lidar_log_path,
        )

        results = pipeline.run_offline(max_samples=2)

        assert len(results) == 2
        assert all("fused_confidence" in item for item in results)
        assert all(0.0 <= item["fused_confidence"] <= 1.0 for item in results)
        assert all(item["inference_target"] == "cpu" for item in results)
        assert all(item["model_artifact"] == "onnx" for item in results)
        assert all(item["precision"] == "FP32" for item in results)
        assert all(item["runtime"] == "ONNX Runtime" for item in results)


def test_pipeline_records_npu_inference_profile():
    pipeline = FusionPipeline(inference_target="npu")

    frame = {
        "timestamp": 1.0,
        "image": 255 * np.ones((64, 64, 3), dtype=np.uint8),
        "source": "folder",
    }
    scan = {
        "timestamp": 1.0,
        "ranges": list(range(200)),
        "source": "log",
    }

    result = pipeline.fuse_sample(frame, scan)

    assert result["inference_target"] == "npu"
    assert result["model_artifact"] == "hef"
    assert result["precision"] == "INT8"
    assert result["runtime"] == "Hailo Runtime"
    assert result["accelerator"] == "Hailo-8L"


def test_pipeline_records_gpu_inference_profile():
    pipeline = FusionPipeline(inference_target="gpu")

    frame = {
        "timestamp": 1.0,
        "image": 255 * np.ones((64, 64, 3), dtype=np.uint8),
        "source": "folder",
    }
    scan = {
        "timestamp": 1.0,
        "ranges": list(range(200)),
        "source": "log",
    }

    result = pipeline.fuse_sample(frame, scan)

    assert result["inference_target"] == "gpu"
    assert result["model_artifact"] == "pt"
    assert result["precision"] == "FP32"
    assert result["runtime"] == "PyTorch CUDA"
    assert result["accelerator"] == "CUDA GPU"
