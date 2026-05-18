# resilient-edge-ai-fusion

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-E95420)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-black)
![Hailo](https://img.shields.io/badge/Hailo-8L-blueviolet)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A)
![Edge AI](https://img.shields.io/badge/Edge-AI-success)
![Sensor Fusion](https://img.shields.io/badge/Sensor-Fusion-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)


Failure-aware edge AI perception system using Vision-LiDAR fusion on Raspberry Pi 5 and Hailo-8L for resilient real-time object detection under degraded sensing conditions.

Benchmarking compares CPU, GPU, and NPU inference latency using YOLOv8n on MS COCO 2017 and the project-specific Vision-LiDAR dataset.



## Research Objectives

- Vision–LiDAR sensor fusion
- Failure-aware edge perception
- Adaptive confidence fusion
- Robustness under degraded sensing
- Real-time embedded AI inference
- Edge AI resilience benchmarking



## Hardware Platform

- Raspberry Pi 5
- Hailo AI HAT+ (Hailo-8L)
- Raspberry Pi Camera Module 3
- Hokuyo URG-04LX-UG-01 LiDAR



## Core Technologies

- YOLOv8n
- ONNX
- ONNX Runtime FP32 CPU baseline
- PyTorch CUDA FP32 GPU comparison
- Hailo Runtime INT8 NPU acceleration
- OpenCV
- NumPy
- Ultralytics
- Python 3.11

## Documentation

- [Technical Project Description](TECHNICAL_PROJECT_DESCRIPTION.md)

## Usage

Run the offline fusion pipeline using a local image folder and LiDAR log:

```bash
python scripts/run_fusion.py --mode offline \
  --image-folder dataset/images \
  --lidar-log dataset/lidar.log \
  --inference-target cpu \
  --max-samples 20
```

Important:
- `--mode offline` runs playback mode rather than live capture.
- `--image-folder` should point to the directory containing test frames.
- `--lidar-log` should point to a text log file with LiDAR range lines.
- `--inference-target cpu` records the ONNX Runtime FP32 baseline path.
- `--inference-target gpu` records the PyTorch CUDA FP32 comparison path.
- `--inference-target npu` records the Hailo Runtime INT8 HEF path for accelerated runs.
- `--max-samples` controls the number of fused samples produced.

Note:
sudo apt install fonts-dejavu-core
