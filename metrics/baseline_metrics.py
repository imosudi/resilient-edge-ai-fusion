import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.inference import (
    CPU_TARGET,
    GPU_TARGET,
    INFERENCE_TARGET_CHOICES,
    NPU_TARGET,
    get_inference_profile,
)


latency_ms = None
fps = None


def measure_yolo_baseline(
    model_path="yolov8n.pt",
    image_path="dataset/test.jpg",
    csv_path="metrics/results.csv",
    inference_target=CPU_TARGET,
):
    """Run a YOLO baseline inference pass and persist latency metrics."""
    import pandas as pd
    import psutil
    from ultralytics import YOLO

    inference_profile = get_inference_profile(inference_target)
    if inference_profile.target == NPU_TARGET:
        raise NotImplementedError(
            "NPU benchmarking requires a compiled HEF model and Hailo Runtime. "
            "Use this helper for CPU/GPU YOLO baselines, then record Hailo "
            "measurements from the Hailo runtime path once Stage 9 is complete."
        )

    yolo_device = "cuda" if inference_profile.target == GPU_TARGET else "cpu"
    model = YOLO(model_path)
    model(image_path, device=yolo_device)

    cpu_before = psutil.cpu_percent(interval=0.1)

    start = time.time()
    model(image_path, device=yolo_device)
    end = time.time()

    cpu_after = psutil.cpu_percent(interval=0.1)
    inference_time = end - start

    metrics = {
        "inference_target": inference_profile.target,
        "inference_label": inference_profile.label,
        "model_artifact": inference_profile.model_artifact,
        "precision": inference_profile.precision,
        "runtime": inference_profile.runtime,
        "accelerator": inference_profile.accelerator,
        "latency_ms": inference_time * 1000,
        "fps": 1 / inference_time,
        "cpu_before": cpu_before,
        "cpu_after": cpu_after,
    }

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([metrics])
    if output_path.exists():
        existing_columns = list(pd.read_csv(output_path, nrows=0).columns)
        if existing_columns == list(df.columns):
            df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Measure YOLO inference metrics for a deployment profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default="yolov8n.pt")
    parser.add_argument("--image-path", default="dataset/test.jpg")
    parser.add_argument("--csv-path", default="metrics/results.csv")
    parser.add_argument(
        "--inference-target",
        choices=INFERENCE_TARGET_CHOICES,
        default=CPU_TARGET,
        help=(
            "Deployment profile recorded with the metrics. "
            "Use cpu for ONNX/FP32 baseline, gpu for CUDA comparison, "
            "or npu for Hailo/INT8 HEF runs."
        ),
    )
    args = parser.parse_args()

    metrics = measure_yolo_baseline(
        model_path=args.model_path,
        image_path=args.image_path,
        csv_path=args.csv_path,
        inference_target=args.inference_target,
    )

    global latency_ms, fps
    latency_ms = metrics["latency_ms"]
    fps = metrics["fps"]

    print("\n=== Baseline Metrics ===")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Inference Target: {metrics['inference_target']}")
    print(f"Runtime: {metrics['runtime']}")
    print(f"Precision: {metrics['precision']}")
    print(f"CPU Usage Before: {metrics['cpu_before']:.2f}%")
    print(f"CPU Usage After: {metrics['cpu_after']:.2f}%")
    print(f"\nMetrics saved to: {args.csv_path}")


if __name__ == "__main__":
    main()
