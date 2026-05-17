import time
from pathlib import Path


latency_ms = None
fps = None


def measure_yolo_baseline(
    model_path="yolov8n.pt",
    image_path="dataset/test.jpg",
    csv_path="metrics/results.csv",
):
    """Run a YOLO baseline inference pass and persist latency metrics."""
    import pandas as pd
    import psutil
    from ultralytics import YOLO

    model = YOLO(model_path)
    model(image_path)

    cpu_before = psutil.cpu_percent(interval=0.1)

    start = time.time()
    model(image_path)
    end = time.time()

    cpu_after = psutil.cpu_percent(interval=0.1)
    inference_time = end - start

    metrics = {
        "latency_ms": inference_time * 1000,
        "fps": 1 / inference_time,
        "cpu_before": cpu_before,
        "cpu_after": cpu_after,
    }

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([metrics])
    if output_path.exists():
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

    return metrics


def main():
    metrics = measure_yolo_baseline()

    global latency_ms, fps
    latency_ms = metrics["latency_ms"]
    fps = metrics["fps"]

    print("\n=== Baseline Metrics ===")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"CPU Usage Before: {metrics['cpu_before']:.2f}%")
    print(f"CPU Usage After: {metrics['cpu_after']:.2f}%")
    print("\nMetrics saved to: metrics/results.csv")


if __name__ == "__main__":
    main()
