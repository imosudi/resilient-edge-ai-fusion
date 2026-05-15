import time
import psutil
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

image = "dataset/test.jpg"

# CPU usage before inference
cpu_before = psutil.cpu_percent()

# Start timing
start = time.time()

# Run inference
results = model(image)

# End timing
end = time.time()

# CPU usage after inference
cpu_after = psutil.cpu_percent()

# Metrics
latency_ms = (end - start) * 1000
fps = 1 / (end - start)

print(f"Latency: {latency_ms:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"CPU Usage Before: {cpu_before}%")
print(f"CPU Usage After: {cpu_after}%")