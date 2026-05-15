from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n.pt")

# Load image
image_path = "dataset/test.jpg"

# Start timer
start = time.time()

# Run inference
results = model(image_path)

# End timer
end = time.time()

# Calculate latency
latency_ms = (end - start) * 1000

print(f"Inference Latency: {latency_ms:.2f} ms")

# Display results
annotated_frame = results[0].plot()

cv2.imshow("YOLOv8n Detection", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Print detections
print(results[0].boxes)