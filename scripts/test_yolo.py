import time


def run_yolo_smoke_test(
    model_path="yolov8n.pt",
    image_path="dataset/test.jpg",
    show_window=True,
):
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    model(image_path)

    start = time.time()
    results = model(image_path)
    end = time.time()

    annotated_frame = results[0].plot()
    if show_window:
        cv2.imshow("YOLOv8n Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "latency_ms": (end - start) * 1000,
        "boxes": results[0].boxes,
    }


def main():
    result = run_yolo_smoke_test()

    print(f"Inference Latency: {result['latency_ms']:.2f} ms")
    print(result["boxes"])


if __name__ == "__main__":
    main()
