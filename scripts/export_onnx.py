def export_yolo_to_onnx(model_path="yolov8n.pt"):
    from ultralytics import YOLO

    model = YOLO(model_path)
    return model.export(format="onnx")


def main():
    export_path = export_yolo_to_onnx()
    print(f"ONNX export completed: {export_path}")


if __name__ == "__main__":
    main()
