import argparse


def export_yolo_to_onnx(model_path="yolov8n.pt"):
    from ultralytics import YOLO

    model = YOLO(model_path)
    return model.export(format="onnx")


def describe_hailo_preparation(onnx_path="models/exported/yolov8n.onnx"):
    return {
        "source_artifact": onnx_path,
        "target_artifact": "models/hailo/yolov8n.hef",
        "precision": "INT8",
        "toolchain": "Hailo Dataflow Compiler",
        "runtime": "Hailo Runtime",
        "note": (
            "Compile the validated ONNX model with the Hailo toolchain on a "
            "machine where the Hailo SDK is installed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLOv8n artifacts for CPU and NPU inference paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default="yolov8n.pt")
    parser.add_argument(
        "--target",
        choices=["cpu", "npu", "both"],
        default="cpu",
        help="cpu exports ONNX; npu prints the required Hailo HEF preparation.",
    )
    args = parser.parse_args()

    export_path = None
    if args.target in {"cpu", "both"}:
        export_path = export_yolo_to_onnx(args.model_path)
        print(f"CPU ONNX export completed: {export_path}")

    if args.target in {"npu", "both"}:
        hailo_info = describe_hailo_preparation(export_path or "models/exported/yolov8n.onnx")
        print("NPU Hailo preparation required:")
        for key, value in hailo_info.items():
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
