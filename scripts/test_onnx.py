def inspect_onnx_model(model_path="yolov8n.onnx"):
    import onnxruntime as ort

    session = ort.InferenceSession(model_path)
    return {
        "inputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_outputs()
        ],
    }


def main():
    model_info = inspect_onnx_model()

    print("Inputs:")
    for item in model_info["inputs"]:
        print(item["name"], item["shape"], item["type"])

    print("\nOutputs:")
    for item in model_info["outputs"]:
        print(item["name"], item["shape"], item["type"])

    print("\nONNX model loaded successfully")


if __name__ == "__main__":
    main()
