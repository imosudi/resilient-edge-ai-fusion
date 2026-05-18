import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def apply_low_light(image, severity=0.3):
    """
    Simulate low-light degradation.

    severity:
        0.0 -> no degradation
        1.0 -> maximum darkness
    """

    severity = np.clip(severity, 0.0, 1.0)

    brightness_factor = 1.0 - severity

    degraded = (image.astype(np.float32) * brightness_factor)

    degraded = np.clip(degraded, 0, 255).astype(np.uint8)

    return degraded


def process_image(input_path, output_path, severity):
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    degraded = apply_low_light(image, severity)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), degraded)

    metadata = {
        "degradation": "low_light",
        "severity": severity,
        "input": str(input_path),
        "output": str(output_path),
    }

    metadata_path = output_path.with_suffix(".json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Saved degraded image -> {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or directory",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--severity",
        type=float,
        default=0.5,
        help="Low-light severity (0.0 - 1.0)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_dir = Path(args.output)

    if input_path.is_file():
        output_file = output_dir / input_path.name

        process_image(
            input_path=input_path,
            output_path=output_file,
            severity=args.severity,
        )

    else:
        for image_path in input_path.glob("*.*"):
            output_file = output_dir / image_path.name

            process_image(
                input_path=image_path,
                output_path=output_file,
                severity=args.severity,
            )
            


if __name__ == "__main__":
    main()


"""
python preprocessing/degradations/low_light.py \
    --input dataset/images \
    --output dataset/degraded/low_light_30 \
    --severity 0.3
"""