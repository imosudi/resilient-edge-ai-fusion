import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def apply_fog(image, intensity=0.5):
    fog = np.full_like(image, 255)

    blended = cv2.addWeighted(
        image,
        1.0 - intensity,
        fog,
        intensity,
        0,
    )

    return blended


def process_image(input_path, output_path, intensity):
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(input_path)

    degraded = apply_fog(
        image=image,
        intensity=intensity,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), degraded)

    metadata = {
        "degradation": "fog",
        "intensity": intensity,
    }

    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intensity", type=float, default=0.5)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        process_image(
            input_path,
            output_dir / input_path.name,
            args.intensity,
        )

    else:
        for image_path in input_path.glob("*.*"):
            process_image(
                image_path,
                output_dir / image_path.name,
                args.intensity,
            )


if __name__ == "__main__":
    main()

"""
python preprocessing/degradations/fog_simulation.py \
    --input dataset/images \
    --output dataset/degraded/fog_intensity_50 \
    --intensity 0.5
"""