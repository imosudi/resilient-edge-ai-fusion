import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_motion_kernel(kernel_size, direction="horizontal"):
    kernel = np.zeros((kernel_size, kernel_size))

    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1

    elif direction == "vertical":
        kernel[:, kernel_size // 2] = 1

    elif direction == "diagonal":
        np.fill_diagonal(kernel, 1)

    kernel = kernel / kernel_size

    return kernel


def apply_motion_blur(image, kernel_size=15, direction="horizontal"):
    kernel = build_motion_kernel(kernel_size, direction)

    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


def process_image(input_path, output_path, kernel_size, direction):
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    blurred = apply_motion_blur(
        image=image,
        kernel_size=kernel_size,
        direction=direction,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), blurred)

    metadata = {
        "degradation": "motion_blur",
        "kernel_size": kernel_size,
        "direction": direction,
        "input": str(input_path),
        "output": str(output_path),
    }

    metadata_path = output_path.with_suffix(".json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Saved blurred image -> {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--direction",
        choices=["horizontal", "vertical", "diagonal"],
        default="horizontal",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_dir = Path(args.output)

    if input_path.is_file():
        output_file = output_dir / input_path.name

        process_image(
            input_path=input_path,
            output_path=output_file,
            kernel_size=args.kernel_size,
            direction=args.direction,
        )

    else:
        for image_path in input_path.glob("*.*"):
            output_file = output_dir / image_path.name

            process_image(
                input_path=image_path,
                output_path=output_file,
                kernel_size=args.kernel_size,
                direction=args.direction,
            )


if __name__ == "__main__":
    main()


"""python preprocessing/degradations/motion_blur.py \
    --input dataset/images \
    --output dataset/degraded/motion_blur_15_horizontal \
    --kernel-size 15 \
    --direction horizontal
"""