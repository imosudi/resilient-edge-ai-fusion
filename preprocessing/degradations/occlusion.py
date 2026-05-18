import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def apply_occlusion(
    image,
    occlusion_ratio=0.3,
    position="center",
):
    h, w = image.shape[:2]

    occ_w = int(w * occlusion_ratio)
    occ_h = int(h * occlusion_ratio)

    if position == "center":
        x = (w - occ_w) // 2
        y = (h - occ_h) // 2

    elif position == "top":
        x = (w - occ_w) // 2
        y = 0

    elif position == "bottom":
        x = (w - occ_w) // 2
        y = h - occ_h

    else:
        x = 0
        y = 0

    occluded = image.copy()

    cv2.rectangle(
        occluded,
        (x, y),
        (x + occ_w, y + occ_h),
        (0, 0, 0),
        -1,
    )

    return occluded


def process_image(input_path, output_path, ratio, position):
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(input_path)

    degraded = apply_occlusion(
        image=image,
        occlusion_ratio=ratio,
        position=position,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), degraded)

    metadata = {
        "degradation": "occlusion",
        "ratio": ratio,
        "position": position,
    }

    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument(
        "--position",
        choices=["center", "top", "bottom"],
        default="center",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        process_image(
            input_path,
            output_dir / input_path.name,
            args.ratio,
            args.position,
        )

    else:
        for image_path in input_path.glob("*.*"):
            process_image(
                image_path,
                output_dir / image_path.name,
                args.ratio,
                args.position,
            )


if __name__ == "__main__":
    main()


"""
python preprocessing/degradations/occlusion.py \
    --input dataset/images \
    --output dataset/degraded/occlusion_30_center \
    --ratio 0.3 \
    --position center
"""