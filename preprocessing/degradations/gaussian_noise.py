import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def apply_gaussian_noise(image, mean=0, sigma=25, seed=None):
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(
        mean,
        sigma,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def process_image(input_path, output_path, sigma, seed):
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(input_path)

    noisy = apply_gaussian_noise(
        image=image,
        sigma=sigma,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), noisy)

    metadata = {
        "degradation": "gaussian_noise",
        "sigma": sigma,
        "seed": seed,
    }

    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        process_image(
            input_path,
            output_dir / input_path.name,
            args.sigma,
            args.seed,
        )

    else:
        for image_path in input_path.glob("*.*"):
            process_image(
                image_path,
                output_dir / image_path.name,
                args.sigma,
                args.seed,
            )


if __name__ == "__main__":
    main()

"""
python preprocessing/degradations/gaussian_noise.py \
    --input dataset/images \
    --output dataset/degraded/gaussian_noise_25 \
    --sigma 25 \
    --seed 42
"""