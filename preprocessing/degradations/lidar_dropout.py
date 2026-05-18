import argparse
import json
from pathlib import Path

import numpy as np


def apply_dropout(
    ranges,
    dropout_ratio=0.3,
    seed=42,
):
    np.random.seed(seed)

    degraded = ranges.copy()

    total = len(ranges)

    dropout_count = int(total * dropout_ratio)

    indices = np.random.choice(
        total,
        dropout_count,
        replace=False,
    )

    degraded[indices] = np.nan

    return degraded


def process_scan(input_path, output_path, ratio, seed):
    ranges = np.loadtxt(input_path)

    degraded = apply_dropout(
        ranges=ranges,
        dropout_ratio=ratio,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_path, degraded)

    metadata = {
        "degradation": "lidar_dropout",
        "ratio": ratio,
        "seed": seed,
    }

    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    process_scan(
        Path(args.input),
        Path(args.output),
        args.ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()


"""
python preprocessing/degradations/lidar_dropout.py \
    --input dataset/lidar/scan1.txt \
    --output dataset/degraded/lidar_dropout_30/scan1.txt \
    --ratio 0.3 \
    --seed 42
"""