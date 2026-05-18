import argparse
import json
from pathlib import Path

import pandas as pd


def apply_temporal_offset(
    dataframe,
    offset_ms=100,
):
    dataframe = dataframe.copy()

    dataframe["timestamp"] = (
        dataframe["timestamp"] + offset_ms
    )

    return dataframe


def process_csv(input_path, output_path, offset_ms):
    df = pd.read_csv(input_path)

    degraded = apply_temporal_offset(
        dataframe=df,
        offset_ms=offset_ms,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    degraded.to_csv(output_path, index=False)

    metadata = {
        "degradation": "temporal_desync",
        "offset_ms": offset_ms,
    }

    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset-ms", type=int, default=100)

    args = parser.parse_args()

    process_csv(
        Path(args.input),
        Path(args.output),
        args.offset_ms,
    )


if __name__ == "__main__":
    main()

"""
python preprocessing/degradations/temporal_desync.py \
    --input dataset/data.csv \
    --output dataset/degraded/temporal_desync_100 \
    --offset-ms 100
"""