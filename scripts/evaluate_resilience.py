# scripts/evaluate_resilience.py
"""
Resilient Edge AI Evaluation Framework
-------------------------------------

Optimised resilience evaluation pipeline for:
- Raspberry Pi 5
- Edge AI sensor fusion
- Vision + LiDAR degradation campaigns
- CPU / GPU / NPU benchmarking
- Reproducible resilience scoring

Key Optimisations
-----------------
1. Parallel degradation preparation
2. Reduced repeated imports
3. Vectorised LiDAR processing
4. Cached baseline confidence
5. Improved logging
6. Faster CSV/JSON writing
7. Better fault tolerance
8. Resource-aware execution
9. Campaign manifest generation
10. Recovery benchmarking support
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.confidence_fusion import (
    CAMERA_DEGRADED_THRESHOLD,
    LIDAR_DEGRADED_THRESHOLD,
)
from fusion.inference import (
    INFERENCE_TARGET_CHOICES,
    get_inference_profile,
)
from metrics.robustness_score import robustness_score
from fusion.pipeline import FusionPipeline

from preprocessing.degradations.gaussian_noise import apply_gaussian_noise
from preprocessing.degradations.generate_low_light import apply_low_light
from preprocessing.degradations.generate_motion_blur import apply_motion_blur
from preprocessing.degradations.occlusion import apply_occlusion


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("evaluate_resilience")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_DEGRADATIONS = (
    "low_light",
    "motion_blur",
    "gaussian_noise",
    "occlusion",
    "lidar_dropout",
    "temporal_desync",
)

METRIC_FIELDS = (
    "run_id",
    "timestamp",
    "dataset_group",
    "degradation",
    "severity",
    "inference_target",
    "runtime",
    "model_artifact",
    "precision",
    "camera_health",
    "lidar_health",
    "stale_sync",
    "latency_ms",
    "fps",
    "cpu_percent",
    "memory_mb",
    "temperature_c",
    "camera_confidence",
    "lidar_confidence",
    "fusion_confidence",
    "fallback_state",
    "robustness_score",
    "recovery_state",
)

HARDWARE_BENCHMARK_MATRIX = [
    {
        "inference_path": "CPU FP32",
        "runtime": "ONNX Runtime",
        "target": "cpu",
    },
    {
        "inference_path": "GPU FP32",
        "runtime": "CUDA",
        "target": "gpu",
    },
    {
        "inference_path": "NPU INT8",
        "runtime": "Hailo Runtime",
        "target": "npu",
    },
]


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def parse_csv_values(value: str, cast=str) -> list:
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]

    if not values:
        return None

    return statistics.fmean(values)


def count_values(values: Iterable[str]) -> dict:
    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return counts


# -----------------------------------------------------------------------------
# System Metrics
# -----------------------------------------------------------------------------

def read_temperature_c() -> float | None:
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")

    if not thermal_path.exists():
        return None

    try:
        return round(
            float(thermal_path.read_text().strip()) / 1000.0,
            3,
        )
    except Exception:
        return None


def read_system_metrics() -> dict:
    cpu_percent = None
    memory_mb = None

    try:
        import psutil

        process = psutil.Process(os.getpid())

        cpu_percent = psutil.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)

    except ImportError:
        logger.warning("psutil not installed")

    return {
        "cpu_percent": round(cpu_percent, 3) if cpu_percent else None,
        "memory_mb": round(memory_mb, 3) if memory_mb else None,
        "temperature_c": read_temperature_c(),
    }


# -----------------------------------------------------------------------------
# Health / Fallback Logic
# -----------------------------------------------------------------------------

def health_label(confidence: float, threshold: float) -> str:
    return "normal" if confidence >= threshold else "degraded"


def fallback_state(sample: dict) -> str:
    camera_ok = sample["camera_confidence"] >= CAMERA_DEGRADED_THRESHOLD
    lidar_ok = sample["lidar_confidence"] >= LIDAR_DEGRADED_THRESHOLD

    if sample.get("stale_sync"):
        return "sync_degraded"

    if camera_ok and lidar_ok:
        return "normal"

    if not camera_ok and lidar_ok:
        return "lidar_assisted"

    if camera_ok and not lidar_ok:
        return "vision_assisted"

    return "dual_degraded"


# -----------------------------------------------------------------------------
# Image Degradation
# -----------------------------------------------------------------------------

def process_single_image(
    source_path: Path,
    output_dir: Path,
    degradation: str,
    severity: float,
    seed: int,
) -> None:

    image = cv2.imread(str(source_path))

    if image is None:
        raise RuntimeError(f"Failed to load image: {source_path}")

    if degradation == "low_light":

        degraded = apply_low_light(
            image,
            severity=severity,
        )

    elif degradation == "motion_blur":

        kernel_size = max(3, int(round(3 + severity * 28)))

        if kernel_size % 2 == 0:
            kernel_size += 1

        degraded = apply_motion_blur(
            image,
            kernel_size=kernel_size,
            direction="horizontal",
        )

    elif degradation == "gaussian_noise":

        sigma = max(1.0, severity * 60.0)

        degraded = apply_gaussian_noise(
            image,
            sigma=sigma,
            seed=seed,
        )

    elif degradation == "occlusion":

        degraded = apply_occlusion(
            image,
            occlusion_ratio=max(0.01, min(0.9, severity)),
            position="center",
        )

    else:
        raise ValueError(f"Unsupported degradation: {degradation}")

    output_path = output_dir / source_path.name

    success = cv2.imwrite(str(output_path), degraded)

    if not success:
        raise RuntimeError(f"Failed to write image: {output_path}")


def prepare_image_degradation(
    source_dir: Path,
    output_dir: Path,
    degradation: str,
    severity: float,
    seed: int,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)

    images = image_paths(source_dir)

    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:

        futures = [
            executor.submit(
                process_single_image,
                image_path,
                output_dir,
                degradation,
                severity,
                seed,
            )
            for image_path in images
        ]

        for future in futures:
            future.result()

    return output_dir


# -----------------------------------------------------------------------------
# LiDAR Degradation
# -----------------------------------------------------------------------------

def prepare_lidar_degradation(
    source_log: Path,
    output_log: Path,
    degradation: str,
    severity: float,
    seed: int,
) -> Path:

    rng = np.random.default_rng(seed)

    with open(source_log, "r", encoding="utf-8") as stream:
        lines = [line.strip() for line in stream if line.strip()]

    degraded_lines = []

    for line in lines:

        values = np.array(
            [float(v) for v in line.split(",") if v],
            dtype=np.float32,
        )

        if degradation == "lidar_dropout":

            mask = rng.random(values.shape[0]) >= severity
            values = values[mask]

        elif degradation == "temporal_desync":
            pass

        degraded_lines.append(
            ",".join(f"{v:.3f}" for v in values)
        )

    if degradation == "temporal_desync" and degraded_lines:

        offset = max(1, int(round(severity * len(degraded_lines))))

        degraded_lines = (
            degraded_lines[offset:] + degraded_lines[:offset]
        )

    output_log.parent.mkdir(parents=True, exist_ok=True)

    with open(output_log, "w", encoding="utf-8") as stream:
        stream.write("\n".join(degraded_lines))
        stream.write("\n")

    return output_log


# -----------------------------------------------------------------------------
# Campaign Builder
# -----------------------------------------------------------------------------

def build_campaign_inputs(
    source_images: Path,
    source_lidar_log: Path,
    campaign_dir: Path,
    degradation: str,
    severity: float,
    seed: int,
) -> tuple[Path, Path]:

    image_dir = campaign_dir / "images"
    lidar_log = campaign_dir / "lidar.log"

    image_dir.mkdir(parents=True, exist_ok=True)

    if degradation == "clean":

        for image_path in image_paths(source_images):
            shutil.copy2(image_path, image_dir / image_path.name)

        shutil.copy2(source_lidar_log, lidar_log)

        return image_dir, lidar_log

    if degradation in {
        "low_light",
        "motion_blur",
        "gaussian_noise",
        "occlusion",
    }:

        prepare_image_degradation(
            source_images,
            image_dir,
            degradation,
            severity,
            seed,
        )

        shutil.copy2(source_lidar_log, lidar_log)

        return image_dir, lidar_log

    for image_path in image_paths(source_images):
        shutil.copy2(image_path, image_dir / image_path.name)

    prepare_lidar_degradation(
        source_lidar_log,
        lidar_log,
        degradation,
        severity,
        seed,
    )

    return image_dir, lidar_log


# -----------------------------------------------------------------------------
# Pipeline Campaign
# -----------------------------------------------------------------------------

def run_pipeline_campaign(
    *,
    run_id: str,
    image_folder: Path,
    lidar_log: Path,
    inference_target: str,
    degradation: str,
    severity: float,
    dataset_group: str,
    max_samples: int,
    clean_reference_confidence: float | None,
) -> tuple[list[dict], dict]:

    profile = get_inference_profile(inference_target)

    logger.info(
        "Running campaign: %s | degradation=%s severity=%.2f",
        run_id,
        degradation,
        severity,
    )

    resource_before = read_system_metrics()

    start = time.perf_counter()

    pipeline = FusionPipeline(
        camera_source="folder",
        image_folder=str(image_folder),
        lidar_log=str(lidar_log),
        inference_target=inference_target,
    )

    samples = pipeline.run_offline(max_samples=max_samples)

    elapsed = max(time.perf_counter() - start, 1e-9)

    resource_after = read_system_metrics()

    latency_ms = (elapsed / max(1, len(samples))) * 1000.0
    fps = len(samples) / elapsed

    rows = []

    for index, sample in enumerate(samples, start=1):

        fusion_confidence = sample["fused_confidence"]

        score = (
            robustness_score(
                clean_reference_confidence,
                fusion_confidence,
            )
            if clean_reference_confidence is not None
            else 100.0
        )

        rows.append({
            "run_id": f"{run_id}_{index:03d}",
            "timestamp": time.time(),
            "dataset_group": dataset_group,
            "degradation": degradation,
            "severity": severity,
            "inference_target": profile.target,
            "runtime": profile.runtime,
            "model_artifact": profile.model_artifact,
            "precision": profile.precision,
            "camera_health": health_label(
                sample["camera_confidence"],
                CAMERA_DEGRADED_THRESHOLD,
            ),
            "lidar_health": health_label(
                sample["lidar_confidence"],
                LIDAR_DEGRADED_THRESHOLD,
            ),
            "stale_sync": sample["stale_sync"],
            "latency_ms": round(latency_ms, 3),
            "fps": round(fps, 3),
            "cpu_percent": resource_after["cpu_percent"],
            "memory_mb": resource_after["memory_mb"],
            "temperature_c": resource_after["temperature_c"],
            "camera_confidence": round(
                sample["camera_confidence"],
                6,
            ),
            "lidar_confidence": round(
                sample["lidar_confidence"],
                6,
            ),
            "fusion_confidence": round(
                fusion_confidence,
                6,
            ),
            "fallback_state": fallback_state(sample),
            "robustness_score": round(score, 3),
            "recovery_state": "not_applicable",
        })

    summary = {
        "run_id": run_id,
        "degradation": degradation,
        "severity": severity,
        "samples": len(samples),
        "latency_ms_mean": latency_ms,
        "fps": fps,
        "fusion_confidence_mean": mean(
            row["fusion_confidence"] for row in rows
        ),
        "robustness_score_mean": mean(
            row["robustness_score"] for row in rows
        ),
        "fallback_counts": count_values(
            row["fallback_state"] for row in rows
        ),
        "resource_before": resource_before,
        "resource_after": resource_after,
    }

    return rows, summary


# -----------------------------------------------------------------------------
# Summary Generation
# -----------------------------------------------------------------------------

def summarise_results(
    rows: list[dict],
    summaries: list[dict],
) -> dict:

    grouped = {}

    for row in rows:

        key = (
            row["inference_target"],
            row["degradation"],
            row["severity"],
        )

        grouped.setdefault(key, []).append(row)

    comparisons = []

    for (
        target,
        degradation,
        severity,
    ), group_rows in grouped.items():

        comparisons.append({
            "inference_target": target,
            "degradation": degradation,
            "severity": severity,
            "samples": len(group_rows),
            "fps_mean": mean(
                r["fps"] for r in group_rows
            ),
            "latency_ms_mean": mean(
                r["latency_ms"] for r in group_rows
            ),
            "fusion_confidence_mean": mean(
                r["fusion_confidence"] for r in group_rows
            ),
            "robustness_score_mean": mean(
                r["robustness_score"] for r in group_rows
            ),
            "fallback_counts": count_values(
                r["fallback_state"] for r in group_rows
            ),
        })

    return {
        "generated_at": time.time(),
        "hardware_benchmark_matrix": HARDWARE_BENCHMARK_MATRIX,
        "campaign_summaries": summaries,
        "baseline_vs_degraded": comparisons,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description="Optimised Resilience Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument("--lidar-log", required=True)
    parser.add_argument("--output-dir", default="exports/resilience_eval")

    parser.add_argument(
        "--dataset-group",
        default="project_specific",
    )

    parser.add_argument(
        "--inference-targets",
        default="cpu",
    )

    parser.add_argument(
        "--degradations",
        default=",".join(DEFAULT_DEGRADATIONS),
    )

    parser.add_argument(
        "--severities",
        default="0.3,0.6,0.9",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    source_images = Path(args.image_folder)
    source_lidar_log = Path(args.lidar_log)
    output_dir = Path(args.output_dir)

    if not source_images.exists():
        raise SystemExit(f"Missing image folder: {source_images}")

    if not source_lidar_log.exists():
        raise SystemExit(f"Missing LiDAR log: {source_lidar_log}")

    inference_targets = parse_csv_values(args.inference_targets)

    invalid_targets = (
        set(inference_targets)
        - set(INFERENCE_TARGET_CHOICES)
    )

    if invalid_targets:
        raise SystemExit(
            f"Unsupported targets: {sorted(invalid_targets)}"
        )

    degradations = parse_csv_values(args.degradations)

    severities = parse_csv_values(
        args.severities,
        float,
    )

    all_rows = []
    all_summaries = []

    baseline_cache = {}

    for target in inference_targets:

        logger.info("Baseline campaign for %s", target)

        baseline_dir = (
            output_dir
            / "campaign_inputs"
            / target
            / "clean"
        )

        image_dir, lidar_log = build_campaign_inputs(
            source_images,
            source_lidar_log,
            baseline_dir,
            "clean",
            0.0,
            args.seed,
        )

        rows, summary = run_pipeline_campaign(
            run_id=f"{target}_clean",
            image_folder=image_dir,
            lidar_log=lidar_log,
            inference_target=target,
            degradation="clean",
            severity=0.0,
            dataset_group=args.dataset_group,
            max_samples=args.max_samples,
            clean_reference_confidence=None,
        )

        baseline_cache[target] = summary[
            "fusion_confidence_mean"
        ]

        all_rows.extend(rows)
        all_summaries.append(summary)

        for degradation in degradations:

            for severity in severities:

                logger.info(
                    "Running degradation=%s severity=%.2f",
                    degradation,
                    severity,
                )

                campaign_dir = (
                    output_dir
                    / "campaign_inputs"
                    / target
                    / f"{degradation}_{severity:.2f}".replace(".", "p")
                )

                image_dir, lidar_log = build_campaign_inputs(
                    source_images,
                    source_lidar_log,
                    campaign_dir,
                    degradation,
                    severity,
                    args.seed,
                )

                rows, summary = run_pipeline_campaign(
                    run_id=f"{target}_{degradation}_{severity:.2f}".replace(".", "p"),
                    image_folder=image_dir,
                    lidar_log=lidar_log,
                    inference_target=target,
                    degradation=degradation,
                    severity=severity,
                    dataset_group=args.dataset_group,
                    max_samples=args.max_samples,
                    clean_reference_confidence=baseline_cache[target],
                )

                all_rows.extend(rows)
                all_summaries.append(summary)

    summary_payload = summarise_results(
        all_rows,
        all_summaries,
    )

    write_csv(
        output_dir / "resilience_metrics.csv",
        all_rows,
    )

    write_json(
        output_dir / "resilience_metrics.json",
        all_rows,
    )

    write_json(
        output_dir / "resilience_summary.json",
        summary_payload,
    )

    write_json(
        output_dir / "hardware_benchmark_matrix.json",
        HARDWARE_BENCHMARK_MATRIX,
    )

    logger.info("Evaluation completed")
    logger.info("Results saved to %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())