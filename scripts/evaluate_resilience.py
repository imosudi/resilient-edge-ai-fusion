import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.confidence_fusion import (  # noqa: E402
    CAMERA_DEGRADED_THRESHOLD,
    LIDAR_DEGRADED_THRESHOLD,
)
from fusion.inference import INFERENCE_TARGET_CHOICES, get_inference_profile  # noqa: E402
from metrics.robustness_score import robustness_score  # noqa: E402


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


def parse_csv_values(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def image_paths(image_folder: Path) -> list[Path]:
    return sorted(
        path
        for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_clean_images(source_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths(source_dir):
        shutil.copy2(image_path, output_dir / image_path.name)
    return output_dir


def prepare_image_degradation(
    source_dir: Path,
    output_dir: Path,
    degradation: str,
    severity: float,
    seed: int,
) -> Path:
    import cv2

    from preprocessing.degradations.gaussian_noise import apply_gaussian_noise
    from preprocessing.degradations.generate_low_light import apply_low_light
    from preprocessing.degradations.generate_motion_blur import apply_motion_blur
    from preprocessing.degradations.occlusion import apply_occlusion

    output_dir.mkdir(parents=True, exist_ok=True)

    for source_path in image_paths(source_dir):
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")

        if degradation == "low_light":
            degraded = apply_low_light(image, severity=severity)
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
            raise ValueError(f"Unsupported image degradation: {degradation}")

        output_path = output_dir / source_path.name
        if not cv2.imwrite(str(output_path), degraded):
            raise RuntimeError(f"Failed to write degraded image: {output_path}")

    write_json(
        output_dir / "degradation_metadata.json",
        {
            "degradation": degradation,
            "severity": severity,
            "seed": seed,
            "source": str(source_dir),
        },
    )
    return output_dir


def prepare_lidar_degradation(
    source_log: Path,
    output_log: Path,
    degradation: str,
    severity: float,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    with open(source_log, "r", encoding="utf-8") as source:
        lines = [line.strip() for line in source if line.strip()]

    degraded_lines = []
    for line in lines:
        values = []
        for item in line.split(","):
            try:
                values.append(float(item))
            except ValueError:
                continue

        if degradation == "lidar_dropout":
            if values:
                mask = rng.random(len(values)) < severity
                values = [
                    value for value, drop in zip(values, mask, strict=False)
                    if not drop
                ]
        elif degradation == "temporal_desync":
            # The pipeline timestamps samples at read time. We simulate desync
            # by rotating scan order so frames pair with stale spatial content.
            pass
        else:
            raise ValueError(f"Unsupported LiDAR degradation: {degradation}")

        degraded_lines.append(",".join(format_range(value) for value in values))

    if degradation == "temporal_desync" and degraded_lines:
        offset = max(1, int(round(severity * len(degraded_lines))))
        degraded_lines = degraded_lines[offset:] + degraded_lines[:offset]

    with open(output_log, "w", encoding="utf-8") as destination:
        destination.write("\n".join(degraded_lines))
        destination.write("\n")

    write_json(
        output_log.with_suffix(".json"),
        {
            "degradation": degradation,
            "severity": severity,
            "seed": seed,
            "source": str(source_log),
            "output": str(output_log),
        },
    )
    return output_log


def format_range(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


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

    if degradation == "clean":
        prepare_clean_images(source_images, image_dir)
        shutil.copy2(source_lidar_log, lidar_log)
        return image_dir, lidar_log

    if degradation in {"low_light", "motion_blur", "gaussian_noise", "occlusion"}:
        prepare_image_degradation(
            source_images,
            image_dir,
            degradation=degradation,
            severity=severity,
            seed=seed,
        )
        shutil.copy2(source_lidar_log, lidar_log)
        return image_dir, lidar_log

    prepare_clean_images(source_images, image_dir)
    prepare_lidar_degradation(
        source_lidar_log,
        lidar_log,
        degradation=degradation,
        severity=severity,
        seed=seed,
    )
    return image_dir, lidar_log


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
    recovery_state: str = "not_applicable",
) -> tuple[list[dict], dict]:
    from fusion.pipeline import FusionPipeline

    profile = get_inference_profile(inference_target)
    resource_before = read_system_metrics()
    started = time.perf_counter()

    pipeline = FusionPipeline(
        camera_source="folder",
        image_folder=str(image_folder),
        lidar_log=str(lidar_log),
        inference_target=inference_target,
    )
    samples = pipeline.run_offline(max_samples=max_samples)

    elapsed = max(time.perf_counter() - started, 1e-9)
    resource_after = read_system_metrics()
    latency_ms = (elapsed / max(1, len(samples))) * 1000.0
    fps = len(samples) / elapsed

    rows = []
    for index, sample in enumerate(samples, start=1):
        fusion_confidence = sample["fused_confidence"]
        score = 100.0
        if clean_reference_confidence is not None:
            score = robustness_score(clean_reference_confidence, fusion_confidence)

        rows.append(
            {
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
                "camera_confidence": round(sample["camera_confidence"], 6),
                "lidar_confidence": round(sample["lidar_confidence"], 6),
                "fusion_confidence": round(fusion_confidence, 6),
                "fallback_state": fallback_state(sample),
                "robustness_score": round(score, 3),
                "recovery_state": recovery_state,
            }
        )

    summary = {
        "run_id": run_id,
        "degradation": degradation,
        "severity": severity,
        "inference_target": profile.target,
        "runtime": profile.runtime,
        "samples": len(samples),
        "elapsed_s": elapsed,
        "latency_ms_mean": latency_ms,
        "fps": fps,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "fusion_confidence_mean": mean(
            row["fusion_confidence"] for row in rows
        ),
        "robustness_score_mean": mean(
            row["robustness_score"] for row in rows
        ),
        "fallback_counts": count_values(row["fallback_state"] for row in rows),
    }
    return rows, summary


def read_system_metrics() -> dict:
    cpu_percent = None
    memory_mb = None

    try:
        import psutil

        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except ImportError:
        pass

    return {
        "cpu_percent": round(cpu_percent, 3) if cpu_percent is not None else None,
        "memory_mb": round(memory_mb, 3) if memory_mb is not None else None,
        "temperature_c": read_temperature_c(),
    }


def read_temperature_c() -> float | None:
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not thermal_path.exists():
        return None

    try:
        return round(float(thermal_path.read_text(encoding="utf-8").strip()) / 1000.0, 3)
    except (OSError, ValueError):
        return None


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


def recovery_state(
    baseline_confidence: float | None,
    recovery_confidence: float | None,
    tolerance: float,
) -> str:
    if baseline_confidence is None or recovery_confidence is None:
        return "not_evaluated"
    if baseline_confidence == 0:
        return "not_evaluated"

    ratio = recovery_confidence / baseline_confidence
    if ratio >= tolerance:
        return "recovered"
    if ratio >= tolerance * 0.75:
        return "partial_recovery"
    return "not_recovered"


def mean(values) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return statistics.fmean(values)


def count_values(values) -> dict:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarise_results(rows: list[dict], summaries: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        key = (
            row["inference_target"],
            row["degradation"],
            row["severity"],
        )
        grouped.setdefault(key, []).append(row)

    comparisons = []
    for (target, degradation, severity), group_rows in sorted(grouped.items()):
        comparisons.append(
            {
                "inference_target": target,
                "degradation": degradation,
                "severity": severity,
                "samples": len(group_rows),
                "latency_ms_mean": mean(row["latency_ms"] for row in group_rows),
                "fps_mean": mean(row["fps"] for row in group_rows),
                "fusion_confidence_mean": mean(
                    row["fusion_confidence"] for row in group_rows
                ),
                "robustness_score_mean": mean(
                    row["robustness_score"] for row in group_rows
                ),
                "fallback_counts": count_values(
                    row["fallback_state"] for row in group_rows
                ),
                "recovery_counts": count_values(
                    row["recovery_state"] for row in group_rows
                ),
            }
        )

    return {
        "generated_at": time.time(),
        "hardware_benchmark_matrix": HARDWARE_BENCHMARK_MATRIX,
        "campaign_summaries": summaries,
        "baseline_vs_degraded": comparisons,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible resilience evaluation campaigns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument("--lidar-log", required=True)
    parser.add_argument("--output-dir", default="exports/resilience_eval")
    parser.add_argument("--dataset-group", default="project_specific")
    parser.add_argument(
        "--inference-targets",
        default="cpu",
        help="Comma-separated targets from: cpu,gpu,npu",
    )
    parser.add_argument(
        "--degradations",
        default=",".join(DEFAULT_DEGRADATIONS),
        help="Comma-separated degradations or 'none'.",
    )
    parser.add_argument(
        "--severities",
        default="0.3,0.6,0.9",
        help="Comma-separated severity values in [0, 1].",
    )
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recovery-tolerance",
        type=float,
        default=0.9,
        help="Recovery confidence ratio required to mark a recovery run as recovered.",
    )
    args = parser.parse_args()

    source_images = Path(args.image_folder)
    source_lidar_log = Path(args.lidar_log)
    output_dir = Path(args.output_dir)
    campaign_root = output_dir / "campaign_inputs"

    if not source_images.is_dir():
        raise SystemExit(f"Image folder not found: {source_images}")
    if not image_paths(source_images):
        raise SystemExit(f"No images found in: {source_images}")
    if not source_lidar_log.is_file():
        raise SystemExit(f"LiDAR log not found: {source_lidar_log}")

    inference_targets = parse_csv_values(args.inference_targets)
    invalid_targets = set(inference_targets) - set(INFERENCE_TARGET_CHOICES)
    if invalid_targets:
        raise SystemExit(f"Unsupported inference targets: {sorted(invalid_targets)}")

    degradations = parse_csv_values(args.degradations)
    if degradations == ["none"]:
        degradations = []

    severities = parse_csv_values(args.severities, float)

    all_rows = []
    all_summaries = []
    clean_reference_by_target = {}

    for target in inference_targets:
        baseline_dir = campaign_root / target / "clean"
        image_dir, lidar_log = build_campaign_inputs(
            source_images,
            source_lidar_log,
            baseline_dir,
            degradation="clean",
            severity=0.0,
            seed=args.seed,
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
        clean_reference_by_target[target] = summary["fusion_confidence_mean"]
        all_rows.extend(rows)
        all_summaries.append(summary)

        for degradation in degradations:
            for severity in severities:
                campaign_dir = (
                    campaign_root
                    / target
                    / f"{degradation}_{severity:.2f}".replace(".", "p")
                )
                image_dir, lidar_log = build_campaign_inputs(
                    source_images,
                    source_lidar_log,
                    campaign_dir,
                    degradation=degradation,
                    severity=severity,
                    seed=args.seed,
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
                    clean_reference_confidence=clean_reference_by_target[target],
                )
                all_rows.extend(rows)
                all_summaries.append(summary)

        recovery_dir = campaign_root / target / "recovery_clean"
        image_dir, lidar_log = build_campaign_inputs(
            source_images,
            source_lidar_log,
            recovery_dir,
            degradation="clean",
            severity=0.0,
            seed=args.seed,
        )
        rows, summary = run_pipeline_campaign(
            run_id=f"{target}_recovery",
            image_folder=image_dir,
            lidar_log=lidar_log,
            inference_target=target,
            degradation="recovery_clean",
            severity=0.0,
            dataset_group=args.dataset_group,
            max_samples=args.max_samples,
            clean_reference_confidence=clean_reference_by_target[target],
            recovery_state=recovery_state(
                clean_reference_by_target[target],
                None,
                args.recovery_tolerance,
            ),
        )
        summary["recovery_state"] = recovery_state(
            clean_reference_by_target[target],
            summary["fusion_confidence_mean"],
            args.recovery_tolerance,
        )
        for row in rows:
            row["recovery_state"] = summary["recovery_state"]
        all_rows.extend(rows)
        all_summaries.append(summary)

    summary_payload = summarise_results(all_rows, all_summaries)

    write_csv(output_dir / "resilience_metrics.csv", all_rows)
    write_json(output_dir / "resilience_metrics.json", all_rows)
    write_json(output_dir / "resilience_summary.json", summary_payload)
    write_json(output_dir / "hardware_benchmark_matrix.json", HARDWARE_BENCHMARK_MATRIX)

    print(f"Saved metrics CSV: {output_dir / 'resilience_metrics.csv'}")
    print(f"Saved metrics JSON: {output_dir / 'resilience_metrics.json'}")
    print(f"Saved summary JSON: {output_dir / 'resilience_summary.json'}")
    print(f"Saved hardware matrix: {output_dir / 'hardware_benchmark_matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
