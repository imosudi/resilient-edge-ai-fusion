"""metrics/benchmark.py

Benchmark orchestration, structured metric capture, and summary reporting for
resilient Vision-LiDAR campaigns.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metrics import build_metric_record
from metrics.robustness_score import robustness_score


@dataclass(frozen=True)
class BenchmarkCampaign:
    name: str
    degradation: str
    severity: float
    image_folder: str | None = None
    lidar_log: str | None = None
    dataset_group: str = "vision_lidar"
    mode: str = "offline"
    inference_targets: list[str] | None = None
    max_samples: int = 50
    run_id: str | None = None

    def targets(self) -> list[str]:
        return self.inference_targets or ["cpu", "gpu", "npu"]


def _build_run_id(campaign: BenchmarkCampaign, target: str) -> str:
    timestamp = int(time.time())
    severity_label = f"{int(campaign.severity * 100):02d}"
    return campaign.run_id or (
        f"benchmark_{campaign.dataset_group}_{campaign.name}_{campaign.degradation}_"
        f"{severity_label}_{target}_{timestamp}"
    )


def _annotate_sample(
    sample: dict[str, Any],
    campaign: BenchmarkCampaign,
    target: str,
    run_id: str,
    sample_index: int,
) -> dict[str, Any]:
    sample.update(
        {
            "run_id": run_id,
            "dataset_group": campaign.dataset_group,
            "campaign_name": campaign.name,
            "sample_index": sample_index,
            "degradation": campaign.degradation,
            "severity": campaign.severity,
            "inference_target": target,
        }
    )
    return sample


def _add_recovery_states(samples: list[dict[str, Any]]) -> None:
    previous_degraded = False
    for sample in samples:
        currently_normal = (
            sample.get("camera_health") == "normal"
            and sample.get("lidar_health") == "normal"
            and sample.get("fallback_state") == "normal"
        )
        sample["recovery_state"] = bool(previous_degraded and currently_normal)
        previous_degraded = (
            sample.get("camera_health") != "normal"
            or sample.get("lidar_health") != "normal"
            or sample.get("fallback_state") != "normal"
        )


def append_metrics_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_fieldnames != fieldnames:
            fieldnames = list(dict.fromkeys(existing_fieldnames + fieldnames))
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({key: existing_row.get(key) for key in fieldnames})

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row.get(key) for key in keys)
        groups.setdefault(group_key, []).append(row)
    return groups


def summarise_metrics(
    rows: list[dict[str, Any]],
    group_by: list[str] | None = None,
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    group_by = group_by or ["inference_target", "degradation", "severity"]
    metrics = metrics or [
        "latency_ms",
        "fps",
        "cpu_percent",
        "memory_mb",
        "temperature_c",
        "camera_confidence",
        "lidar_confidence",
        "fusion_confidence",
        "robustness_score",
    ]

    groups = group_rows(rows, group_by)
    summaries: list[dict[str, Any]] = []
    for key, group in groups.items():
        summary = dict(zip(group_by, key))
        for metric in metrics:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            if not values:
                continue
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_median"] = statistics.median(values)
            if len(values) > 1:
                summary[f"{metric}_stdev"] = statistics.stdev(values)
        summary["sample_count"] = len(group)
        summaries.append(summary)
    return summaries


def compare_baseline_vs_degraded(
    rows: list[dict[str, Any]],
    baseline_degradation: str = "clean",
) -> list[dict[str, Any]]:
    degraded_rows = [row for row in rows if row.get("degradation") != baseline_degradation]
    baseline_rows = [row for row in rows if row.get("degradation") == baseline_degradation]

    if not baseline_rows or not degraded_rows:
        return []

    baseline_groups = group_rows(baseline_rows, ["dataset_group", "inference_target"])
    degraded_groups = group_rows(degraded_rows, ["dataset_group", "inference_target", "degradation", "severity"])

    comparisons: list[dict[str, Any]] = []
    for key, degraded_group in degraded_groups.items():
        dataset_group, target, degradation, severity = key
        baseline_key = (dataset_group, target)
        baseline_group = baseline_groups.get(baseline_key)
        if not baseline_group:
            continue

        baseline_fusion = [float(item["fusion_confidence"]) for item in baseline_group if item.get("fusion_confidence") is not None]
        degraded_fusion = [float(item["fusion_confidence"]) for item in degraded_group if item.get("fusion_confidence") is not None]
        if not baseline_fusion or not degraded_fusion:
            continue

        baseline_mean = statistics.mean(baseline_fusion)
        degraded_mean = statistics.mean(degraded_fusion)
        comparison = {
            "dataset_group": dataset_group,
            "inference_target": target,
            "baseline_degradation": baseline_degradation,
            "degradation": degradation,
            "severity": severity,
            "baseline_fusion_mean": baseline_mean,
            "degraded_fusion_mean": degraded_mean,
            "fusion_mean_delta": degraded_mean - baseline_mean,
            "robustness_score": robustness_score(baseline_mean, degraded_mean),
            "baseline_sample_count": len(baseline_group),
            "degraded_sample_count": len(degraded_group),
        }
        comparisons.append(comparison)
    return comparisons


def recovery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovery_events = [row for row in rows if row.get("recovery_state")]
    degraded_events = [row for row in rows if row.get("fallback_state") != "normal" or row.get("camera_health") != "normal" or row.get("lidar_health") != "normal"]
    return {
        "total_samples": len(rows),
        "recovery_events": len(recovery_events),
        "degraded_samples": len(degraded_events),
        "recovery_rate": (len(recovery_events) / len(degraded_events) * 100) if degraded_events else 0.0,
    }


def load_campaign_definitions(path: str | Path) -> list[BenchmarkCampaign]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    campaigns: list[BenchmarkCampaign] = []
    for item in data:
        campaigns.append(BenchmarkCampaign(
            name=item["name"],
            degradation=item["degradation"],
            severity=float(item.get("severity", 0.0)),
            image_folder=item.get("image_folder"),
            lidar_log=item.get("lidar_log"),
            dataset_group=item.get("dataset_group", "vision_lidar"),
            mode=item.get("mode", "offline"),
            inference_targets=item.get("inference_targets"),
            max_samples=int(item.get("max_samples", 50)),
            run_id=item.get("run_id"),
        ))
    return campaigns


def run_benchmark_campaign(
    campaigns: list[BenchmarkCampaign],
    csv_path: str | Path = "metrics/benchmark_results.csv",
    append: bool = True,
) -> list[dict[str, Any]]:
    from fusion.pipeline import FusionPipeline

    all_results: list[dict[str, Any]] = []
    for campaign in campaigns:
        for target in campaign.targets():
            run_id = _build_run_id(campaign, target)
            if campaign.mode == "offline" and not campaign.image_folder:
                raise ValueError("Offline campaigns require image_folder")
            if campaign.mode == "offline" and not campaign.lidar_log:
                raise ValueError("Offline campaigns require lidar_log")

            pipeline = FusionPipeline(
                camera_source="folder" if campaign.mode == "offline" else "camera",
                image_folder=campaign.image_folder,
                lidar_log=campaign.lidar_log,
                inference_target=target,
                degradation=campaign.degradation,
                severity=campaign.severity,
                run_id=run_id,
            )
            if campaign.mode == "offline":
                results = pipeline.run_offline(max_samples=campaign.max_samples)
            else:
                results = pipeline.run_live(max_samples=campaign.max_samples)

            processed = [
                _annotate_sample(result, campaign, target, run_id, index + 1)
                for index, result in enumerate(results)
            ]
            _add_recovery_states(processed)
            append_metrics_csv(processed, csv_path)
            all_results.extend(processed)
    return all_results


def save_summary_report(report: list[dict[str, Any]], report_path: str | Path) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
