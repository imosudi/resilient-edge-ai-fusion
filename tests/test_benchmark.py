import csv
import json
import tempfile

from metrics.benchmark import (
    BenchmarkCampaign,
    append_metrics_csv,
    compare_baseline_vs_degraded,
    group_rows,
    load_campaign_definitions,
    recovery_summary,
    run_benchmark_campaign,
    save_summary_report,
    summarise_metrics,
)


def test_group_rows_by_keys():
    rows = [
        {"inference_target": "cpu", "degradation": "clean"},
        {"inference_target": "cpu", "degradation": "low_light"},
        {"inference_target": "gpu", "degradation": "clean"},
    ]
    groups = group_rows(rows, ["inference_target", "degradation"])

    assert len(groups) == 3
    assert groups[("cpu", "clean")][0]["degradation"] == "clean"


def test_append_metrics_csv_writes_header_and_rows(tmp_path):
    rows = [
        {"run_id": "run_001", "fusion_confidence": 0.5, "camera_health": "normal"},
        {"run_id": "run_002", "fusion_confidence": 0.4, "camera_health": "degraded"},
    ]
    csv_path = tmp_path / "benchmark.csv"
    append_metrics_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "run_id" in reader.fieldnames
        assert len(list(reader)) == 2


def test_summarise_metrics_computes_means_and_medians():
    rows = [
        {"inference_target": "cpu", "degradation": "clean", "fusion_confidence": 0.8},
        {"inference_target": "cpu", "degradation": "clean", "fusion_confidence": 0.6},
        {"inference_target": "cpu", "degradation": "low_light", "fusion_confidence": 0.4},
    ]
    report = summarise_metrics(rows, group_by=["inference_target", "degradation"], metrics=["fusion_confidence"])

    assert any(item["fusion_confidence_mean"] == 0.7 for item in report)
    assert any(item["fusion_confidence_median"] == 0.7 for item in report)


def test_compare_baseline_vs_degraded_computes_deltas():
    rows = [
        {
            "dataset_group": "vision_lidar",
            "inference_target": "cpu",
            "degradation": "clean",
            "fusion_confidence": 0.8,
        },
        {
            "dataset_group": "vision_lidar",
            "inference_target": "cpu",
            "degradation": "low_light",
            "severity": 0.3,
            "fusion_confidence": 0.5,
        },
    ]
    comparisons = compare_baseline_vs_degraded(rows)

    assert comparisons[0]["fusion_mean_delta"] == -0.30000000000000004
    assert comparisons[0]["robustness_score"] == 62.5


def test_recovery_summary_counts_recovery_events():
    rows = [
        {"fallback_state": "lidar_assisted", "camera_health": "degraded", "lidar_health": "normal", "recovery_state": False},
        {"fallback_state": "normal", "camera_health": "normal", "lidar_health": "normal", "recovery_state": True},
    ]
    summary = recovery_summary(rows)

    assert summary["total_samples"] == 2
    assert summary["recovery_events"] == 1
    assert summary["recovery_rate"] == 100.0


def test_load_campaign_definitions_from_json(tmp_path):
    config = [
        {
            "name": "clean_baseline",
            "degradation": "clean",
            "severity": 0.0,
            "image_folder": "dataset/images",
            "lidar_log": "dataset/lidar.log",
            "dataset_group": "vision_lidar",
            "mode": "offline",
            "inference_targets": ["cpu", "gpu"],
            "max_samples": 1,
        }
    ]
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    campaigns = load_campaign_definitions(config_path)
    assert len(campaigns) == 1
    assert campaigns[0].name == "clean_baseline"
    assert campaigns[0].targets() == ["cpu", "gpu"]


def test_save_summary_report_writes_json(tmp_path):
    report = {"total_samples": 0}
    report_path = tmp_path / "summary.json"
    save_summary_report(report, report_path)
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
