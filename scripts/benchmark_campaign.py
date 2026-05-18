import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics.benchmark import (
    BenchmarkCampaign,
    load_campaign_definitions,
    run_benchmark_campaign,
    save_summary_report,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("scripts.benchmark_campaign")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a reproducible resilient benchmark campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--campaign-file",
        help="Path to a JSON campaign file defining degradation and target runs.",
    )
    parser.add_argument(
        "--csv-path",
        default="metrics/benchmark_results.csv",
        help="Path to save benchmark metric records.",
    )
    parser.add_argument(
        "--report-path",
        default="metrics/benchmark_summary.json",
        help="Path to save aggregated benchmark report.",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print the generated summary to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.campaign_file:
        raise SystemExit("A campaign file is required. Use --campaign-file <path>.")

    campaign_definitions = load_campaign_definitions(args.campaign_file)
    results = run_benchmark_campaign(campaign_definitions, csv_path=args.csv_path)

    report = {
        "total_samples": len(results),
        "campaigns": [
            {
                "name": campaign.name,
                "degradation": campaign.degradation,
                "severity": campaign.severity,
                "targets": campaign.targets(),
                "max_samples": campaign.max_samples,
            }
            for campaign in campaign_definitions
        ],
    }
    save_summary_report(report, args.report_path)

    if args.show_summary:
        print(json.dumps(report, indent=2))

    LOGGER.info("Benchmark campaign completed: %d rows written", len(results))
    LOGGER.info("Metrics: %s", args.csv_path)
    LOGGER.info("Summary: %s", args.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
