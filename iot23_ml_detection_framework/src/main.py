"""Main entry point for IoT-23 ML detection framework.

This version is demo-safe:
- It accepts --input from the command line.
- It uses the sample dataset in data/sample by default.
- It does NOT force preprocessing of the full IoT-23 raw file.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, ROOT
from predict import classify, write_dashboard


DEFAULT_SAMPLE_INPUT = ROOT / "data" / "sample" / "sample_iot23_dataset.csv"


def main():
    parser = argparse.ArgumentParser(description="Run IoT-23 ML Detection Framework")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_SAMPLE_INPUT),
        help="Path to input CSV file. Default uses the GitHub-safe IoT-23 sample dataset."
    )
    args = parser.parse_args()

    print("Starting IoT-23 ML Detection Framework...\n")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "For the GitHub demo, use: data/sample/sample_iot23_dataset.csv"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    results = classify(input_path)

    results_path = OUTPUT_DIR / "classified_results.csv"
    alerts_path = OUTPUT_DIR / "alerts.jsonl"
    summary_path = OUTPUT_DIR / "summary.txt"
    dashboard_path = OUTPUT_DIR / "dashboard.html"

    results.to_csv(results_path, index=False)

    alerts = results[results["prediction"] == "malicious"]
    with open(alerts_path, "w", encoding="utf-8") as f:
        for _, row in alerts.iterrows():
            alert = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source_ip": row.get("source_ip", ""),
                "destination_ip": row.get("destination_ip", ""),
                "protocol": row.get("protocol", ""),
                "service": row.get("service", ""),
                "destination_port": int(row.get("destination_port", 0)),
                "prediction": row["prediction"],
                "confidence": float(row["confidence"]),
            }
            f.write(json.dumps(alert) + "\n")

    total = len(results)
    benign = int((results["prediction"] == "benign").sum())
    malicious = int((results["prediction"] == "malicious").sum())

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("IoT-23 ML Threat Detection Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Total records classified: {total}\n")
        f.write(f"Benign predictions: {benign}\n")
        f.write(f"Malicious predictions: {malicious}\n")
        f.write(f"Results file: {results_path}\n")
        f.write(f"Alerts file: {alerts_path}\n")
        f.write(f"Dashboard: {dashboard_path}\n")

    write_dashboard(results, dashboard_path)

    print("Prediction complete.\n")
    print(f"Input file: {input_path}")
    print(f"Total records classified: {total}")
    print(f"Benign predictions: {benign}")
    print(f"Malicious predictions: {malicious}")
    print(f"\nResults saved to: {results_path}")
    print(f"Alerts saved to: {alerts_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main()
