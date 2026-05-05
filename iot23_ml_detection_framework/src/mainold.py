"""Main entry point for IoT-23 ML detection framework"""

from pathlib import Path
from predict import classify, write_dashboard
from config import OUTPUT_DIR, PROCESSED_DATA

def main():
    print("Starting IoT-23 ML Detection Framework...\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    input_path = Path(PROCESSED_DATA)

    results = classify(input_path)

    results_path = OUTPUT_DIR / "classified_results.csv"
    alerts_path = OUTPUT_DIR / "alerts.jsonl"
    summary_path = OUTPUT_DIR / "summary.txt"
    dashboard_path = OUTPUT_DIR / "dashboard.html"

    results.to_csv(results_path, index=False)

    alerts = results[results["prediction"] == "malicious"]
    with open(alerts_path, "w", encoding="utf-8") as f:
        for _, row in alerts.iterrows():
            f.write(f"{row.to_dict()}\n")

    total = len(results)
    benign = int((results["prediction"] == "benign").sum())
    malicious = int((results["prediction"] == "malicious").sum())

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("IoT-23 ML Threat Detection Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total records: {total}\n")
        f.write(f"Benign: {benign}\n")
        f.write(f"Malicious: {malicious}\n")

    write_dashboard(results, dashboard_path)

    print("Prediction complete.\n")
    print(f"Total records classified: {total}")
    print(f"Benign predictions: {benign}")
    print(f"Malicious predictions: {malicious}")
    print(f"\nResults saved to: {results_path}")
    print(f"Alerts saved to: {alerts_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Dashboard saved to: {dashboard_path}")

if __name__ == "__main__":
    main()
