"""Classify IoT traffic records using the trained IoT-23 ML model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from config import FEATURE_COLUMNS, OUTPUT_DIR, SELECTED_MODEL, PROCESSED_DATA
from preprocess_iot23 import preprocess
from train_models import train_models


def ensure_ready():
    """Ensure processed data and trained model exist."""
    if not PROCESSED_DATA.exists():
        preprocess()
    if not SELECTED_MODEL.exists():
        train_models()


def prepare_input(path: Path) -> pd.DataFrame:
    """Load either a processed CSV or raw Zeek conn.log.labeled file."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        from preprocess_iot23 import preprocess as preprocess_file
        df = preprocess_file(path)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def classify(path: Path) -> pd.DataFrame:
    """Classify traffic records as benign or malicious."""
    ensure_ready()
    model = joblib.load(SELECTED_MODEL)
    df = prepare_input(path)

    predictions = model.predict(df[FEATURE_COLUMNS])

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(df[FEATURE_COLUMNS]).max(axis=1)
    else:
        confidence = [1.0] * len(df)

    df["prediction"] = ["malicious" if x == 1 else "benign" for x in predictions]
    df["confidence"] = [round(float(x), 4) for x in confidence]

    return df


def write_dashboard(df: pd.DataFrame, dashboard_path: Path):
    """Generate a simple HTML dashboard for classified traffic results."""
    total = len(df)
    malicious = int((df["prediction"] == "malicious").sum())
    benign = int((df["prediction"] == "benign").sum())
    alerts = df[df["prediction"] == "malicious"].head(100)

    cols = [
        c for c in [
            "source_ip", "destination_ip", "protocol", "service", "destination_port",
            "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
            "prediction", "confidence"
        ]
        if c in df.columns
    ]

    all_table = df[cols].head(200).to_html(index=False)
    alert_table = (
        alerts[cols].to_html(index=False)
        if len(alerts)
        else "<p>No malicious traffic detected.</p>"
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>IoT-23 ML Detection Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; background:#f4f7fb; padding:22px; color:#111827; }}
h1 {{ margin-bottom:4px; }}
.card {{ display:inline-block; background:white; padding:16px 22px; margin:8px; border-radius:10px; box-shadow:0 2px 8px #ccd; }}
.num {{ font-size:30px; font-weight:bold; }}
.bad {{ color:#b91c1c; }}
.good {{ color:#047857; }}
table {{ border-collapse:collapse; background:white; width:100%; margin-top:12px; }}
th, td {{ border:1px solid #d1d5db; padding:7px; font-size:12px; }}
th {{ background:#111827; color:white; }}
</style>
</head>
<body>
<h1>IoT-23 ML Threat Detection Dashboard</h1>
<p>Results generated from a trained Random Forest model using IoT-23 network traffic features.</p>
<div class="card">Total Records<br><span class="num">{total}</span></div>
<div class="card">Benign Predictions<br><span class="num good">{benign}</span></div>
<div class="card">Malicious Predictions<br><span class="num bad">{malicious}</span></div>
<h2>Generated Alerts</h2>
{alert_table}
<h2>Classified Traffic Sample</h2>
{all_table}
</body>
</html>"""
    dashboard_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run IoT-23 ML traffic classification")
    parser.add_argument(
        "--input",
        default=str(PROCESSED_DATA),
        help="Processed CSV or Zeek conn.log.labeled file"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

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
        f.write(f"Dashboard: {dashboard_path}\n")

    write_dashboard(results, dashboard_path)

    print("Prediction complete.")
    print(f"Input file: {input_path}")
    print(f"Total records classified: {total}")
    print(f"Benign predictions: {benign}")
    print(f"Malicious predictions: {malicious}")
    print(f"Results: {results_path}")
    print(f"Alerts: {alerts_path}")
    print(f"Summary: {summary_path}")
    print(f"Dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()
