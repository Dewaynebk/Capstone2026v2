"""Preprocess IoT-23 Zeek conn.log.labeled files into clean ML-ready CSV.

IoT-23 connection logs are Zeek TSV-style files. They usually include metadata
lines beginning with # and field names in a line beginning with #fields.
This script extracts useful numeric features and converts textual labels into:
    0 = benign
    1 = malicious
"""
from pathlib import Path
import pandas as pd

from config import RAW_LOG, PROCESSED_DATA

NUMERIC_COLUMNS = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

def read_zeek_conn_log(path: Path) -> pd.DataFrame:
    """Read a Zeek-style conn.log.labeled file into a pandas DataFrame."""
    fields = None
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                continue

            if line.startswith("#"):
                continue

            if fields is None:
                raise ValueError("Could not find #fields line in Zeek log.")

            values = line.split("\t")
            if len(values) < len(fields):
                values += ["-"] * (len(fields) - len(values))
            elif len(values) > len(fields):
                values = values[:len(fields)]

            rows.append(values)

    return pd.DataFrame(rows, columns=fields)

def normalize_label(value: str) -> int:
    """Convert IoT-23 textual labels to binary class labels."""
    text = str(value).lower()
    if "benign" in text:
        return 0
    return 1

def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert Zeek missing values (-) to 0 and coerce to numeric."""
    return pd.to_numeric(series.replace("-", 0), errors="coerce").fillna(0)

def preprocess(raw_path: Path = RAW_LOG, output_path: Path = PROCESSED_DATA) -> pd.DataFrame:
    """Preprocess the raw IoT-23 log and write a clean CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_zeek_conn_log(raw_path)

    # Locate label column. IoT-23 labeled logs commonly use 'label'.
    if "label" not in df.columns:
        possible = [c for c in df.columns if "label" in c.lower()]
        if not possible:
            raise ValueError("No label column found in uploaded IoT-23 log.")
        label_col = possible[0]
    else:
        label_col = "label"

    # Ensure required numeric columns exist. If missing, create them as zero.
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = clean_numeric(df[col])

    processed = pd.DataFrame()
    processed["uid"] = df["uid"] if "uid" in df.columns else range(len(df))
    processed["source_ip"] = df["id.orig_h"] if "id.orig_h" in df.columns else ""
    processed["destination_ip"] = df["id.resp_h"] if "id.resp_h" in df.columns else ""
    processed["protocol"] = df["proto"] if "proto" in df.columns else ""
    processed["service"] = df["service"] if "service" in df.columns else ""
    processed["destination_port"] = clean_numeric(df["id.resp_p"]) if "id.resp_p" in df.columns else 0

    processed["duration"] = df["duration"]
    processed["orig_bytes"] = df["orig_bytes"]
    processed["resp_bytes"] = df["resp_bytes"]
    processed["orig_pkts"] = df["orig_pkts"]
    processed["resp_pkts"] = df["resp_pkts"]
    processed["total_bytes"] = processed["orig_bytes"] + processed["resp_bytes"]
    processed["total_pkts"] = processed["orig_pkts"] + processed["resp_pkts"]
    processed["bytes_per_packet"] = processed["total_bytes"] / processed["total_pkts"].replace(0, 1)

    processed["label"] = df[label_col].apply(normalize_label)
    processed["label_text"] = df[label_col]

    processed.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to: {output_path}")
    print(f"Rows: {len(processed)}")
    print(processed["label"].value_counts().rename({0: "benign", 1: "malicious"}))
    return processed

if __name__ == "__main__":
    preprocess()
