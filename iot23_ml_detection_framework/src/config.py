"""Central configuration for the IoT-23 ML detection framework."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_LOG = ROOT / "data" / "raw" / "conn.log.labeled"
PROCESSED_DATA = ROOT / "data" / "processed" / "iot23_processed.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

SELECTED_MODEL = MODEL_DIR / "selected_model.joblib"
MODEL_METADATA = MODEL_DIR / "model_metadata.json"

FEATURE_COLUMNS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "total_bytes",
    "total_pkts",
    "bytes_per_packet",
]

RANDOM_STATE = 42
