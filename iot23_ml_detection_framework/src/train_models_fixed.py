"""Train and compare Decision Tree and Random Forest models.

Demo-safe version:
- Works even if the sample dataset only contains one class.
- Keeps labels fixed as 0 = benign and 1 = malicious for reports.
- Saves selected_model.joblib for prediction.
"""

import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    PROCESSED_DATA,
    MODEL_DIR,
    OUTPUT_DIR,
    SELECTED_MODEL,
    MODEL_METADATA,
    FEATURE_COLUMNS,
    RANDOM_STATE,
)


def load_dataset() -> pd.DataFrame:
    """Load the processed dataset used for training."""
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA}\n"
            "For the GitHub demo, first copy the sample dataset:\n"
            "copy data\\sample\\sample_iot23_dataset.csv data\\processed\\iot23_processed.csv"
        )

    df = pd.read_csv(PROCESSED_DATA)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "label" not in df.columns:
        raise ValueError(
            "The training dataset must contain a 'label' column where 0 = benign and 1 = malicious."
        )

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    return df


def safe_split(X, y):
    """Split data safely even when a small sample has only one class."""
    if y.nunique() > 1 and len(y) >= 4:
        return train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    # Single-class fallback for demo datasets.
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=None,
    )


def evaluate_model(model, X_test, y_test):
    """Evaluate model while forcing report labels to include benign and malicious."""
    pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, labels=[0, 1], average="binary", zero_division=0)),
        "recall": float(recall_score(y_test, pred, labels=[0, 1], average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, labels=[0, 1], average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report_text": classification_report(
            y_test,
            pred,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            zero_division=0,
        ),
    }


def train_models():
    """Train Decision Tree and Random Forest models."""
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = safe_split(X, y)

    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced" if y.nunique() > 1 else None,
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    selected = "random_forest"
    joblib.dump(models[selected], SELECTED_MODEL)

    metadata = {
        "selected_model": selected,
        "features": FEATURE_COLUMNS,
        "training_rows": int(len(df)),
        "label_distribution": {
            "benign": int((df["label"] == 0).sum()),
            "malicious": int((df["label"] == 1).sum()),
        },
        "results": results,
        "note": (
            "If this demo dataset contains only one class, model reports will show that limitation. "
            "The full IoT-23 dataset should be used for stronger evaluation."
        ),
    }

    with open(MODEL_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    analysis_path = OUTPUT_DIR / "model_analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("IoT-23 Machine Learning Model Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training rows: {metadata['training_rows']}\n")
        f.write(f"Benign rows: {metadata['label_distribution']['benign']}\n")
        f.write(f"Malicious rows: {metadata['label_distribution']['malicious']}\n\n")

        if y.nunique() == 1:
            only_class = "benign" if int(y.iloc[0]) == 0 else "malicious"
            f.write("NOTE: This sample training dataset contains only one class: ")
            f.write(f"{only_class}.\n")
            f.write("This is acceptable for demonstrating the pipeline, but a balanced or fuller IoT-23 subset ")
            f.write("is recommended for stronger model evaluation.\n\n")

        for name, result in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            f.write(f"Confusion Matrix [[TN, FP], [FN, TP]]: {result['confusion_matrix']}\n")
            f.write(result["classification_report_text"])
            f.write("\n" + "-" * 60 + "\n\n")

        f.write(f"Selected model for prediction: {selected}\n")

    print("Training complete.")
    print(f"Training rows: {len(df)}")
    print(f"Benign rows: {int((df['label'] == 0).sum())}")
    print(f"Malicious rows: {int((df['label'] == 1).sum())}")
    print(f"Selected model: {selected}")
    print(f"Model saved to: {SELECTED_MODEL}")
    print(f"Model analysis saved to: {analysis_path}")


if __name__ == "__main__":
    train_models()
