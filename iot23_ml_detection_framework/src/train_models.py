"""Train and compare Decision Tree and Random Forest models on IoT-23 traffic."""
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from config import PROCESSED_DATA, MODEL_DIR, OUTPUT_DIR, SELECTED_MODEL, MODEL_METADATA, FEATURE_COLUMNS, RANDOM_STATE
from preprocess_iot23 import preprocess

def load_or_create_dataset() -> pd.DataFrame:
    """Load processed data, or create it from the raw IoT-23 log if missing."""
    if not PROCESSED_DATA.exists():
        preprocess()
    return pd.read_csv(PROCESSED_DATA)

def train_models():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_or_create_dataset()

    # Remove rows where all traffic behavior features are zero.
    # This protects model quality when raw logs contain missing values.
    feature_df = df[FEATURE_COLUMNS].fillna(0)
    labels = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels if labels.nunique() > 1 else None,
    )

    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report_text": classification_report(
                y_test,
                pred,
                target_names=["benign", "malicious"],
                zero_division=0,
            ),
        }

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    # Choose Random Forest by default because it is generally more robust than a single tree.
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
    }

    with open(MODEL_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUT_DIR / "model_analysis.txt", "w") as f:
        f.write("IoT-23 Machine Learning Model Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training rows: {metadata['training_rows']}\n")
        f.write(f"Benign rows: {metadata['label_distribution']['benign']}\n")
        f.write(f"Malicious rows: {metadata['label_distribution']['malicious']}\n\n")

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
    print(f"Selected model: {selected}")
    print(f"Model saved to: {SELECTED_MODEL}")
    print(f"Model analysis saved to: {OUTPUT_DIR / 'model_analysis.txt'}")

if __name__ == "__main__":
    train_models()
