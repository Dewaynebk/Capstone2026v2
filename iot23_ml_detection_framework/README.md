# IoT-23 ML Cybersecurity Detection Framework

This project trains machine learning models to detect anomalous/malicious IoT network traffic using an IoT-23 `conn.log.labeled` file.

## Why this version matters

This is the post-school-friendly version of the project because it is built as a reusable pipeline:

1. Raw IoT-23 Zeek log input
2. Preprocessing into a clean ML dataset
3. Decision Tree and Random Forest model training
4. Model evaluation and analysis
5. Prediction on traffic records
6. Alert logs, summary reports, and HTML dashboard output

## Install dependencies

```bash
pip install -r requirements.txt
```

## Step 1: Preprocess IoT-23 data

```bash
python src/preprocess_iot23.py
```

This creates:

```text
data/processed/iot23_processed.csv
```

## Step 2: Train and evaluate models

```bash
python src/train_models.py
```

This trains:

- Decision Tree
- Random Forest

It creates:

```text
models/selected_model.joblib
outputs/model_analysis.txt
```

## Step 3: Run prediction / detection

```bash
python src/main.py
```

This creates:

```text
outputs/classified_results.csv
outputs/alerts.jsonl
outputs/summary.txt
outputs/dashboard.html
```

## One-command version

You can also run:

```bash
python src/main.py
```

If preprocessing or model training has not happened yet, the system will automatically perform the missing steps.

## What to screenshot

- Terminal output from `python src/train_models.py`
- `outputs/model_analysis.txt`
- Terminal output from `python src/main.py`
- `outputs/dashboard.html`
- `outputs/alerts.jsonl`
- GitHub repository page
