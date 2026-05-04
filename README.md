# Churn prediction coursework

Two related assignments: baseline streaming-style churn modeling, then comparative models on Telco churn data.

## Layout

| Folder | Scripts / data |
|--------|------------------|
| `sprint1_baseline/` | `m1_baseline_model.py`, `stream_pulse_customer_data.csv` — scikit-learn pipeline + grid search + `joblib` exports. |
| `sprint3_advanced/` | `m2_advanced_classification.py`, `WA_Fn-UseC_-Telco-Customer-Churn.csv` — multiple models, tuned logistic regression, confusion matrix figure. |

Telco dataset: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

Always `cd` into the sprint folder that contains both the script and its CSV before running.

### Sprint 1 — baseline

```bash
cd sprint1_baseline
python m1_baseline_model.py
```

Produces `baseline_model.pkl` and `preprocessing_pipeline.pkl` in that folder (ignored by `.gitignore` unless you force-add).

### Sprint 3 — advanced

```bash
cd sprint3_advanced
python m2_advanced_classification.py
```

Produces `optimized_churn_model.pkl` and `confusion_matrix.png` in that folder (also gitignored).

Sprints use different schemas and CSVs on purpose — they are independent runs, not a shared training pipeline.
