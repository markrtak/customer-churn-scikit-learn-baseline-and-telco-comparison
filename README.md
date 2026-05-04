# Customer churn prediction

Baseline subscription churn modeling alongside a Telco benchmark with logistic regression and related classifiers—grid search for pipeline tuning, model comparison on holdout metrics, and confusion-matrix reporting.

<p align="left">
  <img src="https://skillicons.dev/icons?i=py,pandas,numpy,sklearn,matplotlib,seaborn" height="48" alt="Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn" />
</p>

[![Joblib](https://img.shields.io/badge/Joblib-EF8C06?style=flat&logo=python&logoColor=white&labelColor=333)](https://joblib.readthedocs.io)

Repository: [markrtak/customer-churn-scikit-learn-baseline-and-telco-comparison](https://github.com/markrtak/customer-churn-scikit-learn-baseline-and-telco-comparison)

## Layout

| Folder | Contents |
|--------|----------|
| `baseline_subscription/` | `m1_baseline_model.py`, `stream_pulse_customer_data.csv` — preprocessing pipeline + `GridSearchCV` + `joblib` exports |
| `telco_benchmark/` | `m2_advanced_classification.py`, `WA_Fn-UseC_-Telco-Customer-Churn.csv` — model comparison + tuned logistic regression + confusion matrix figure |

Telco dataset: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

Run each workflow from its own directory (alongside its CSV).

### Baseline (subscription churn)

```bash
cd baseline_subscription
python m1_baseline_model.py
```

Writes `baseline_model.pkl` and `preprocessing_pipeline.pkl` there (these are `.gitignored` unless you publish artifacts intentionally).

### Telco benchmark

```bash
cd telco_benchmark
python m2_advanced_classification.py
```

Writes `optimized_churn_model.pkl` and `confusion_matrix.png` there (also `.gitignored`).

The two tracks use **different schemas and CSVs**; treat them as separate experiments, not a single shared training pipeline.
