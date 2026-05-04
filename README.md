# Customer churn prediction

Baseline subscription churn modeling alongside a Telco benchmark with logistic regression and related classifiers—grid search for pipeline tuning, model comparison on holdout metrics, and confusion-matrix reporting.

<p><img src="https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/><img src="https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/><img src="https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/><img src="https://img.shields.io/badge/SCIKITLEARN-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/><img src="https://img.shields.io/badge/MATPLOTLIB-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/><img src="https://img.shields.io/badge/SEABORN-9C554A?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn"/><img src="https://img.shields.io/badge/JOBLIB-EF8C06?style=for-the-badge" alt="Joblib"/></p>

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
