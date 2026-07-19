# Credit Risk / Loan Default Prediction with Explainable AI

Predicting whether a loan applicant will experience serious delinquency within 2 years, using machine learning — with SHAP explanations so every prediction can be justified, not just delivered as a black-box score.

**https://credit-risk-prediction-system-11.streamlit.app/**  **[Live App Demo](#)**

## Problem Statement

Lenders need to assess the risk of loan default before approving credit. A model that only outputs "approve/reject" isn't enough in regulated lending — decisions must be explainable. This project builds a classification model to predict default risk and pairs it with SHAP (SHapley Additive exPlanations) to show *why* each prediction was made.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) (Kaggle) — 150,000 loan applicants with 10 features including revolving credit utilization, age, debt ratio, monthly income, credit lines, and payment delinquency history. Target: `SeriousDlqin2yrs` (1 = defaulted within 2 years).

## Key Findings from EDA

- **Severe class imbalance**: 93.3% non-default vs. 6.7% default — addressed with SMOTE oversampling on the training set only (never the test set, to avoid data leakage)
- **Data quality issues found and fixed**: a single `age = 0` row (impossible value), and 269 rows with placeholder values of 96/98 in the delinquency columns (not genuine payment counts — flagged and capped rather than used literally)
- **Age is strongly predictive**: default rate drops from ~11% (age 18-25) to ~2.4% (age 65+), a clean, business-intuitive pattern
- **Multicollinearity**: the three "days past due" columns correlate at 0.99+ with each other, informing feature engineering decisions

## Approach

1. **EDA** — identified imbalance, anomalies, and feature relationships
2. **Cleaning & Feature Engineering** — fixed anomalies, imputed missing values (median income grouped by credit lines, mode for dependents), engineered 5 new features (`TotalPastDueIncidents`, `EstimatedMonthlyDebtPayment`, `IncomePerDependent`, `CreditLinesPerAgeYear`, `HasRealEstateLoan`)
3. **Class Imbalance Handling** — SMOTE applied after train/test split
4. **Modeling** — compared Logistic Regression, Random Forest, and XGBoost
5. **Evaluation** — precision, recall, F1, ROC-AUC, and confusion matrices (not accuracy alone, since it's misleading under imbalance)
6. **Explainability** — SHAP global feature importance and per-applicant local explanations
7. **Deployment** — interactive Streamlit app for live predictions with explanations

## Results

| Model | ROC-AUC | Default Recall |
|---|---|---|
| Logistic Regression | ~0.845 | 0.70 |
| Random Forest | ~0.858 | 0.76 |
| XGBoost | ~0.832 | 0.56 |

Random Forest achieved the best ROC-AUC; the deployed app uses XGBoost for its fast, exact SHAP compatibility via `TreeExplainer`. Model choice in production would depend on whether the business prioritizes catching more defaulters (recall) or precision.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · Streamlit · Matplotlib · Seaborn

## Project Structure

```
├── credit_risk_project.ipynb   # Full analysis notebook (EDA → modeling → SHAP)
├── streamlit_app/
│   ├── app.py                  # Interactive prediction app
│   ├── model.json              # Trained XGBoost model
│   ├── feature_columns.json    # Expected feature order
│   ├── defaults.json           # Default form values
│   └── requirements.txt        # Dependencies
└── README.md
```

## Running Locally

```bash
# Clone the repo
git clone <your-repo-url>
cd credit-risk-prediction-system

# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run the app
cd streamlit_app
streamlit run app.py
```

## What This Project Demonstrates

- Handling real-world messy data (missing values, placeholder anomalies, outliers)
- Correctly sequencing train/test split before class imbalance handling (avoiding data leakage)
- Choosing business-relevant evaluation metrics over misleading accuracy
- Explainable AI for regulated, high-stakes predictions
- End-to-end deployment, not just a notebook
