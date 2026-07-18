"""
Credit Risk / Loan Default Prediction App with Explainable AI (SHAP)
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

# Get the folder this script lives in, so file paths work regardless of
# what directory the app is launched FROM (fixes deployment path issues)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== Load model and artifacts (cached so it only loads once) =====
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(APP_DIR, "model.json"))

    with open(os.path.join(APP_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    with open(os.path.join(APP_DIR, "defaults.json")) as f:
        defaults = json.load(f)

    explainer = shap.TreeExplainer(model)
    return model, feature_columns, defaults, explainer

model, feature_columns, defaults, explainer = load_artifacts()

st.title("💳 Credit Risk / Loan Default Prediction")
st.markdown(
    "Enter an applicant's details below to predict their risk of **serious delinquency "
    "in the next 2 years**, along with an explanation of *why* the model made that prediction."
)

st.divider()

# ===== Input Form =====
st.subheader("Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=defaults["age_median"])
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0, value=int(defaults["MonthlyIncome_median"]), step=100
    )
    debt_ratio = st.number_input(
        "Debt Ratio (monthly debt payments / income)", min_value=0.0, max_value=5.0, value=0.3, step=0.01
    )

with col2:
    revolving_util = st.number_input(
        "Revolving Utilization of Unsecured Lines (0-2, e.g. 0.5 = 50%)",
        min_value=0.0, max_value=2.0, value=0.3, step=0.01
    )
    open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", min_value=0, max_value=50, value=8)
    real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, max_value=20, value=1)

with col3:
    late_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, max_value=20, value=0)
    late_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, max_value=20, value=0)
    late_90_plus = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)

dependents = st.slider("Number of Dependents", min_value=0, max_value=10, value=defaults["NumberOfDependents_mode"])

st.divider()

# ===== Predict Button =====
if st.button("🔍 Predict Credit Risk", type="primary", use_container_width=True):

    # ===== Build feature row exactly matching training pipeline =====
    total_past_due = late_30_59 + late_60_89 + late_90_plus
    estimated_monthly_debt = debt_ratio * monthly_income
    income_per_dependent = monthly_income / (dependents + 1)
    credit_lines_per_age_year = open_credit_lines / (age - 17) if age > 17 else 0
    has_real_estate_loan = 1 if real_estate_loans > 0 else 0

    input_dict = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90_plus,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents,
        "NumberOfTime30-59DaysPastDueNotWorse_was_anomaly": 0,
        "NumberOfTimes90DaysLate_was_anomaly": 0,
        "NumberOfTime60-89DaysPastDueNotWorse_was_anomaly": 0,
        "MonthlyIncome_was_missing": 0,
        "TotalPastDueIncidents": total_past_due,
        "EstimatedMonthlyDebtPayment": estimated_monthly_debt,
        "IncomePerDependent": income_per_dependent,
        "CreditLinesPerAgeYear": credit_lines_per_age_year,
        "HasRealEstateLoan": has_real_estate_loan,
    }

    input_df = pd.DataFrame([input_dict])[feature_columns]

    # ===== Predict =====
    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        if prediction == 1:
            st.error(f"### ⚠️ High Risk of Default")
        else:
            st.success(f"### ✅ Low Risk of Default")
        st.metric("Default Probability", f"{proba*100:.1f}%")

    with result_col2:
        st.progress(min(float(proba), 1.0))
        if proba < 0.2:
            st.caption("Low risk — well within safe lending range.")
        elif proba < 0.5:
            st.caption("Moderate risk — may warrant additional review.")
        else:
            st.caption("High risk — strong indicators of potential default.")

    st.divider()

    # ===== SHAP Explanation =====
    st.subheader("Why This Prediction? (SHAP Explanation)")
    st.markdown(
        "The chart below shows which factors pushed this applicant's risk score "
        "**up (red, toward default)** or **down (blue, toward non-default)**, and by how much."
    )

    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.force_plot(
        explainer.expected_value, shap_values[0], input_df.iloc[0],
        matplotlib=True, show=False
    )
    st.pyplot(plt.gcf())
    plt.close()

    # ===== Top contributing factors as a readable list =====
    st.markdown("**Top factors influencing this prediction:**")
    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": shap_values[0],
        "Applicant Value": input_df.iloc[0].values
    })
    shap_df["Impact"] = shap_df["SHAP Value"].apply(lambda x: "🔴 Increases Risk" if x > 0 else "🔵 Decreases Risk")
    shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index)

    st.dataframe(
        shap_df.head(6)[["Feature", "Applicant Value", "Impact", "SHAP Value"]],
        use_container_width=True,
        hide_index=True
    )

st.divider()
st.caption(
    "Model: XGBoost trained on the Kaggle 'Give Me Some Credit' dataset (150,000 loan applicants). "
    "Explanations generated using SHAP (SHapley Additive exPlanations)."
)
