"""
Credit Risk / Loan Default Prediction App with Explainable AI (SHAP)
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import csv
from datetime import datetime
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

# Get the folder this script lives in, so file paths work regardless of
# what directory the app is launched FROM (fixes deployment path issues)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(APP_DIR, "prediction_log.csv")


def log_prediction(proba, prediction):
    """Append one prediction result to the local CSV log."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "probability", "prediction"])
        writer.writerow([datetime.now().isoformat(), proba, int(prediction)])


def read_prediction_log():
    """Read the log file, returning an empty DataFrame if it doesn't exist yet."""
    if os.path.isfile(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["timestamp", "probability", "prediction"])

# ===== Load model and artifacts (cached so it only loads once) =====
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(APP_DIR, "model.json"))

    with open(os.path.join(APP_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    with open(os.path.join(APP_DIR, "defaults.json")) as f:
        defaults = json.load(f)

    with open(os.path.join(APP_DIR, "dataset_averages.json")) as f:
        dataset_averages = json.load(f)

    explainer = shap.TreeExplainer(model)
    return model, feature_columns, defaults, dataset_averages, explainer

model, feature_columns, defaults, dataset_averages, explainer = load_artifacts()

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

    # Log this prediction for the analytics dashboard
    log_prediction(proba, prediction)

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

    # ===== Visualizations: Risk Gauge + Comparison to Dataset =====
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**Risk Gauge**")
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        risk_color = "#e74c3c" if proba >= 0.5 else ("#f39c12" if proba >= 0.2 else "#2ecc71")
        wedges, _ = ax1.pie(
            [proba, 1 - proba],
            colors=[risk_color, "#e8e8e8"],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.35}
        )
        ax1.text(0, 0, f"{proba*100:.1f}%", ha="center", va="center", fontsize=22, fontweight="bold")
        ax1.text(0, -0.25, "Default Risk", ha="center", va="center", fontsize=10, color="gray")
        ax1.set_aspect("equal")
        st.pyplot(fig1)
        plt.close(fig1)

    with viz_col2:
        st.markdown("**Applicant vs. Typical Applicant (Dataset Median)**")
        compare_features = {
            "Age": (age, dataset_averages["age"]),
            "Monthly Income ($)": (monthly_income, dataset_averages["MonthlyIncome"]),
            "Debt Ratio": (debt_ratio, dataset_averages["DebtRatio"]),
            "Open Credit Lines": (open_credit_lines, dataset_averages["NumberOfOpenCreditLinesAndLoans"]),
        }
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        labels = list(compare_features.keys())
        applicant_vals = [v[0] for v in compare_features.values()]
        typical_vals = [v[1] for v in compare_features.values()]

        # Normalize each pair to 0-100 scale so bars are visually comparable side by side
        norm_applicant, norm_typical = [], []
        for a_val, t_val in zip(applicant_vals, typical_vals):
            max_val = max(a_val, t_val, 1)
            norm_applicant.append((a_val / max_val) * 100)
            norm_typical.append((t_val / max_val) * 100)

        y_pos = np.arange(len(labels))
        ax2.barh(y_pos - 0.2, norm_applicant, height=0.4, label="This Applicant", color="#3498db")
        ax2.barh(y_pos + 0.2, norm_typical, height=0.4, label="Typical Applicant", color="#95a5a6")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel("Relative Scale (%)")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

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

# ===== Prediction Analytics Dashboard =====
st.subheader("📊 Prediction Analytics")
st.caption("Aggregate stats across all predictions made using this app (this session/deployment).")

log_df = read_prediction_log()

if len(log_df) == 0:
    st.info("No predictions made yet. Run a prediction above to start building analytics.")
else:
    total_predictions = len(log_df)
    high_risk_count = int((log_df["prediction"] == 1).sum())
    low_risk_count = int((log_df["prediction"] == 0).sum())
    avg_risk = log_df["probability"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predictions", total_predictions)
    m2.metric("High-Risk Predictions", high_risk_count)
    m3.metric("Low-Risk Predictions", low_risk_count)
    m4.metric("Average Risk Score", f"{avg_risk*100:.1f}%")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Risk Distribution**")
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        counts = [low_risk_count, high_risk_count]
        labels_pie = [f"Low Risk ({low_risk_count})", f"High Risk ({high_risk_count})"]
        colors_pie = ["#2ecc71", "#e74c3c"]
        # Avoid drawing an empty pie if one category is zero
        non_zero = [(c, l, col) for c, l, col in zip(counts, labels_pie, colors_pie) if c > 0]
        if non_zero:
            ax3.pie(
                [c for c, _, _ in non_zero],
                labels=[l for _, l, _ in non_zero],
                colors=[col for _, _, col in non_zero],
                autopct="%1.0f%%",
                startangle=90
            )
        ax3.set_aspect("equal")
        st.pyplot(fig3)
        plt.close(fig3)

    with chart_col2:
        st.markdown("**Risk Score Distribution (Histogram)**")
        fig4, ax4 = plt.subplots(figsize=(4, 4))
        ax4.hist(log_df["probability"] * 100, bins=10, color="#3498db", edgecolor="white")
        ax4.set_xlabel("Predicted Risk (%)")
        ax4.set_ylabel("Number of Predictions")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    with st.expander("View raw prediction log"):
        st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear Prediction History"):
        os.remove(LOG_FILE)
        st.rerun()

st.divider()
st.caption(
    "Model: XGBoost trained on the Kaggle 'Give Me Some Credit' dataset (150,000 loan applicants). "
    "Explanations generated using SHAP (SHapley Additive exPlanations)."
)
