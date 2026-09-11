from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components import (
    render_card,
    render_flow,
    render_grouped_metric_chart,
    render_metric_card,
    render_result_panel,
    render_shap_bar_chart,
)
try:
    from src.inference import (
        generate_shap_explanation,
        get_fairness_metrics,
        get_metric_comparison,
        get_model_comparison,
        predict_credit,
    )
    USING_REAL_BACKEND = True
except Exception:
    from utils.mock_data import (
        get_fairness_metrics,
        get_metric_comparison,
        get_model_comparison,
    )
    from utils.mock_prediction import generate_shap_explanation, predict_credit
    USING_REAL_BACKEND = False

CATEGORY_LABEL_MAP = {
    "checking_account_status": {
        "1": "No account",
        "2": "Negative balance",
        "3": "Low balance",
        "4": "Healthy balance",
    },
    "credit_history": {
        "1": "No credits / paid back duly",
        "2": "All credits paid back duly",
        "3": "Existing credits paid back duly",
        "4": "Delay in paying off",
        "5": "Critical account / other credits",
    },
    "savings_status": {
        "1": "Unknown / no savings",
        "2": "Low",
        "3": "Moderate",
        "4": "High",
        "5": "Very high",
    },
    "employment_status": {
        "1": "Unemployed",
        "2": "Less than 1 year",
        "3": "1-4 years",
        "4": "4-7 years",
        "5": "More than 7 years",
    },
    "housing_status": {
        "Rent": "Rent",
        "Own": "Own",
        "Free": "Free",
    },
    "job_type": {
        "1": "Unemployed / unskilled",
        "2": "Unskilled resident",
        "3": "Skilled employee",
        "4": "Management / self-employed",
    },
    "loan_purpose": {
        "0": "Car (new)",
        "1": "Car (used)",
        "2": "Furniture / equipment",
        "3": "Radio / television",
        "4": "Domestic appliances",
        "5": "Repairs",
        "6": "Education",
        "8": "Retraining",
        "9": "Business",
        "10": "Other",
    },
    "other_debtors": {
        "1": "None",
        "2": "Co-applicant",
        "3": "Guarantor",
    },
    "property_type": {
        "1": "Real estate",
        "2": "Savings / insurance",
        "3": "Car or other assets",
        "4": "Unknown / no property",
    },
    "other_installment_plans": {
        "1": "Bank",
        "2": "Stores",
        "3": "None",
    },
    "telephone": {
        "1": "No registered telephone",
        "2": "Registered telephone",
    },
    "foreign_worker": {
        "1": "Yes",
        "2": "No",
    },
}


def _prettify_shap_feature_name(feature_name: str) -> str:
    if not isinstance(feature_name, str):
        return str(feature_name)

    if feature_name.startswith("numeric__"):
        return feature_name.replace("numeric__", "").replace("_", " ").title()

    if feature_name.startswith("categorical__"):
        payload = feature_name.split("categorical__", 1)[1]
        matching_key = next(
            (
                key
                for key in CATEGORY_LABEL_MAP
                if payload.startswith(f"{key}_")
            ),
            None,
        )
        if matching_key is not None:
            raw_value = payload[len(matching_key) + 1 :]
            key_label = matching_key.replace("_", " ").title()
            mapped_value = CATEGORY_LABEL_MAP[matching_key].get(raw_value)
            if mapped_value is None:
                mapped_value = raw_value.replace("_", " ").title()
            return f"{key_label} = {mapped_value}"
        return payload.replace("_", " ").title()

    return feature_name.replace("_", " ").title()


def _clean_shap_feature_names(shap_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = shap_df.copy()
    if "feature" in cleaned.columns:
        cleaned["feature"] = cleaned["feature"].apply(_prettify_shap_feature_name)
    return cleaned


def _fairness_status(metric_name: str, value: float) -> str:
    if metric_name == "Disparate Impact Ratio":
        if value >= 0.8:
            return "Fair"
        if value >= 0.65:
            return "Needs Review"
        return "High Bias Risk"

    absolute_value = abs(value)
    if absolute_value <= 0.05:
        return "Fair"
    if absolute_value <= 0.15:
        return "Needs Review"
    return "High Bias Risk"


def _format_real_fairness_metrics(raw_metrics: dict) -> list[dict]:
    if not raw_metrics:
        return []

    model_key = "tabnet_debiased" if "tabnet_debiased" in raw_metrics else next(iter(raw_metrics))
    model_metrics = raw_metrics.get(model_key, [])
    sex_metrics = next(
        (item for item in model_metrics if item.get("sensitive_attribute") == "sex"),
        model_metrics[0] if model_metrics else None,
    )
    if not sex_metrics:
        return []

    cards = [
        (
            "Demographic Parity Difference",
            float(sex_metrics.get("demographic_parity_difference", 0.0)),
            "Close to 0 means approval rates are similar across groups.",
        ),
        (
            "Equalized Odds Difference",
            float(sex_metrics.get("equalized_odds_difference", 0.0)),
            "Close to 0 means error rates are similar across groups.",
        ),
        (
            "Disparate Impact Ratio",
            float(sex_metrics.get("disparate_impact_ratio", 0.0)),
            "A value greater than or equal to 0.8 is commonly treated as acceptable.",
        ),
    ]
    return [
        {
            "name": name,
            "display_value": f"{value:.3f}",
            "status": _fairness_status(name, value),
            "explanation": explanation,
        }
        for name, value, explanation in cards
    ]


def _fairness_comparison_frame(raw_metrics: dict, sensitive_attribute: str) -> pd.DataFrame:
    labels = {
        "logistic_regression": "Logistic Regression",
        "tabnet_baseline": "TabNet Baseline",
        "tabnet_debiased": "Fairness-Aware TabNet",
    }
    rows = []
    for model_key, model_label in labels.items():
        metric = next(
            (
                item
                for item in raw_metrics.get(model_key, [])
                if item.get("sensitive_attribute") == sensitive_attribute
            ),
            None,
        )
        if metric:
            rows.append(
                {
                    "Model": model_label,
                    "Demographic Parity Difference": metric.get("demographic_parity_difference"),
                    "Equalized Odds Difference": metric.get("equalized_odds_difference"),
                    "Disparate Impact Ratio": metric.get("disparate_impact_ratio"),
                }
            )
    return pd.DataFrame(rows)


def render_dashboard_page() -> None:
    st.subheader("Portfolio Overview")
    st.markdown(
        "A compact view of the scoring workflow, model evidence, and fairness checks "
        "behind each applicant decision."
    )

    cols = st.columns(4)
    cards = [
        ("Prediction Model", "TabNet", "Neural tabular model for credit risk prediction"),
        (
            "Fairness Method",
            "Reweighting",
            "Balances sensitive group representation during training",
        ),
        ("Explainability", "SHAP", "Shows feature-level evidence for each decision"),
        ("Monitoring", "AUC, F1, DP, EO, DI", "Tracks accuracy and fairness side by side"),
    ]
    for col, card in zip(cols, cards):
        with col:
            render_card(*card)

    st.markdown("### Decision Workflow")
    render_flow(
        [
            "Applicant Input",
            "Preprocessing",
            "TabNet Model",
            "Prediction",
            "SHAP + Fairness Metrics",
            "Results",
        ]
    )


def render_predict_applicant_page() -> None:
    st.subheader("Applicant Review")
    if USING_REAL_BACKEND:
        st.markdown(
            "Enter applicant details to run the saved preprocessing pipeline and "
            "trained TabNet model."
        )
    else:
        st.markdown(
            "Enter applicant details below. Fallback mock outputs are active because "
            "trained backend artifacts are unavailable."
        )

    model_options = {
        "Fairness-Aware TabNet": "tabnet_debiased",
        "TabNet Baseline": "tabnet_baseline",
        "Logistic Regression": "logistic_regression",
    }

    with st.form("applicant_form"):
        selected_model = st.selectbox("Prediction model", list(model_options))
        personal, credit, financial = st.tabs(
            ["Personal Information", "Credit Information", "Financial Information"]
        )

        with personal:
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=80, value=34, step=1)
            with c2:
                gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            with c3:
                employment_status = st.selectbox(
                    "Employment status",
                    ["Unemployed", "Less than 1 year", "1-4 years", "4-7 years", "More than 7 years"],
                    index=2,
                )

        with credit:
            c1, c2, c3 = st.columns(3)
            with c1:
                credit_history = st.selectbox(
                    "Credit history",
                    [
                        "No credits / paid back duly",
                        "All credits paid back duly",
                        "Existing credits paid back duly",
                        "Delay in paying off",
                        "Critical account / other credits",
                    ],
                    index=2,
                )
                existing_credits = st.slider("Existing credits", 1, 4, 1)
            with c2:
                loan_amount = st.number_input(
                    "Loan amount",
                    min_value=250,
                    max_value=25000,
                    value=3500,
                    step=250,
                )
                savings_status = st.selectbox(
                    "Savings status",
                    ["Unknown / no savings", "Low", "Moderate", "High", "Very high"],
                    index=2,
                )
            with c3:
                loan_duration = st.slider("Loan duration (months)", 6, 72, 24, step=3)
                checking_status = st.selectbox(
                    "Checking account status",
                    ["No account", "Negative balance", "Low balance", "Healthy balance"],
                    index=2,
                )

        with financial:
            c1, c2, c3 = st.columns(3)
            with c1:
                installment_rate = st.slider("Installment rate (% of income)", 1, 4, 2)
            with c2:
                housing_status = st.selectbox("Housing status", ["Rent", "Own", "Free"], index=1)
                job_type = st.selectbox(
                    "Job type",
                    ["Unemployed / unskilled", "Unskilled resident", "Skilled employee", "Management / self-employed"],
                    index=2,
                )
            with c3:
                dependents = st.radio("Number of dependents", [1, 2], horizontal=True)

        with st.expander("Additional German Credit Features", expanded=False):
            st.caption(
                "These extra fields are part of the trained German Credit feature set. "
                "Adjusting them can produce more representative predictions than relying "
                "on hidden defaults."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                loan_purpose = st.selectbox(
                    "Loan purpose",
                    [
                        "Car (new)",
                        "Car (used)",
                        "Furniture / equipment",
                        "Radio / television",
                        "Domestic appliances",
                        "Repairs",
                        "Education",
                        "Retraining",
                        "Business",
                        "Other",
                    ],
                    index=3,
                )
                other_debtors = st.selectbox(
                    "Other debtors / guarantors",
                    ["None", "Co-applicant", "Guarantor"],
                    index=0,
                )
                other_installment_plans = st.selectbox(
                    "Other installment plans",
                    ["Bank", "Stores", "None"],
                    index=2,
                )
            with c2:
                residence_duration = st.slider("Residence duration category", 1, 4, 4)
                property_type = st.selectbox(
                    "Property type",
                    [
                        "Real estate",
                        "Savings / insurance",
                        "Car or other assets",
                        "Unknown / no property",
                    ],
                    index=2,
                )
            with c3:
                telephone = st.selectbox(
                    "Telephone status",
                    ["No registered telephone", "Registered telephone"],
                    index=0,
                )
                foreign_worker = st.selectbox(
                    "Foreign worker",
                    ["Yes", "No"],
                    index=1,
                )

        submitted = st.form_submit_button("Run Prediction", width="stretch")

    if submitted:
        applicant_data = {
            "age": age,
            "gender": gender,
            "employment_status": employment_status,
            "credit_history": credit_history,
            "loan_amount": loan_amount,
            "loan_duration": loan_duration,
            "existing_credits": existing_credits,
            "savings_status": savings_status,
            "checking_account_status": checking_status,
            "installment_rate": installment_rate,
            "housing_status": housing_status,
            "job_type": job_type,
            "dependents": dependents,
            "loan_purpose": loan_purpose,
            "other_debtors": other_debtors,
            "residence_duration": residence_duration,
            "property_type": property_type,
            "other_installment_plans": other_installment_plans,
            "telephone": telephone,
            "foreign_worker": foreign_worker,
        }
        st.session_state["applicant_data"] = applicant_data
        model_key = model_options[selected_model]
        st.session_state["selected_model"] = selected_model
        st.session_state["prediction_result"] = predict_credit(
            applicant_data,
            model_key=model_key,
        )
        st.session_state["shap_explanation"] = generate_shap_explanation(
            applicant_data,
            model_key=model_key,
        )

    if "prediction_result" in st.session_state:
        st.markdown("### Decision Summary")
        render_result_panel(st.session_state["prediction_result"])

        st.markdown("#### Applicant Record")
        st.dataframe(pd.DataFrame([st.session_state["applicant_data"]]), width="stretch")
    else:
        st.markdown(
            "<div class='section-note'>Submit the form to generate the prediction, "
            "confidence score, risk level, and applicant-level explanation.</div>",
            unsafe_allow_html=True,
        )


def render_shap_explanation_page() -> None:
    st.subheader("Decision Evidence")
    st.markdown(
        "Positive values increase the chance of being classified as creditworthy, "
        "while negative values reduce the chance."
    )

    explanation = st.session_state.get("shap_explanation")
    if explanation is None:
        demo_applicant = {
            "age": 34,
            "employment_status": "1-4 years",
            "credit_history": "Existing credits paid back duly",
            "loan_amount": 3500,
            "loan_duration": 24,
            "checking_account_status": "Low balance",
        }
        explanation = generate_shap_explanation(demo_applicant)
        if USING_REAL_BACKEND:
            st.info("Run a prediction to update this explanation for a specific applicant.")
        else:
            st.info("Showing a mock explanation. Run a prediction to update this panel for a specific applicant.")

    shap_values = _clean_shap_feature_names(explanation["values"])
    positives = shap_values[shap_values["contribution"] > 0].sort_values("contribution", ascending=False)
    negatives = shap_values[shap_values["contribution"] < 0].sort_values("contribution")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top Positive Contributing Features")
        st.dataframe(positives.head(5), hide_index=True, width="stretch")
    with right:
        st.markdown("#### Top Negative Contributing Features")
        st.dataframe(negatives.head(5), hide_index=True, width="stretch")

    st.markdown("#### Feature Contribution Chart")
    render_shap_bar_chart(shap_values)

    if USING_REAL_BACKEND and "base_value" in explanation:
        st.caption(
            f"SHAP base value: {explanation['base_value']:.4f} | "
            f"Base value plus all contributions: {explanation['additivity_total']:.4f} | "
            f"Model probability: {explanation['model_probability']:.4f} | "
            f"Additivity error: {explanation['additivity_error']:.2e}"
        )
    else:
        st.markdown(
            "<div class='section-note'>Showing fallback explanation values because trained "
            "SHAP artifacts are unavailable.</div>",
            unsafe_allow_html=True,
        )
    st.write(explanation["plain_language"])


def render_fairness_metrics_page() -> None:
    st.subheader("Fairness Monitoring")
    raw_metrics = get_fairness_metrics()
    metrics = _format_real_fairness_metrics(raw_metrics) if USING_REAL_BACKEND else raw_metrics

    if USING_REAL_BACKEND:
        available_attributes = sorted(
            {
                item.get("sensitive_attribute")
                for model_metrics in raw_metrics.values()
                for item in model_metrics
                if item.get("sensitive_attribute")
            }
        )
        selected_attribute = st.selectbox(
            "Protected attribute",
            available_attributes or ["sex"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        selected_model = st.selectbox(
            "Highlighted model",
            ["tabnet_debiased", "tabnet_baseline", "logistic_regression"],
            format_func=lambda value: {
                "logistic_regression": "Logistic Regression",
                "tabnet_baseline": "TabNet Baseline",
                "tabnet_debiased": "Fairness-Aware TabNet",
            }[value],
        )
        selected_metrics = {
            selected_model: [
                item
                for item in raw_metrics.get(selected_model, [])
                if item.get("sensitive_attribute") == selected_attribute
            ]
        }
        metrics = _format_real_fairness_metrics(selected_metrics)
        st.caption("Compare fairness metrics across all saved models for the selected protected attribute.")

    cols = st.columns(3)
    for col, metric in zip(cols, metrics):
        with col:
            render_metric_card(
                metric["name"],
                metric["display_value"],
                metric["status"],
                metric["explanation"],
            )

    st.markdown("### Logistic Regression vs TabNet Models")
    if USING_REAL_BACKEND:
        fairness_comparison = _fairness_comparison_frame(raw_metrics, selected_attribute)
        if not fairness_comparison.empty:
            st.dataframe(
                fairness_comparison.style.format(
                    {
                        "Demographic Parity Difference": "{:.3f}",
                        "Equalized Odds Difference": "{:.3f}",
                        "Disparate Impact Ratio": "{:.3f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            render_grouped_metric_chart(fairness_comparison)
        else:
            st.warning("Fairness metrics are not available for the selected protected attribute.")
    else:
        metric_comparison = get_metric_comparison()
        if not metric_comparison.empty:
            render_grouped_metric_chart(metric_comparison)
        else:
            st.warning("Model comparison metrics are not available yet. Run the backend training pipeline first.")


def render_model_comparison_page() -> None:
    st.subheader("Model Performance")
    comparison = get_model_comparison()

    def highlight_debiased(row: pd.Series) -> list[str]:
        if row["Model"] in {"TabNet + Adversarial Debiasing", "TabNet + Fairness-Aware Reweighting"}:
            return ["background-color: #dcfce7; color: #14532d; font-weight: 700"] * len(row)
        return [""] * len(row)

    if comparison.empty:
        st.warning("Model comparison is not available yet. Run the backend training pipeline first.")
    else:
        st.dataframe(
            comparison.style.apply(highlight_debiased, axis=1).format(
                {
                    "AUC-ROC": "{:.3f}",
                    "Weighted F1": "{:.3f}",
                    "Demographic Parity Difference": "{:.3f}",
                    "Equalized Odds Difference": "{:.3f}",
                    "Disparate Impact Ratio": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        "<div class='section-note'>The debiased TabNet model is expected to reduce "
        "fairness gaps while preserving competitive predictive performance.</div>",
        unsafe_allow_html=True,
    )


def render_about_page() -> None:
    st.subheader("Methodology")

    st.markdown("### Project Aim")
    st.write(
        "To develop a fair and explainable credit scoring system using TabNet, "
        "adversarial debiasing, and SHAP."
    )

    st.markdown("### Methods Used")
    st.markdown(
        """
        - **TabNet for prediction:** a deep learning architecture designed for tabular data.
        - **Fairness-aware reweighting for fairness improvement:** reduces imbalance across sensitive groups during training.
        - **SHAP for explainability:** estimates how each feature contributes to an individual decision.
        - **Streamlit for frontend interface:** provides an interactive academic demonstration dashboard.
        """
    )

    st.markdown("### Datasets")
    st.markdown(
        """
        - **German Credit dataset:** primary benchmark for credit scoring and fairness evaluation.
        - **HELOC dataset:** optional extension for additional credit-risk validation.
        """
    )

    st.markdown("### Disclaimer")
    st.warning(
        "This application is for academic research and demonstration only. It should "
        "not be used for real credit approval decisions."
    )
