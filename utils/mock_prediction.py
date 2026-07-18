from __future__ import annotations

import pandas as pd


def _normalise(value: float, lower: float, upper: float) -> float:
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def predict_credit(applicant_data: dict) -> dict:
    """Mock prediction function designed to be replaced by real TabNet inference.

    The score is deterministic and intentionally simple so demonstrations are
    repeatable before trained model artifacts are connected.
    """
    score = 0.52

    credit_history = applicant_data.get("credit_history", "")
    employment_status = applicant_data.get("employment_status", "")
    savings_status = applicant_data.get("savings_status", "")
    checking_status = applicant_data.get("checking_account_status", "")
    housing_status = applicant_data.get("housing_status", "")
    job_type = applicant_data.get("job_type", "")

    if "Critical" in credit_history or "paid back duly" in credit_history:
        score += 0.12
    if "Delay" in credit_history or "No credits" in credit_history:
        score -= 0.10
    if employment_status in {"4-7 years", "More than 7 years"}:
        score += 0.08
    elif employment_status == "Unemployed":
        score -= 0.12
    if savings_status in {"High", "Very high"}:
        score += 0.07
    elif savings_status in {"Unknown / no savings", "Low"}:
        score -= 0.04
    if checking_status == "Healthy balance":
        score += 0.08
    elif checking_status in {"No account", "Negative balance"}:
        score -= 0.09
    if housing_status == "Own":
        score += 0.04
    if job_type == "Management / self-employed":
        score += 0.03

    loan_amount = float(applicant_data.get("loan_amount", 3500))
    duration = float(applicant_data.get("loan_duration", 24))
    installment_rate = float(applicant_data.get("installment_rate", 2))
    age = float(applicant_data.get("age", 34))

    score -= _normalise(loan_amount, 250, 25000) * 0.18
    score -= _normalise(duration, 6, 72) * 0.09
    score -= _normalise(installment_rate, 1, 4) * 0.05
    if 28 <= age <= 55:
        score += 0.04

    probability = max(0.05, min(0.95, score))
    is_creditworthy = probability >= 0.5
    confidence = probability if is_creditworthy else 1 - probability

    if probability >= 0.68:
        risk_level = "Low Risk"
    elif probability >= 0.48:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    if is_creditworthy:
        interpretation = (
            "The applicant is predicted as creditworthy because the model identified "
            "favourable credit history, manageable loan conditions, or stable financial "
            "signals as positive factors."
        )
    else:
        interpretation = (
            "The applicant is predicted as not creditworthy because the model identified "
            "higher loan burden, weaker account status, or less stable repayment signals."
        )

    return {
        "prediction": "Creditworthy" if is_creditworthy else "Not Creditworthy",
        "probability_creditworthy": probability,
        "confidence": confidence,
        "risk_level": risk_level,
        "interpretation": interpretation,
    }


def generate_shap_explanation(applicant_data: dict) -> dict:
    """Return mock SHAP-like feature contributions for the active applicant."""
    credit_history = applicant_data.get("credit_history", "Existing credits paid back duly")
    employment_status = applicant_data.get("employment_status", "1-4 years")
    checking_status = applicant_data.get("checking_account_status", "Low balance")
    loan_amount = float(applicant_data.get("loan_amount", 3500))
    duration = float(applicant_data.get("loan_duration", 24))
    age = float(applicant_data.get("age", 34))

    values = [
        {
            "feature": "Credit history",
            "contribution": 0.18 if "paid back duly" in credit_history or "Critical" in credit_history else -0.13,
        },
        {
            "feature": "Loan amount",
            "contribution": -0.04 - _normalise(loan_amount, 250, 25000) * 0.16,
        },
        {
            "feature": "Duration",
            "contribution": -0.02 - _normalise(duration, 6, 72) * 0.11,
        },
        {
            "feature": "Age",
            "contribution": 0.07 if 28 <= age <= 55 else -0.04,
        },
        {
            "feature": "Employment status",
            "contribution": 0.10 if employment_status in {"4-7 years", "More than 7 years", "1-4 years"} else -0.11,
        },
        {
            "feature": "Checking account status",
            "contribution": 0.09 if checking_status == "Healthy balance" else -0.08,
        },
    ]
    values_df = pd.DataFrame(values)
    strongest_positive = values_df.sort_values("contribution", ascending=False).iloc[0]["feature"]
    strongest_negative = values_df.sort_values("contribution", ascending=True).iloc[0]["feature"]

    return {
        "values": values_df,
        "plain_language": (
            f"In this mock explanation, {strongest_positive} provides the strongest "
            f"positive evidence, while {strongest_negative} applies the strongest "
            "downward pressure on the creditworthy classification."
        ),
    }
