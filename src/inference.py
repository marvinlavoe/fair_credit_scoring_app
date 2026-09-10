from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from .config import DEFAULT_DATASET, get_dataset_config, get_model_paths, get_output_paths
    from .explainability import generate_shap_explanation as generate_kernel_shap_explanation
    from .explainability import get_top_feature_contributions
    from .fairness import evaluate_fairness
    from .preprocess import preprocess_data
except ImportError:
    from config import DEFAULT_DATASET, get_dataset_config, get_model_paths, get_output_paths
    from explainability import generate_shap_explanation as generate_kernel_shap_explanation
    from explainability import get_top_feature_contributions
    from fairness import evaluate_fairness
    from preprocess import preprocess_data


TEXT_TO_CODE = {
    "checking_account_status": {
        "No account": 4,
        "Negative balance": 1,
        "Low balance": 2,
        "Healthy balance": 3,
    },
    "credit_history": {
        "No credits / paid back duly": 1,
        "All credits paid back duly": 2,
        "Existing credits paid back duly": 3,
        "Delay in paying off": 4,
        "Critical account / other credits": 5,
    },
    "savings_account_status": {
        "Unknown / no savings": 1,
        "Low": 2,
        "Moderate": 3,
        "High": 4,
        "Very high": 5,
    },
    "employment_duration": {
        "Unemployed": 1,
        "Less than 1 year": 2,
        "1-4 years": 3,
        "4-7 years": 4,
        "More than 7 years": 5,
    },
    "housing_status": {"Rent": 1, "Own": 2, "Free": 3},
    "job_type": {
        "Unemployed / unskilled": 1,
        "Unskilled resident": 2,
        "Skilled employee": 3,
        "Management / self-employed": 4,
    },
    "loan_purpose": {
        "Car (new)": 0,
        "Car (used)": 1,
        "Furniture / equipment": 2,
        "Radio / television": 3,
        "Domestic appliances": 4,
        "Repairs": 5,
        "Education": 6,
        "Retraining": 8,
        "Business": 9,
        "Other": 10,
    },
    "other_debtors": {
        "None": 1,
        "Co-applicant": 2,
        "Guarantor": 3,
    },
    "property_type": {
        "Real estate": 1,
        "Savings / insurance": 2,
        "Car or other assets": 3,
        "Unknown / no property": 4,
    },
    "other_installment_plans": {
        "Bank": 1,
        "Stores": 2,
        "None": 3,
    },
    "telephone": {
        "No registered telephone": 1,
        "Registered telephone": 2,
    },
    "foreign_worker": {
        "Yes": 1,
        "No": 2,
    },
}

FORCED_DECISION_THRESHOLD = 0.60

LEGACY_DECISION_THRESHOLDS = {
    "german": {
        "tabnet_baseline": FORCED_DECISION_THRESHOLD,
        "tabnet_debiased": FORCED_DECISION_THRESHOLD,
    }
}


GERMAN_DEFAULT_APPLICANT = {
    "checking_account_status": 2,
    "loan_duration_months": 24,
    "credit_history": 3,
    "loan_purpose": 3,
    "loan_amount": 3500,
    "savings_account_status": 3,
    "employment_duration": 3,
    "installment_rate": 2,
    "other_debtors": 1,
    "residence_duration": 4,
    "property_type": 3,
    "age": 34,
    "other_installment_plans": 3,
    "housing_status": 2,
    "existing_credits": 1,
    "job_type": 3,
    "number_of_dependents": 1,
    "telephone": 1,
    "foreign_worker": 2,
}


def _load_tabnet(path):
    model = TabNetClassifier()
    model.load_model(str(path))
    return model


def _load_saved_metrics(dataset: str) -> dict:
    metrics_path = get_output_paths(dataset)["metrics"]
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _resolve_decision_threshold(dataset: str, model_key: str) -> float:
    saved_metrics = _load_saved_metrics(dataset)
    if model_key in saved_metrics and "decision_threshold" in saved_metrics[model_key]:
        return FORCED_DECISION_THRESHOLD
    return FORCED_DECISION_THRESHOLD


@lru_cache(maxsize=4)
def load_artifacts(dataset: str = DEFAULT_DATASET):
    model_paths = get_model_paths(dataset)
    model_path = model_paths["tabnet_debiased"]
    model_key = "tabnet_debiased"
    if not model_path.exists():
        model_path = model_paths["tabnet_baseline"]
        model_key = "tabnet_baseline"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained TabNet model found for dataset '{dataset}'.")

    preprocessor_path = model_paths["preprocessor"]
    feature_columns_path = model_paths["raw_feature_columns"]
    feature_names_path = model_paths["feature_names"]
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing preprocessor artifact: {preprocessor_path}")
    if not feature_columns_path.exists():
        raise FileNotFoundError(f"Missing raw feature artifact: {feature_columns_path}")

    feature_names = joblib.load(feature_names_path) if feature_names_path.exists() else []

    return {
        "dataset": dataset,
        "model_key": model_key,
        "model": _load_tabnet(model_path),
        "model_path": model_path,
        "decision_threshold": _resolve_decision_threshold(dataset, model_key),
        "preprocessor": joblib.load(preprocessor_path),
        "raw_feature_columns": joblib.load(feature_columns_path),
        "feature_names": feature_names,
    }


def _default_applicant(dataset: str) -> dict:
    if dataset == "german":
        return GERMAN_DEFAULT_APPLICANT.copy()
    return {}


def _normalise_applicant(applicant_data: dict, raw_feature_columns: list[str], dataset: str) -> pd.DataFrame:
    data = _default_applicant(dataset)
    aliases = {
        "loan_duration": "loan_duration_months",
        "savings_status": "savings_account_status",
        "employment_status": "employment_duration",
        "dependents": "number_of_dependents",
    }
    for key, value in applicant_data.items():
        target_key = aliases.get(key, key)
        if dataset == "german" and target_key in TEXT_TO_CODE and isinstance(value, str):
            data[target_key] = TEXT_TO_CODE[target_key].get(value, data.get(target_key))
        else:
            data[target_key] = value

    return pd.DataFrame([{col: data.get(col, 0) for col in raw_feature_columns}])


def _risk_level(prediction_label: int, confidence: float) -> str:
    if prediction_label == 1 and confidence >= 0.75:
        return "Low Risk"
    if 0.55 <= confidence < 0.75:
        return "Medium Risk"
    if prediction_label == 0:
        return "High Risk"
    return "Medium Risk"


def _format_model_name(model_key: str) -> str:
    return {
        "tabnet_baseline": "TabNet baseline",
        "tabnet_debiased": "TabNet fairness-aware reweighted",
    }.get(model_key, model_key.replace("_", " "))


def _format_explanation_feature_name(feature_name: str, raw_feature_columns: list[str]) -> str:
    if feature_name.startswith("categorical__"):
        payload = feature_name.split("categorical__", 1)[1]
        matching_column = next(
            (
                column
                for column in sorted(raw_feature_columns, key=len, reverse=True)
                if payload.startswith(f"{column}_")
            ),
            None,
        )
        if matching_column is not None:
            value = payload[len(matching_column) + 1 :].replace("_", " ")
            return f"{matching_column.replace('_', ' ').title()} = {value.title()}"
        return payload.replace("_", " ").title()
    if feature_name.startswith("numeric__"):
        return feature_name.replace("numeric__", "").replace("_", " ").title()
    return feature_name.replace("_", " ").title()


def predict_credit(applicant_data: dict, dataset: str = DEFAULT_DATASET) -> dict:
    artifacts = load_artifacts(dataset)
    input_df = _normalise_applicant(applicant_data, artifacts["raw_feature_columns"], dataset)
    X = artifacts["preprocessor"].transform(input_df).astype(np.float32)
    proba = artifacts["model"].predict_proba(X)[0]
    threshold = float(artifacts["decision_threshold"])
    creditworthy_proba = float(proba[1])
    prediction_label = int(creditworthy_proba >= threshold)
    confidence = creditworthy_proba if prediction_label == 1 else (1.0 - creditworthy_proba)
    decision_margin = abs(creditworthy_proba - threshold)
    prediction = "Creditworthy" if prediction_label == 1 else "Not Creditworthy"

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "creditworthy_probability": creditworthy_proba,
        "decision_threshold": threshold,
        "decision_margin": decision_margin,
        "risk_level": _risk_level(prediction_label, confidence),
        "interpretation": (
            f"The trained {dataset.upper()} {_format_model_name(artifacts['model_key'])} "
            f"model produced this prediction using a deployed decision threshold of {threshold:.2f}."
        ),
    }


def explain_prediction(applicant_data: dict, dataset: str = DEFAULT_DATASET) -> dict:
    artifacts = load_artifacts(dataset)
    input_df = _normalise_applicant(applicant_data, artifacts["raw_feature_columns"], dataset)
    X = artifacts["preprocessor"].transform(input_df).astype(np.float32)
    if not artifacts["feature_names"]:
        raise ValueError("Missing feature names for explanation.")

    background = preprocess_data(dataset=dataset)["X_train"]
    shap_result = generate_kernel_shap_explanation(
        artifacts["model"],
        background,
        X,
    )
    shap_values = shap_result["shap_values"]
    contributions = get_top_feature_contributions(
        shap_values,
        artifacts["feature_names"],
        top_n=10,
    )
    values_df = pd.DataFrame(
        {
            "feature": artifacts["feature_names"],
            "contribution": shap_values,
        }
    )
    if values_df.empty:
        raise ValueError("Empty explanation output.")

    strongest_positive = _format_explanation_feature_name(
        values_df.sort_values("contribution", ascending=False).iloc[0]["feature"],
        artifacts["raw_feature_columns"],
    )
    strongest_negative = _format_explanation_feature_name(
        values_df.sort_values("contribution", ascending=True).iloc[0]["feature"],
        artifacts["raw_feature_columns"],
    )
    additivity_total = shap_result["base_value"] + float(values_df["contribution"].sum())
    return {
        "top_contributions": contributions,
        "values": values_df,
        "base_value": shap_result["base_value"],
        "model_probability": shap_result["explained_probability"],
        "additivity_total": additivity_total,
        "additivity_error": abs(additivity_total - shap_result["explained_probability"]),
        "plain_language": (
            f"In this SHAP explanation for the trained {dataset.upper()} model, {strongest_positive} "
            f"contributes the strongest positive signal, while {strongest_negative} "
            "contributes the strongest negative signal for the current prediction."
        ),
    }


def generate_shap_explanation(applicant_data: dict, dataset: str = DEFAULT_DATASET) -> dict:
    return explain_prediction(applicant_data, dataset=dataset)


def get_fairness_metrics(dataset: str = DEFAULT_DATASET):
    path = get_output_paths(dataset)["fairness_results"]
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _majority_class_row(dataset: str) -> dict:
    data = preprocess_data(dataset=dataset, save_preprocessor=False)
    y_test = np.asarray(data["y_test"], dtype=int)
    majority_label = int(np.bincount(y_test).argmax())
    y_pred = np.full(y_test.shape, majority_label, dtype=int)
    fairness = evaluate_fairness(y_test, y_pred, data["A_test"])
    primary_fairness = next(
        (item for item in fairness if item.get("sensitive_attribute") == "sex"),
        fairness[0] if fairness else {},
    )
    return {
        "Model": "Majority-Class Baseline",
        "AUC-ROC": 0.5,
        "Weighted F1": float(f1_score(y_test, y_pred, average="weighted")),
        "Demographic Parity Difference": primary_fairness.get(
            "demographic_parity_difference", np.nan
        ),
        "Equalized Odds Difference": primary_fairness.get(
            "equalized_odds_difference", np.nan
        ),
        "Disparate Impact Ratio": primary_fairness.get(
            "disparate_impact_ratio", np.nan
        ),
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Comment": f"Always predicts class {majority_label} (Creditworthy).",
    }


def get_model_comparison(dataset: str = DEFAULT_DATASET):
    path = get_output_paths(dataset)["model_comparison"]
    if not path.exists():
        return pd.DataFrame()
    comparison = pd.read_csv(path)
    if "Majority-Class Baseline" not in comparison.get("Model", pd.Series(dtype=str)).values:
        comparison = pd.concat(
            [pd.DataFrame([_majority_class_row(dataset)]), comparison],
            ignore_index=True,
        )
    return comparison


def get_metric_comparison(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    comparison = get_model_comparison(dataset)
    if comparison.empty:
        return pd.DataFrame()

    filtered = comparison[
        comparison["Model"].isin(
            ["TabNet Baseline", "TabNet + Fairness-Aware Reweighting"]
        )
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.rename(columns={"Weighted F1": "F1 Score"})
    filtered["Model"] = filtered["Model"].replace(
        {"TabNet + Fairness-Aware Reweighting": "Debiased TabNet"}
    )
    return filtered[
        [
            "Model",
            "AUC-ROC",
            "F1 Score",
            "Demographic Parity Difference",
            "Equalized Odds Difference",
            "Disparate Impact Ratio",
        ]
    ]
