from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability import generate_shap_explanation
from src.fairness import evaluate_fairness
from src.inference import (
    GERMAN_DEFAULT_APPLICANT,
    explain_prediction,
    get_fairness_metrics,
    get_model_comparison,
    load_artifacts,
    predict_credit,
)
from src.preprocess import ensure_german_sensitive_columns, preprocess_data


class LinearProbabilityModel:
    """Small deterministic model used to test the SHAP integration."""

    def predict_proba(self, values):
        values = np.asarray(values, dtype=np.float32)
        probability = np.clip(0.25 + 0.1 * values[:, 0] + 0.05 * values[:, 1], 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def test_sensitive_columns_are_created_from_german_source():
    frame = pd.DataFrame(
        {
            "personal_status_sex": [2, 1],
            "age": [20, 40],
            "target": [1, 0],
        }
    )

    result = ensure_german_sensitive_columns(frame)

    assert list(result["sex"]) == ["female", "male"]
    assert list(result["age_group"]) == ["below_25", "25_and_above"]


def test_missing_sensitive_source_column_raises_clear_error():
    frame = pd.DataFrame({"age": [30], "target": [1]})

    with pytest.raises(ValueError, match="personal_status_sex"):
        ensure_german_sensitive_columns(frame)


def test_preprocessing_returns_aligned_arrays():
    data = preprocess_data(dataset="german", save_preprocessor=False)

    assert data["X_train"].shape[0] == len(data["y_train"])
    assert data["X_test"].shape[0] == len(data["y_test"])
    assert data["X_train"].shape[1] == data["X_test"].shape[1]
    assert len(data["feature_names"]) == data["X_train"].shape[1]
    assert set(data["A_train"].columns) == {"sex", "age_group"}


def test_empty_shap_background_is_rejected():
    model = LinearProbabilityModel()

    with pytest.raises(ValueError, match="X_background is required"):
        generate_shap_explanation(model, np.empty((0, 2)), np.array([[1.0, 2.0]]))


def test_shap_values_reconstruct_probability():
    model = LinearProbabilityModel()
    background = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    instance = np.array([[0.5, 0.25]], dtype=np.float32)

    result = generate_shap_explanation(
        model,
        background,
        instance,
        background_size=2,
        nsamples=20,
    )

    assert result["shap_values"].shape == (2,)
    assert result["explained_probability"] == pytest.approx(
        result["base_value"] + result["shap_values"].sum(), abs=1e-6
    )


def test_fairness_metrics_include_required_measures():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    protected = pd.Series(["female", "female", "female", "male", "male", "male"])

    result = evaluate_fairness(y_true, y_pred, pd.DataFrame({"sex": protected}))

    assert len(result) == 1
    assert result[0]["sensitive_attribute"] == "sex"
    assert {
        "demographic_parity_difference",
        "equalized_odds_difference",
        "disparate_impact_ratio",
    }.issubset(result[0])


@pytest.mark.integration
def test_saved_artifacts_flow_from_prediction_to_explanation():
    pytest.importorskip("pytorch_tabnet")
    try:
        load_artifacts.cache_clear()
        artifacts = load_artifacts("german")
    except FileNotFoundError as error:
        pytest.skip(str(error))

    prediction = predict_credit(dict(GERMAN_DEFAULT_APPLICANT))
    explanation = explain_prediction(dict(GERMAN_DEFAULT_APPLICANT))

    assert prediction["prediction"] in {"Creditworthy", "Not Creditworthy"}
    assert 0.0 <= prediction["creditworthy_probability"] <= 1.0
    assert len(explanation["top_contributions"]) == 10
    assert explanation["additivity_total"] == pytest.approx(
        explanation["model_probability"], abs=1e-5
    )
    assert artifacts["feature_names"]


@pytest.mark.integration
def test_saved_metric_outputs_are_displayable():
    fairness = get_fairness_metrics("german")
    comparison = get_model_comparison("german")

    if not fairness or comparison.empty:
        pytest.skip("Saved fairness or model comparison outputs are unavailable")

    assert "tabnet_debiased" in fairness
    assert {"Model", "AUC-ROC", "Weighted F1"}.issubset(comparison.columns)
    assert len(comparison) >= 3
