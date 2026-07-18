from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

try:
    from .config import OUTPUT_DIR
except ImportError:
    from config import OUTPUT_DIR


def _to_series(values, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values, name=name)


def selection_rate_by_group(y_pred, sensitive_features) -> dict:
    df = pd.DataFrame(
        {
            "prediction": _to_series(y_pred, "prediction").astype(int),
            "group": _to_series(sensitive_features, "group").astype(str),
        }
    )
    return df.groupby("group")["prediction"].mean().to_dict()


def true_positive_rate_by_group(y_true, y_pred, sensitive_features) -> dict:
    df = pd.DataFrame(
        {
            "y_true": _to_series(y_true, "y_true").astype(int),
            "prediction": _to_series(y_pred, "prediction").astype(int),
            "group": _to_series(sensitive_features, "group").astype(str),
        }
    )
    rates = {}
    for group, group_df in df.groupby("group"):
        positives = group_df[group_df["y_true"] == 1]
        rates[group] = float(positives["prediction"].mean()) if len(positives) else np.nan
    return rates


def false_positive_rate_by_group(y_true, y_pred, sensitive_features) -> dict:
    df = pd.DataFrame(
        {
            "y_true": _to_series(y_true, "y_true").astype(int),
            "prediction": _to_series(y_pred, "prediction").astype(int),
            "group": _to_series(sensitive_features, "group").astype(str),
        }
    )
    rates = {}
    for group, group_df in df.groupby("group"):
        negatives = group_df[group_df["y_true"] == 0]
        rates[group] = float(negatives["prediction"].mean()) if len(negatives) else np.nan
    return rates


def disparate_impact_ratio(y_pred, sensitive_features) -> float:
    rates = selection_rate_by_group(y_pred, sensitive_features)
    valid_rates = [rate for rate in rates.values() if not pd.isna(rate)]
    if len(valid_rates) < 2 or max(valid_rates) == 0:
        return float("nan")
    return float(min(valid_rates) / max(valid_rates))


def disparate_impact_status(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    return "acceptable" if value >= 0.8 else "needs review"


def evaluate_fairness_for_attribute(
    y_true,
    y_pred,
    sensitive_features,
    sensitive_attribute: str,
) -> dict:
    sensitive = _to_series(sensitive_features, sensitive_attribute).astype(str)
    y_true_series = _to_series(y_true, "y_true").astype(int)
    y_pred_series = _to_series(y_pred, "y_pred").astype(int)

    di_ratio = disparate_impact_ratio(y_pred_series, sensitive)
    return {
        "sensitive_attribute": sensitive_attribute,
        "demographic_parity_difference": float(
            demographic_parity_difference(
                y_true_series,
                y_pred_series,
                sensitive_features=sensitive,
            )
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(
                y_true_series,
                y_pred_series,
                sensitive_features=sensitive,
            )
        ),
        "disparate_impact_ratio": di_ratio,
        "disparate_impact_interpretation": disparate_impact_status(di_ratio),
        "selection_rate_by_group": selection_rate_by_group(y_pred_series, sensitive),
        "true_positive_rate_by_group": true_positive_rate_by_group(
            y_true_series, y_pred_series, sensitive
        ),
        "false_positive_rate_by_group": false_positive_rate_by_group(
            y_true_series, y_pred_series, sensitive
        ),
    }


def evaluate_fairness(y_true, y_pred, A_test: pd.DataFrame) -> list[dict]:
    if A_test is None or A_test.empty:
        return []
    return [
        evaluate_fairness_for_attribute(y_true, y_pred, A_test[column], column)
        for column in A_test.columns
    ]


def compute_group_reweighting(y_train, sensitive_features) -> np.ndarray:
    """Compute fairness-aware reweighting by target class and sensitive group."""
    df = pd.DataFrame(
        {
            "target": _to_series(y_train, "target").astype(int),
            "group": _to_series(sensitive_features, "group").astype(str),
        }
    )
    total = len(df)
    weights = np.ones(total, dtype=np.float32)

    target_prob = df["target"].value_counts(normalize=True)
    group_prob = df["group"].value_counts(normalize=True)
    joint_prob = df.groupby(["target", "group"]).size() / total

    for idx, row in df.iterrows():
        observed = joint_prob.loc[(row["target"], row["group"])]
        expected = target_prob[row["target"]] * group_prob[row["group"]]
        weights[idx] = float(expected / observed) if observed > 0 else 1.0

    return weights


def save_fairness_results(results, path: Path = OUTPUT_DIR / "fairness_results.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def compute_fairness_metrics(y_true, y_pred, protected):
    """Backward-compatible single-attribute fairness summary."""
    if protected is None:
        return {
            "Demographic Parity Difference": None,
            "Equalized Odds Difference": None,
            "Disparate Impact Ratio": None,
        }
    result = evaluate_fairness_for_attribute(y_true, y_pred, protected, "protected")
    return {
        "Demographic Parity Difference": result["demographic_parity_difference"],
        "Equalized Odds Difference": result["equalized_odds_difference"],
        "Disparate Impact Ratio": result["disparate_impact_ratio"],
    }


compute_reweighting_sample_weights = compute_group_reweighting
