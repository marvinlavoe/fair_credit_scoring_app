from __future__ import annotations

import pandas as pd


def get_fairness_metrics() -> list[dict]:
    """Return mock fairness metrics until backend evaluation artifacts are connected."""
    return [
        {
            "name": "Demographic Parity Difference",
            "value": 0.047,
            "display_value": "0.047",
            "status": "Fair",
            "explanation": "Close to 0 means approval rates are similar across groups.",
        },
        {
            "name": "Equalized Odds Difference",
            "value": 0.083,
            "display_value": "0.083",
            "status": "Needs Review",
            "explanation": "Close to 0 means error rates are similar across groups.",
        },
        {
            "name": "Disparate Impact Ratio",
            "value": 0.86,
            "display_value": "0.86",
            "status": "Fair",
            "explanation": "A value greater than or equal to 0.8 is commonly treated as acceptable.",
        },
    ]


def get_metric_comparison() -> pd.DataFrame:
    """Return a compact comparison for the grouped fairness/performance chart."""
    return pd.DataFrame(
        [
            {
                "Model": "Baseline TabNet",
                "AUC-ROC": 0.781,
                "F1 Score": 0.742,
                "Demographic Parity Difference": 0.142,
                "Equalized Odds Difference": 0.168,
                "Disparate Impact Ratio": 0.66,
            },
            {
                "Model": "Debiased TabNet",
                "AUC-ROC": 0.769,
                "F1 Score": 0.733,
                "Demographic Parity Difference": 0.047,
                "Equalized Odds Difference": 0.083,
                "Disparate Impact Ratio": 0.86,
            },
        ]
    )


def get_model_comparison() -> pd.DataFrame:
    """Return mock model comparison data for the demonstration table."""
    return pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "AUC-ROC": 0.724,
                "Weighted F1": 0.701,
                "Demographic Parity Difference": 0.118,
                "Equalized Odds Difference": 0.151,
                "Disparate Impact Ratio": 0.71,
                "Comment": "Interpretable baseline with moderate fairness gaps.",
            },
            {
                "Model": "TabNet Baseline",
                "AUC-ROC": 0.781,
                "Weighted F1": 0.742,
                "Demographic Parity Difference": 0.142,
                "Equalized Odds Difference": 0.168,
                "Disparate Impact Ratio": 0.66,
                "Comment": "Strong predictive model but higher fairness risk.",
            },
            {
                "Model": "TabNet + Adversarial Debiasing",
                "AUC-ROC": 0.769,
                "Weighted F1": 0.733,
                "Demographic Parity Difference": 0.047,
                "Equalized Odds Difference": 0.083,
                "Disparate Impact Ratio": 0.86,
                "Comment": "Best fairness balance with competitive performance.",
            },
        ]
    )
