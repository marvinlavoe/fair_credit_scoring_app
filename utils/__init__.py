"""Mock frontend utilities and compatibility helpers."""

from utils.mock_data import get_fairness_metrics, get_metric_comparison, get_model_comparison
from utils.mock_prediction import generate_shap_explanation, predict_credit


def format_prediction(label):
    """Compatibility helper for the older app/streamlit_app.py entry point."""
    return "Creditworthy" if int(label) == 1 else "Not Creditworthy"


__all__ = [
    "format_prediction",
    "generate_shap_explanation",
    "get_fairness_metrics",
    "get_metric_comparison",
    "get_model_comparison",
    "predict_credit",
]
