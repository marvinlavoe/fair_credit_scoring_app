from __future__ import annotations

import os
from pathlib import Path

try:
    from .config import OUTPUT_DIR, PROJECT_ROOT, RANDOM_STATE
except ImportError:
    from config import OUTPUT_DIR, PROJECT_ROOT, RANDOM_STATE

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _positive_class_predict_fn(model):
    def predict_positive(X):
        proba = model.predict_proba(np.asarray(X, dtype=np.float32))
        return proba[:, 1]

    return predict_positive


def generate_shap_values(
    model,
    X_background,
    X_instance,
    background_size: int = 50,
    nsamples: int = 100,
):
    """Generate SHAP values with KernelExplainer for TabNet-compatible models."""
    if X_background is None or len(X_background) == 0:
        raise ValueError("X_background is required for SHAP explanations.")
    if X_instance is None or len(X_instance) == 0:
        raise ValueError("X_instance is required for SHAP explanations.")

    background = np.asarray(X_background, dtype=np.float32)
    instance = np.asarray(X_instance, dtype=np.float32)

    if len(background) > background_size:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(background), size=background_size, replace=False)
        background = background[sample_idx]

    explainer = shap.KernelExplainer(_positive_class_predict_fn(model), background)
    shap_values = explainer.shap_values(instance, nsamples=nsamples)
    if shap_values is None or len(shap_values) == 0:
        raise ValueError("Empty SHAP output generated.")
    return shap_values


def get_top_feature_contributions(shap_values, feature_names, top_n: int = 10) -> list[dict]:
    values = np.asarray(shap_values)
    if values.ndim > 1:
        values = values[0]
    contributions = pd.DataFrame({"feature": feature_names, "value": values})
    contributions["abs_value"] = contributions["value"].abs()
    contributions = contributions.sort_values("abs_value", ascending=False).head(top_n)
    return [
        {
            "feature": row.feature,
            "value": float(row.value),
            "direction": "positive" if row.value >= 0 else "negative",
        }
        for row in contributions.itertuples(index=False)
    ]


def generate_shap_waterfall_plot(
    shap_values,
    feature_names,
    output_path: Path = OUTPUT_DIR / "shap_waterfall.png",
) -> Path:
    values = np.asarray(shap_values)
    if values.ndim > 1:
        values = values[0]
    series = pd.Series(values, index=feature_names).sort_values(key=np.abs).tail(12)

    colors = ["#11845b" if value >= 0 else "#b42318" for value in series]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(series.index, series.values, color=colors)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_title("SHAP Feature Contributions")
    ax.set_xlabel("Contribution to creditworthy probability")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def generate_shap_summary_plot(
    shap_values,
    X_sample,
    feature_names,
    output_path: Path = OUTPUT_DIR / "shap_summary.png",
) -> Path:
    values = np.asarray(shap_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        values,
        pd.DataFrame(X_sample, columns=feature_names),
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


# Backward-compatible helper used by earlier project code.
def explain_prediction(model_predict_fn, background_data, input_data, feature_names):
    explainer = shap.KernelExplainer(model_predict_fn, background_data)
    shap_values = explainer.shap_values(input_data)
    return shap_values


def to_dataframe(values, feature_names):
    return pd.DataFrame(values, columns=feature_names)
