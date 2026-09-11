from __future__ import annotations

import argparse
import json

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

try:
    from .config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from .explainability import (
        generate_shap_summary_plot,
        generate_shap_values,
        generate_shap_waterfall_plot,
        get_global_shap_importance,
        get_top_feature_contributions,
    )
    from .preprocess import preprocess_data
except ImportError:
    from config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from explainability import (
        generate_shap_summary_plot,
        generate_shap_values,
        generate_shap_waterfall_plot,
        get_global_shap_importance,
        get_top_feature_contributions,
    )
    from preprocess import preprocess_data


def generate_backend_shap_artifacts(dataset: str = DEFAULT_DATASET) -> dict:
    print(f"[INFO] Generating SHAP explanation artifacts for {dataset}...")
    model_paths = get_model_paths(dataset)
    output_paths = get_output_paths(dataset)
    model_path = model_paths["tabnet_debiased"]
    if not model_path.exists():
        model_path = model_paths["tabnet_baseline"]
    if not model_path.exists():
        raise FileNotFoundError(f"No TabNet model found for SHAP artifact generation: {dataset}")

    data = preprocess_data(dataset=dataset, save_preprocessor=False)
    model = TabNetClassifier()
    model.load_model(str(model_path))

    X_background = data["X_train"][:20]
    X_sample = data["X_test"]
    shap_values = generate_shap_values(
        model,
        X_background,
        X_sample,
        background_size=20,
        nsamples=80,
    )

    sample_probability = float(model.predict_proba(X_sample[:1])[0, 1])
    sample_values = np.asarray(shap_values[0] if np.asarray(shap_values).ndim > 1 else shap_values)
    sample_contributions = get_top_feature_contributions(
        sample_values,
        data["feature_names"],
        top_n=10,
    )
    positive = [item for item in sample_contributions if item["value"] > 0]
    negative = [item for item in sample_contributions if item["value"] < 0]
    global_importance = get_global_shap_importance(
        shap_values,
        data["feature_names"],
        top_n=10,
    )

    summary_path = generate_shap_summary_plot(
        shap_values,
        X_sample,
        data["feature_names"],
        output_paths["shap_summary"],
    )
    waterfall_path = generate_shap_waterfall_plot(
        sample_values,
        data["feature_names"],
        output_paths["shap_waterfall"],
    )

    report = {
        "dataset": dataset,
        "model": "tabnet_debiased" if model_path == model_paths["tabnet_debiased"] else "tabnet_baseline",
        "test_records_explained": int(len(X_sample)),
        "background_records": int(len(X_background)),
        "nsamples": 80,
        "sample_applicant": {
            "creditworthy_probability": sample_probability,
            "top_positive_contributions": positive,
            "top_negative_contributions": negative,
        },
        "global_mean_absolute_shap": global_importance,
        "summary_plot": str(summary_path),
        "waterfall_plot": str(waterfall_path),
    }
    output_paths["shap_report"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[DONE] SHAP summary saved to {summary_path}")
    print(f"[INFO] Sample creditworthy probability: {sample_probability:.4f}")
    print(f"[INFO] SHAP report saved to {output_paths['shap_report']}")
    return {
        "summary_path": str(summary_path),
        "waterfall_path": str(waterfall_path),
        "report_path": str(output_paths["shap_report"]),
        "sample_probability": sample_probability,
        "top_positive_contributions": positive,
        "top_negative_contributions": negative,
        "global_mean_absolute_shap": global_importance,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP artifacts for a trained dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset to explain: german or heloc")
    args = parser.parse_args()
    generate_backend_shap_artifacts(dataset=args.dataset)
