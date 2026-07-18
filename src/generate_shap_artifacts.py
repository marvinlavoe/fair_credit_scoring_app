from __future__ import annotations

import argparse

from pytorch_tabnet.tab_model import TabNetClassifier

try:
    from .config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from .explainability import (
        generate_shap_summary_plot,
        generate_shap_values,
        generate_shap_waterfall_plot,
    )
    from .preprocess import preprocess_data
except ImportError:
    from config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from explainability import (
        generate_shap_summary_plot,
        generate_shap_values,
        generate_shap_waterfall_plot,
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
    X_sample = data["X_test"][:3]
    shap_values = generate_shap_values(
        model,
        X_background,
        X_sample,
        background_size=20,
        nsamples=80,
    )

    summary_path = generate_shap_summary_plot(
        shap_values,
        X_sample,
        data["feature_names"],
        output_paths["shap_summary"],
    )
    waterfall_path = generate_shap_waterfall_plot(
        shap_values[0],
        data["feature_names"],
        output_paths["shap_waterfall"],
    )

    print(f"[DONE] SHAP summary saved to {summary_path}")
    return {"summary_path": str(summary_path), "waterfall_path": str(waterfall_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP artifacts for a trained dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset to explain: german or heloc")
    args = parser.parse_args()
    generate_backend_shap_artifacts(dataset=args.dataset)
