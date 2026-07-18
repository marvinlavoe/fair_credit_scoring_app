from __future__ import annotations

import argparse
import json

try:
    from .clean_headers import clean_german_credit
    from .config import (
        DEFAULT_DATASET,
        MODEL_DIR,
        OUTPUT_DIR,
        RANDOM_STATE,
        SUPPORTED_DATASETS,
        TEST_SIZE,
        get_dataset_config,
        get_model_paths,
        get_output_paths,
    )
    from .evaluate import evaluate_saved_models, generate_model_comparison
    from .generate_shap_artifacts import generate_backend_shap_artifacts
    from .train_baseline import train_baseline
    from .train_debiased_tabnet import train_debiased_tabnet
    from .train_tabnet import train_tabnet_baseline
except ImportError:
    from clean_headers import clean_german_credit
    from config import (
        DEFAULT_DATASET,
        MODEL_DIR,
        OUTPUT_DIR,
        RANDOM_STATE,
        SUPPORTED_DATASETS,
        TEST_SIZE,
        get_dataset_config,
        get_model_paths,
        get_output_paths,
    )
    from evaluate import evaluate_saved_models, generate_model_comparison
    from generate_shap_artifacts import generate_backend_shap_artifacts
    from train_baseline import train_baseline
    from train_debiased_tabnet import train_debiased_tabnet
    from train_tabnet import train_tabnet_baseline


def _run_single_dataset(dataset: str) -> dict:
    dataset_config = get_dataset_config(dataset)
    output_paths = get_output_paths(dataset)
    model_paths = get_model_paths(dataset)

    if dataset == "german":
        print("[INFO] Cleaning headers...")
        clean_german_credit()

    print(f"[INFO] Training baseline model for {dataset}...")
    baseline = train_baseline(dataset=dataset)

    print(f"[INFO] Training TabNet model for {dataset}...")
    tabnet = train_tabnet_baseline(dataset=dataset)

    debiased = None
    fairness_results = {}
    shap_artifacts = {}

    if dataset_config["sensitive_cols"]:
        print(f"[INFO] Training fairness-aware reweighted TabNet model for {dataset}...")
        debiased = train_debiased_tabnet(dataset=dataset)

        print(f"[INFO] Evaluating fairness for {dataset}...")
        fairness_results = evaluate_saved_models(dataset=dataset)
    else:
        print(
            f"[INFO] Dataset '{dataset}' has no configured sensitive attributes. "
            "Skipping fairness-aware reweighting and fairness evaluation."
        )

    print(f"[INFO] Saving model comparison for {dataset}...")
    comparison = generate_model_comparison(dataset=dataset)

    print(f"[INFO] Generating SHAP summary for {dataset}...")
    shap_artifacts = generate_backend_shap_artifacts(dataset=dataset)

    metadata = {
        "dataset": dataset_config["display_name"],
        "dataset_key": dataset,
        "target_col": dataset_config["target_col"],
        "sensitive_cols": dataset_config["sensitive_cols"],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "debiasing_method": (
            "fairness-aware reweighting"
            if dataset_config["sensitive_cols"]
            else "not applied"
        ),
        "outputs": {
            "metrics": str(output_paths["metrics"]),
            "fairness_results": str(output_paths["fairness_results"]),
            "model_comparison": str(output_paths["model_comparison"]),
            "shap_summary": shap_artifacts["summary_path"],
            "shap_waterfall": shap_artifacts["waterfall_path"],
        },
    }
    model_paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[DONE] Training pipeline completed successfully for {dataset}.")
    print(
        json.dumps(
            {
                "dataset": dataset,
                "logistic_regression": baseline["metrics"],
                "tabnet_baseline": tabnet["metrics"],
                "tabnet_debiased": debiased["metrics"] if debiased else None,
                "fairness_models": list(fairness_results.keys()),
                "comparison_rows": len(comparison),
            },
            indent=2,
        )
    )
    return {
        "dataset": dataset,
        "baseline": baseline,
        "tabnet": tabnet,
        "debiased": debiased,
        "fairness_results": fairness_results,
        "comparison": comparison,
        "shap_artifacts": shap_artifacts,
    }


def main(dataset: str = DEFAULT_DATASET) -> None:
    datasets = list(SUPPORTED_DATASETS) if dataset == "both" else [dataset]
    for dataset_key in datasets:
        _run_single_dataset(dataset_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full training workflow.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset to train: german, heloc, or both",
    )
    args = parser.parse_args()
    main(dataset=args.dataset)
