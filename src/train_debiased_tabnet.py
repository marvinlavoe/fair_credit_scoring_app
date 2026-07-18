from __future__ import annotations

import argparse
import json
import os

import pandas as pd

try:
    from .config import DEFAULT_DATASET, get_dataset_config, get_model_paths, get_output_paths
    from .fairness import compute_group_reweighting, evaluate_fairness, save_fairness_results
    from .preprocess import preprocess_data
    from .train_tabnet import build_tabnet, evaluate_tabnet
except ImportError:
    from config import DEFAULT_DATASET, get_dataset_config, get_model_paths, get_output_paths
    from fairness import compute_group_reweighting, evaluate_fairness, save_fairness_results
    from preprocess import preprocess_data
    from train_tabnet import build_tabnet, evaluate_tabnet


def _load_metrics(dataset: str) -> dict:
    path = get_output_paths(dataset)["metrics"]
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_metrics(dataset: str, metrics: dict) -> None:
    path = get_output_paths(dataset)["metrics"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _primary_fairness_summary(fairness_results: dict, model_key: str) -> dict:
    model_fairness = fairness_results.get(model_key, [])
    if not model_fairness:
        return {
            "Demographic Parity Difference": None,
            "Equalized Odds Difference": None,
            "Disparate Impact Ratio": None,
        }

    preferred_attr = "sex"
    first_item = model_fairness[0]
    result = next(
        (item for item in model_fairness if item.get("sensitive_attribute") == preferred_attr),
        first_item,
    )
    return {
        "Demographic Parity Difference": result.get("demographic_parity_difference"),
        "Equalized Odds Difference": result.get("equalized_odds_difference"),
        "Disparate Impact Ratio": result.get("disparate_impact_ratio"),
    }


def save_model_comparison(dataset: str, metrics: dict) -> pd.DataFrame:
    rows = []
    labels = {
        "logistic_regression": "Logistic Regression",
        "tabnet_baseline": "TabNet Baseline",
        "tabnet_debiased": "TabNet + Fairness-Aware Reweighting",
    }
    output_paths = get_output_paths(dataset)
    fairness_results = (
        json.loads(output_paths["fairness_results"].read_text(encoding="utf-8"))
        if output_paths["fairness_results"].exists()
        else {}
    )

    for key, label in labels.items():
        if key not in metrics:
            continue
        row = metrics[key]
        fairness = _primary_fairness_summary(fairness_results, key)
        rows.append(
            {
                "Model": label,
                "AUC-ROC": row.get("auc_roc"),
                "Weighted F1": row.get("weighted_f1"),
                **fairness,
                "Accuracy": row.get("accuracy"),
                "Comment": (
                    "Fairness-aware sample reweighting by target and sex."
                    if key == "tabnet_debiased"
                    else "Reference model."
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_paths["model_comparison"], index=False)
    return comparison


def train_debiased_tabnet(
    dataset: str = DEFAULT_DATASET,
    max_epochs: int | None = None,
    verbose: int = 1,
) -> dict:
    dataset_config = get_dataset_config(dataset)
    if not dataset_config["sensitive_cols"]:
        raise ValueError(
            f"Dataset '{dataset}' does not define sensitive attributes, so fairness-aware "
            "reweighting cannot be applied."
        )

    print(f"[INFO] Training TabNet with fairness-aware reweighting for {dataset}...")
    data = preprocess_data(dataset=dataset, save_preprocessor=True)
    model_paths = get_model_paths(dataset)
    output_paths = get_output_paths(dataset)

    sample_weights = compute_group_reweighting(data["y_train"], data["A_train"]["sex"])
    epochs = max_epochs or int(os.getenv("FAIR_CREDIT_TABNET_EPOCHS", "100"))

    model = build_tabnet(verbose=verbose)
    model.fit(
        data["X_train"],
        data["y_train"],
        eval_set=[(data["X_test"], data["y_test"])],
        eval_name=["test"],
        eval_metric=["auc"],
        max_epochs=epochs,
        patience=15,
        batch_size=256,
        virtual_batch_size=64,
        weights=sample_weights,
    )

    metrics, y_pred = evaluate_tabnet(model, data["X_test"], data["y_test"])
    fairness = evaluate_fairness(data["y_test"], y_pred, data["A_test"])

    model.save_model(str(model_paths["tabnet_debiased"]).replace(".zip", ""))

    all_metrics = _load_metrics(dataset)
    all_metrics["tabnet_debiased"] = metrics
    _save_metrics(dataset, all_metrics)

    fairness_results = {}
    if output_paths["fairness_results"].exists():
        fairness_results = json.loads(output_paths["fairness_results"].read_text(encoding="utf-8"))
    fairness_results["tabnet_debiased"] = fairness
    save_fairness_results(fairness_results, output_paths["fairness_results"])
    save_model_comparison(dataset, all_metrics)

    print("[DONE] Debiased TabNet model saved.")
    return {"metrics": metrics, "fairness": fairness}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train debiased TabNet.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset to train: german or heloc")
    args = parser.parse_args()
    print(json.dumps(train_debiased_tabnet(dataset=args.dataset)["metrics"], indent=2))
