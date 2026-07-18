from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

try:
    from .config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from .fairness import evaluate_fairness, save_fairness_results
    from .preprocess import preprocess_data
except ImportError:
    from config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from fairness import evaluate_fairness, save_fairness_results
    from preprocess import preprocess_data


def _load_tabnet_model(path):
    model = TabNetClassifier()
    model.load_model(str(path))
    return model


def _load_saved_metrics(dataset: str) -> dict:
    metrics_path = get_output_paths(dataset)["metrics"]
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _predict_tabnet_with_threshold(model: TabNetClassifier, X_test, threshold: float) -> np.ndarray:
    y_proba = model.predict_proba(X_test)[:, 1]
    return (y_proba >= threshold).astype(int)


def evaluate_saved_models(dataset: str = DEFAULT_DATASET) -> dict:
    print(f"[INFO] Evaluating saved model fairness for {dataset}...")
    data = preprocess_data(dataset=dataset, save_preprocessor=False)
    model_paths = get_model_paths(dataset)
    output_paths = get_output_paths(dataset)
    saved_metrics = _load_saved_metrics(dataset)
    results = {}

    logistic_path = model_paths["logistic_regression"]
    if logistic_path.exists():
        model = joblib.load(logistic_path)
        results["logistic_regression"] = evaluate_fairness(
            data["y_test"], model.predict(data["X_test"]), data["A_test"]
        )

    tabnet_paths = {
        "tabnet_baseline": model_paths["tabnet_baseline"],
        "tabnet_debiased": model_paths["tabnet_debiased"],
    }
    for name, path in tabnet_paths.items():
        if path.exists():
            model = _load_tabnet_model(path)
            threshold = float(saved_metrics.get(name, {}).get("decision_threshold", 0.5))
            results[name] = evaluate_fairness(
                data["y_test"],
                _predict_tabnet_with_threshold(model, data["X_test"], threshold),
                data["A_test"],
            )

    save_fairness_results(results, output_paths["fairness_results"])
    return results


def generate_model_comparison(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    output_paths = get_output_paths(dataset)
    metrics_path = output_paths["metrics"]
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    fairness_results = (
        json.loads(output_paths["fairness_results"].read_text(encoding="utf-8"))
        if output_paths["fairness_results"].exists()
        else {}
    )

    rows = []
    labels = {
        "logistic_regression": "Logistic Regression",
        "tabnet_baseline": "TabNet Baseline",
        "tabnet_debiased": "TabNet + Fairness-Aware Reweighting",
    }
    for key, label in labels.items():
        if key not in metrics:
            continue

        model_fairness = fairness_results.get(key, [])
        primary_fairness = model_fairness[0] if model_fairness else {}
        rows.append(
            {
                "Model": label,
                "AUC-ROC": metrics[key].get("auc_roc"),
                "Weighted F1": metrics[key].get("weighted_f1"),
                "Demographic Parity Difference": primary_fairness.get("demographic_parity_difference"),
                "Equalized Odds Difference": primary_fairness.get("equalized_odds_difference"),
                "Disparate Impact Ratio": primary_fairness.get("disparate_impact_ratio"),
                "Accuracy": metrics[key].get("accuracy"),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved models.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset to evaluate: german or heloc")
    args = parser.parse_args()
    evaluate_saved_models(dataset=args.dataset)
    print(generate_model_comparison(dataset=args.dataset))
