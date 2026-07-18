from __future__ import annotations

import argparse
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from .config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from .fairness import evaluate_fairness, save_fairness_results
    from .preprocess import preprocess_data
except ImportError:
    from config import DEFAULT_DATASET, get_model_paths, get_output_paths
    from fairness import evaluate_fairness, save_fairness_results
    from preprocess import preprocess_data


def _load_metrics(dataset: str) -> dict:
    path = get_output_paths(dataset)["metrics"]
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_metrics(dataset: str, metrics: dict) -> None:
    path = get_output_paths(dataset)["metrics"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
    }


def train_baseline(dataset: str = DEFAULT_DATASET) -> dict:
    print(f"[INFO] Training baseline Logistic Regression model for {dataset}...")
    data = preprocess_data(dataset=dataset, save_preprocessor=True)
    model_paths = get_model_paths(dataset)
    output_paths = get_output_paths(dataset)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])
    metrics = evaluate_classifier(model, data["X_test"], data["y_test"])
    fairness = evaluate_fairness(data["y_test"], y_pred, data["A_test"])

    joblib.dump(model, model_paths["logistic_regression"])
    joblib.dump(data["preprocessor"], model_paths["preprocessor"])

    all_metrics = _load_metrics(dataset)
    all_metrics["logistic_regression"] = metrics
    _save_metrics(dataset, all_metrics)

    fairness_payload = {"logistic_regression": fairness}
    save_fairness_results(fairness_payload, output_paths["fairness_results"])
    print("[DONE] Baseline model saved.")
    return {"metrics": metrics, "fairness": fairness}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression baseline.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset to train: german or heloc")
    args = parser.parse_args()
    print(json.dumps(train_baseline(dataset=args.dataset)["metrics"], indent=2))
