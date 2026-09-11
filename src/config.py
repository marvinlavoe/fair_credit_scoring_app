from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "german_credit_data.csv"
DATA_PATH = PROJECT_ROOT / "data" / "german_clean.csv"
HELOC_DATA_PATH = PROJECT_ROOT / "data" / "heloc_dataset_v1 (1).csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
#DEFAULT_DATASET = "german"
DEFAULT_DATASET = "german"
SUPPORTED_DATASETS = ("german", "heloc")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GERMAN_NUMERIC_FEATURES = [
    "loan_duration_months",
    "loan_amount",
    "installment_rate",
    "residence_duration",
    "age",
    "existing_credits",
    "number_of_dependents",
]

GERMAN_CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "loan_purpose",
    "savings_account_status",
    "employment_duration",
    "other_debtors",
    "property_type",
    "other_installment_plans",
    "housing_status",
    "job_type",
    "telephone",
    "foreign_worker",
]

HELOC_NUMERIC_FEATURES = [
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]

TARGET_COL = "target"
SENSITIVE_COLS = ["sex", "age_group"]

DATASET_CONFIGS = {
    "german": {
        "display_name": "German Credit",
        "data_path": DATA_PATH,
        "raw_data_path": RAW_DATA_PATH,
        "target_col": TARGET_COL,
        "sensitive_cols": SENSITIVE_COLS,
        "numeric_features": GERMAN_NUMERIC_FEATURES,
        "categorical_features": GERMAN_CATEGORICAL_FEATURES,
        "excluded_features": [TARGET_COL, "sex", "age_group", "personal_status_sex"],
        "special_missing_codes": [],
    },
    "heloc": {
        "display_name": "HELOC",
        "data_path": HELOC_DATA_PATH,
        "raw_data_path": HELOC_DATA_PATH,
        "target_col": TARGET_COL,
        "sensitive_cols": [],
        "numeric_features": HELOC_NUMERIC_FEATURES,
        "categorical_features": [],
        "excluded_features": [TARGET_COL, "RiskPerformance"],
        "special_missing_codes": [-9, -8, -7],
    },
}


def get_dataset_config(dataset: str = DEFAULT_DATASET) -> dict:
    dataset_key = dataset.lower()
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected one of {SUPPORTED_DATASETS}."
        )
    return DATASET_CONFIGS[dataset_key]


def get_model_paths(dataset: str = DEFAULT_DATASET) -> dict:
    dataset_key = dataset.lower()
    if dataset_key == "german":
        return {
            "logistic_regression": MODEL_DIR / "logistic_regression.pkl",
            "tabnet_baseline": MODEL_DIR / "tabnet_baseline.zip",
            "tabnet_debiased": MODEL_DIR / "tabnet_debiased.zip",
            "preprocessor": MODEL_DIR / "preprocessor.pkl",
            "feature_names": MODEL_DIR / "feature_names.pkl",
            "raw_feature_columns": MODEL_DIR / "raw_feature_columns.pkl",
            "metadata": MODEL_DIR / "metadata.json",
        }
    return {
        "logistic_regression": MODEL_DIR / f"{dataset_key}_logistic_regression.pkl",
        "tabnet_baseline": MODEL_DIR / f"{dataset_key}_tabnet_baseline.zip",
        "tabnet_debiased": MODEL_DIR / f"{dataset_key}_tabnet_debiased.zip",
        "preprocessor": MODEL_DIR / f"{dataset_key}_preprocessor.pkl",
        "feature_names": MODEL_DIR / f"{dataset_key}_feature_names.pkl",
        "raw_feature_columns": MODEL_DIR / f"{dataset_key}_raw_feature_columns.pkl",
        "metadata": MODEL_DIR / f"{dataset_key}_metadata.json",
    }


def get_output_paths(dataset: str = DEFAULT_DATASET) -> dict:
    dataset_key = dataset.lower()
    if dataset_key == "german":
        return {
            "metrics": OUTPUT_DIR / "metrics.json",
            "fairness_results": OUTPUT_DIR / "fairness_results.json",
            "model_comparison": OUTPUT_DIR / "model_comparison.csv",
            "shap_summary": OUTPUT_DIR / "shap_summary.png",
            "shap_waterfall": OUTPUT_DIR / "shap_waterfall.png",
            "shap_report": OUTPUT_DIR / "shap_report.json",
        }
    return {
        "metrics": OUTPUT_DIR / f"{dataset_key}_metrics.json",
        "fairness_results": OUTPUT_DIR / f"{dataset_key}_fairness_results.json",
        "model_comparison": OUTPUT_DIR / f"{dataset_key}_model_comparison.csv",
        "shap_summary": OUTPUT_DIR / f"{dataset_key}_shap_summary.png",
        "shap_waterfall": OUTPUT_DIR / f"{dataset_key}_shap_waterfall.png",
        "shap_report": OUTPUT_DIR / f"{dataset_key}_shap_report.json",
    }
