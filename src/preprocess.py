from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .clean_headers import clean_german_credit, extract_sex
    from .config import (
        DEFAULT_DATASET,
        RANDOM_STATE,
        TEST_SIZE,
        get_dataset_config,
        get_model_paths,
    )
except ImportError:
    from clean_headers import clean_german_credit, extract_sex
    from config import (
        DEFAULT_DATASET,
        RANDOM_STATE,
        TEST_SIZE,
        get_dataset_config,
        get_model_paths,
    )


def ensure_german_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the German Credit data has `sex` and `age_group` columns."""
    df = df.copy()
    if "sex" not in df.columns:
        if "personal_status_sex" not in df.columns:
            raise ValueError("Missing sensitive source column: personal_status_sex")
        df["sex"] = df["personal_status_sex"].apply(extract_sex)
    if "age_group" not in df.columns:
        if "age" not in df.columns:
            raise ValueError("Missing age column required for age_group")
        df["age_group"] = np.where(df["age"] < 25, "below_25", "25_and_above")
    return df


def prepare_heloc_dataframe(df: pd.DataFrame, special_codes: list[int]) -> pd.DataFrame:
    """Convert HELOC target and special codes into model-ready columns."""
    df = df.copy()
    if "RiskPerformance" not in df.columns:
        raise ValueError("Missing HELOC target source column: RiskPerformance")

    df["target"] = df["RiskPerformance"].map({"Bad": 0, "Good": 1}).astype(int)
    feature_cols = [col for col in df.columns if col not in {"RiskPerformance", "target"}]

    for column in feature_cols:
        has_special_code = df[column].isin(special_codes)
        if has_special_code.any():
            df[f"{column}_special_code"] = has_special_code.astype(int)

    df[feature_cols] = df[feature_cols].replace(special_codes, np.nan)
    return df


def load_dataset(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    config = get_dataset_config(dataset)
    path = config["data_path"]
    dataset_key = dataset.lower()

    if dataset_key == "german":
        if not path.exists():
            clean_german_credit()
        print(f"[INFO] Loading {config['display_name']} dataset from {path}")
        df = pd.read_csv(path)
        df = ensure_german_sensitive_columns(df)
    else:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        print(f"[INFO] Loading {config['display_name']} dataset from {path}")
        df = pd.read_csv(path)
        df = prepare_heloc_dataframe(df, config["special_missing_codes"])

    target_col = config["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    for col in config["sensitive_cols"]:
        if col not in df.columns:
            raise ValueError(f"Missing sensitive attribute: {col}")
    return df


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No transformers could be built because no feature columns were provided.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess_data(
    dataset: str = DEFAULT_DATASET,
    exclude_sensitive: bool = True,
    save_preprocessor: bool = False,
) -> dict:
    """Prepare dataset-specific arrays and metadata for training and evaluation."""
    dataset_key = dataset.lower()
    config = get_dataset_config(dataset_key)
    df = load_dataset(dataset_key)

    excluded = {config["target_col"]}
    if exclude_sensitive:
        excluded.update(config["excluded_features"])

    numeric_features = [
        col for col in config["numeric_features"] if col in df.columns and col not in excluded
    ]
    categorical_features = [
        col for col in config["categorical_features"] if col in df.columns and col not in excluded
    ]

    # For HELOC, allow dynamically created special-code flags to participate as numeric features.
    if dataset_key == "heloc":
        derived_numeric = [
            col
            for col in df.columns
            if col.endswith("_special_code") and col not in excluded
        ]
        numeric_features = numeric_features + derived_numeric

    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise ValueError("No model features available after preprocessing exclusions.")

    X = df[feature_columns].copy()
    y = df[config["target_col"]].astype(int).copy()

    if config["sensitive_cols"]:
        A = df[config["sensitive_cols"]].copy()
        split = train_test_split(
            X,
            y,
            A,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        X_train, X_test, y_train, y_test, A_train, A_test = split
        A_train = A_train.reset_index(drop=True)
        A_test = A_test.reset_index(drop=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        A_train = pd.DataFrame()
        A_test = pd.DataFrame()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    result = {
        "dataset": dataset_key,
        "display_name": config["display_name"],
        "X_train": X_train_processed.astype(np.float32),
        "X_test": X_test_processed.astype(np.float32),
        "y_train": y_train.to_numpy(dtype=np.int64),
        "y_test": y_test.to_numpy(dtype=np.int64),
        "A_train": A_train,
        "A_test": A_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "raw_feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "X_train_raw": X_train.reset_index(drop=True),
        "X_test_raw": X_test.reset_index(drop=True),
    }

    if save_preprocessor:
        model_paths = get_model_paths(dataset_key)
        joblib.dump(preprocessor, model_paths["preprocessor"])
        joblib.dump(feature_columns, model_paths["raw_feature_columns"])
        joblib.dump(feature_names, model_paths["feature_names"])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a dataset for model training.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset to preprocess: german or heloc")
    args = parser.parse_args()

    data = preprocess_data(dataset=args.dataset, save_preprocessor=True)
    print("[DONE] Preprocessing completed.")
    print(f"[INFO] Dataset: {data['display_name']}")
    print(f"[INFO] X_train shape: {data['X_train'].shape}")
    print(f"[INFO] X_test shape: {data['X_test'].shape}")
