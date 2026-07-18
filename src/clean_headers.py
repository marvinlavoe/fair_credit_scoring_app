from pathlib import Path

import pandas as pd

try:
    from .config import DATA_PATH, RAW_DATA_PATH
except ImportError:
    from config import DATA_PATH, RAW_DATA_PATH


COLUMN_MAPPING = {
    "laufkont": "checking_account_status",
    "laufzeit": "loan_duration_months",
    "moral": "credit_history",
    "verw": "loan_purpose",
    "hoehe": "loan_amount",
    "sparkont": "savings_account_status",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "residence_duration",
    "verm": "property_type",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing_status",
    "bishkred": "existing_credits",
    "beruf": "job_type",
    "pers": "number_of_dependents",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "target"
}


SEX_MAPPING = {
    "A91": "male",
    "A92": "female",
    "A93": "male",
    "A94": "male",
    "A95": "female",
    1: "male",
    2: "female",
    3: "male",
    4: "male",
    5: "female",
}


def _load_source_data(raw_path: Path = RAW_DATA_PATH, clean_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw German Credit if present, otherwise enhance the existing clean file."""
    if raw_path.exists():
        print(f"[INFO] Loading raw dataset from {raw_path}")
        return pd.read_csv(raw_path).rename(columns=COLUMN_MAPPING)
    if clean_path.exists():
        print(f"[INFO] Raw dataset not found. Using existing cleaned file: {clean_path}")
        return pd.read_csv(clean_path)
    raise FileNotFoundError(
        f"Missing dataset. Expected either {raw_path} or {clean_path}."
    )


def _normalise_target(series: pd.Series) -> pd.Series:
    """Convert target to 1=creditworthy, 0=not creditworthy."""
    values = set(series.dropna().unique().tolist())
    if values.issubset({0, 1}):
        return series.astype(int)
    return series.map({1: 1, 2: 0}).astype(int)


def extract_sex(value) -> str:
    """Extract sex from German Credit personal status/sex coding."""
    return SEX_MAPPING.get(value, SEX_MAPPING.get(str(value), "unknown"))


def clean_german_credit(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = DATA_PATH,
) -> pd.DataFrame:
    print("[INFO] Cleaning German Credit headers and sensitive attributes...")
    df = _load_source_data(raw_path=raw_path, clean_path=output_path)

    missing_columns = {"target", "personal_status_sex", "age"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {sorted(missing_columns)}")

    df["target"] = _normalise_target(df["target"])
    df["sex"] = df["personal_status_sex"].apply(extract_sex)
    df["age_group"] = df["age"].apply(lambda age: "below_25" if age < 25 else "25_and_above")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DONE] Headers converted and saved as {output_path}")
    print(df.head())
    return df


if __name__ == "__main__":
    clean_german_credit()
