import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(X, y, protected_column=None, test_size=0.2, random_state=42):
    """Encode categorical columns, scale numeric columns, and split dataset."""
    X_processed = X.copy()
    encoders = {}

    for col in X_processed.columns:
        if X_processed[col].dtype == "object" or str(X_processed[col].dtype).startswith("category"):
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            encoders[col] = le

    y_encoder = LabelEncoder()
    y_processed = y_encoder.fit_transform(y.astype(str))

    protected = None
    if protected_column and protected_column in X_processed.columns:
        protected = X_processed[protected_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=test_size, random_state=random_state, stratify=y_processed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    protected_train = None
    protected_test = None
    if protected is not None:
        protected_train = protected.loc[X_train.index]
        protected_test = protected.loc[X_test.index]

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X_processed.columns),
        "scaler": scaler,
        "encoders": encoders,
        "y_encoder": y_encoder,
        "protected_train": protected_train,
        "protected_test": protected_test,
        "raw_train_index": X_train.index,
        "raw_test_index": X_test.index,
        "X_processed": X_processed,
    }
