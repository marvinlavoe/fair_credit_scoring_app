import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_german_credit():
    """Load German Credit dataset from UCI using ucimlrepo."""
    german_credit = fetch_ucirepo(id=144)
    X = german_credit.data.features.copy()
    y = german_credit.data.targets.copy()

    # Convert target to a clean binary label if needed
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, y
