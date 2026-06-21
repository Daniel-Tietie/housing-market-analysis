"""
Feature engineering for the ML modeling phase.
Prepares the clean_housing table data for scikit-learn.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DB_PATH = Path(__file__).parent.parent / "data" / "housing.db"

NUMERIC_FEATURES = [
    "overall_qual", "overall_cond", "gr_liv_area", "total_bsmt_sf",
    "first_flr_sf", "second_flr_sf", "garage_area", "garage_cars",
    "lot_area", "total_bathrooms", "bedroom_abvgr", "tot_rms_abvgrd",
    "fireplaces", "wood_deck_sf", "open_porch_sf", "mas_vnr_area",
    "bsmtfin_sf_1", "age_at_sale", "years_since_remodel",
    "central_air", "has_garage", "has_basement", "has_fireplace",
]

CATEGORICAL_FEATURES = [
    "neighborhood", "bldg_type", "house_style", "ms_zoning",
    "foundation", "heating_qc", "kitchen_qual", "exter_qual", "sale_condition",
]

TARGET = "sale_price"


def load_clean(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM clean_housing", conn)


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical features. Returns the modified DataFrame and
    a dict of fitted encoders for inverse-transform in evaluation."""
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna("Unknown")
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (feature matrix) and y (target) ready for scikit-learn."""
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    cols = [c for c in all_features if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET].copy()

    # Fill any remaining nulls with column median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    return X, y


def run() -> tuple[pd.DataFrame, pd.Series, dict]:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = load_clean(conn)
    finally:
        conn.close()

    df, encoders = encode_categoricals(df)
    X, y = prepare_features(df)
    return X, y, encoders


if __name__ == "__main__":
    X, y, _ = run()
    print(f"Feature matrix: {X.shape}")
    print(f"Target: {y.describe()}")
