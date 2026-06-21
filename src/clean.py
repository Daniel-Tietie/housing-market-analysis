"""
Cleaning and transformation pipeline for the Ames Housing dataset.
Each function is independent and testable.
"""

import sqlite3
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "housing.db"
PROCESSED_CSV = Path(__file__).parent.parent / "data" / "processed" / "ames_clean.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Columns where NA literally means "None" (no feature), not missing data
NA_MEANS_NONE = [
    "alley", "bsmt_qual", "bsmt_cond", "bsmt_exposure", "bsmtfin_type_1",
    "bsmtfin_type_2", "fireplace_qu", "garage_type", "garage_finish",
    "garage_qual", "garage_cond", "pool_qc", "fence", "misc_feature",
    "mas_vnr_type",
]


def load_raw(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM raw_housing", conn)


def fix_na_means_none(df: pd.DataFrame) -> pd.DataFrame:
    """Replace literal 'NA' strings with 'None' in categorical columns where
    absence of a feature is recorded as NA by the original dataset."""
    for col in NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].replace("NA", "None").fillna("None")
    return df


def impute_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:
    """Lot frontage is missing for ~18% of rows. Impute with the median
    frontage of the same neighborhood — a reasonable proxy since lots in the
    same neighborhood tend to have similar street access."""
    median_by_nbhd = df.groupby("neighborhood")["lot_frontage"].transform("median")
    overall_median = df["lot_frontage"].median()
    df["lot_frontage"] = df["lot_frontage"].fillna(median_by_nbhd).fillna(overall_median)
    return df


def impute_mas_vnr(df: pd.DataFrame) -> pd.DataFrame:
    """Masonry veneer area: 8 rows are missing area but have a veneer type.
    Fill area with 0 for those (consistent with 'None' type rows)."""
    df["mas_vnr_area"] = df["mas_vnr_area"].fillna(0.0)
    return df


def impute_basement_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Basement square footage columns: missing means no basement, fill with 0."""
    bsmt_num_cols = [
        "bsmtfin_sf_1", "bsmtfin_sf_2", "bsmt_unf_sf",
        "total_bsmt_sf", "bsmt_full_bath", "bsmt_half_bath",
    ]
    for col in bsmt_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def impute_garage(df: pd.DataFrame) -> pd.DataFrame:
    """Garage year built: missing for ~81 rows without a garage; fill with 0
    since we engineer a binary has_garage flag downstream."""
    df["garage_yr_blt"] = df["garage_yr_blt"].fillna(0.0)
    df["garage_cars"] = df["garage_cars"].fillna(0.0)
    df["garage_area"] = df["garage_area"].fillna(0.0)
    return df


def impute_electrical(df: pd.DataFrame) -> pd.DataFrame:
    """One row is missing electrical type; fill with the mode."""
    mode = df["electrical"].mode()[0]
    df["electrical"] = df["electrical"].fillna(mode)
    return df


def drop_extreme_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the two documented outliers: partial-sale homes with
    gr_liv_area > 4000 sqft sold at unusually low prices. The dataset author
    (De Cock, 2011) explicitly flags these as outliers to exclude for modeling."""
    before = len(df)
    df = df[~((df["gr_liv_area"] > 4000) & (df["sale_price"] < 200000))]
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} extreme outlier(s) (gr_liv_area > 4000 with low price)")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features that are more analytically useful than raw columns."""
    df["age_at_sale"] = df["yr_sold"] - df["year_built"]
    df["years_since_remodel"] = df["yr_sold"] - df["year_remod_add"]
    df["total_bathrooms"] = (
        df["full_bath"]
        + 0.5 * df["half_bath"]
        + df["bsmt_full_bath"]
        + 0.5 * df["bsmt_half_bath"]
    )
    df["has_garage"] = (df["garage_area"] > 0).astype(int)
    df["has_basement"] = (df["total_bsmt_sf"] > 0).astype(int)
    df["has_fireplace"] = (df["fireplaces"] > 0).astype(int)
    df["central_air"] = (df["central_air"] == "Y").astype(int)
    return df


CLEAN_COLUMNS = [
    "id", "neighborhood", "year_built", "year_remod_add", "yr_sold", "mo_sold",
    "age_at_sale", "years_since_remodel",
    "overall_qual", "overall_cond",
    "gr_liv_area", "total_bsmt_sf", "first_flr_sf", "second_flr_sf",
    "garage_area", "garage_cars", "lot_area",
    "total_bathrooms", "full_bath", "half_bath", "bedroom_abvgr", "tot_rms_abvgrd",
    "fireplaces", "wood_deck_sf", "open_porch_sf", "mas_vnr_area", "bsmtfin_sf_1",
    "central_air", "has_garage", "has_basement", "has_fireplace",
    "bldg_type", "house_style", "ms_zoning", "foundation", "heating_qc",
    "kitchen_qual", "exter_qual", "sale_condition",
    "sale_price",
]


def select_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CLEAN_COLUMNS if c in df.columns]
    return df[cols]


def validate(df: pd.DataFrame) -> None:
    """Raise if any critical fields are null or nonsensical."""
    assert df["sale_price"].isnull().sum() == 0, "Null sale prices remain"
    assert (df["sale_price"] > 0).all(), "Non-positive sale prices"
    assert df["gr_liv_area"].isnull().sum() == 0, "Null gr_liv_area values"
    assert (df["age_at_sale"] >= 0).all(), "Negative age_at_sale"
    logger.info("Validation passed")


def save_processed_csv(df: pd.DataFrame) -> None:
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    logger.info(f"Processed CSV saved: {PROCESSED_CSV}")


def load_into_db(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM clean_housing")
    df.to_sql("clean_housing", conn, if_exists="append", index=False)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM clean_housing").fetchone()[0]
    logger.info(f"Loaded {count} rows into clean_housing")


def run() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = load_raw(conn)
        logger.info(f"Raw rows loaded: {len(df)}")

        df = fix_na_means_none(df)
        df = impute_lot_frontage(df)
        df = impute_mas_vnr(df)
        df = impute_basement_numeric(df)
        df = impute_garage(df)
        df = impute_electrical(df)
        df = drop_extreme_outliers(df)
        df = engineer_features(df)
        df = select_clean_columns(df)

        # Force numeric types for columns that came in as object from SQLite
        int_cols = [
            "year_built", "year_remod_add", "yr_sold", "mo_sold",
            "overall_qual", "overall_cond", "gr_liv_area", "first_flr_sf",
            "second_flr_sf", "lot_area", "full_bath", "half_bath",
            "bedroom_abvgr", "tot_rms_abvgrd", "fireplaces",
            "wood_deck_sf", "open_porch_sf",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        validate(df)
        save_processed_csv(df)
        load_into_db(df, conn)
    finally:
        conn.close()

    logger.info(f"Cleaning complete. Final row count: {len(df)}")
    return df


if __name__ == "__main__":
    run()
