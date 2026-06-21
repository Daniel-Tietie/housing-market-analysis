"""
Data ingestion for the housing market analysis project.

Dataset: Ames Housing (Ames, Iowa, USA), compiled by Dean De Cock (2011).
Source:  OpenML dataset ID 42165 (https://www.openml.org/d/42165)
         Fetched via scikit-learn's fetch_openml utility — no manual download needed.
License: Public domain / academic use.

Why not a Canadian dataset: CREA and CMHC publish aggregate price indices, not
individual property records. Individual-record data from CHSP requires bulk CSV
downloads that are not stable programmatic sources. Ames provides 80 property
features over 2006-2010, which is sufficient for meaningful ML modeling.
"""

import sqlite3
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAW_CSV = Path(__file__).parent.parent / "data" / "raw" / "ames_housing.csv"
DB_PATH = Path(__file__).parent.parent / "data" / "housing.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "MSSubClass":   "ms_sub_class",
    "MSZoning":     "ms_zoning",
    "LotFrontage":  "lot_frontage",
    "LotArea":      "lot_area",
    "Street":       "street",
    "Alley":        "alley",
    "LotShape":     "lot_shape",
    "LandContour":  "land_contour",
    "Utilities":    "utilities",
    "LotConfig":    "lot_config",
    "LandSlope":    "land_slope",
    "Neighborhood": "neighborhood",
    "Condition1":   "condition_1",
    "Condition2":   "condition_2",
    "BldgType":     "bldg_type",
    "HouseStyle":   "house_style",
    "OverallQual":  "overall_qual",
    "OverallCond":  "overall_cond",
    "YearBuilt":    "year_built",
    "YearRemodAdd": "year_remod_add",
    "RoofStyle":    "roof_style",
    "RoofMatl":     "roof_matl",
    "Exterior1st":  "exterior_1st",
    "Exterior2nd":  "exterior_2nd",
    "MasVnrType":   "mas_vnr_type",
    "MasVnrArea":   "mas_vnr_area",
    "ExterQual":    "exter_qual",
    "ExterCond":    "exter_cond",
    "Foundation":   "foundation",
    "BsmtQual":     "bsmt_qual",
    "BsmtCond":     "bsmt_cond",
    "BsmtExposure": "bsmt_exposure",
    "BsmtFinType1": "bsmtfin_type_1",
    "BsmtFinSF1":   "bsmtfin_sf_1",
    "BsmtFinType2": "bsmtfin_type_2",
    "BsmtFinSF2":   "bsmtfin_sf_2",
    "BsmtUnfSF":    "bsmt_unf_sf",
    "TotalBsmtSF":  "total_bsmt_sf",
    "Heating":      "heating",
    "HeatingQC":    "heating_qc",
    "CentralAir":   "central_air",
    "Electrical":   "electrical",
    "1stFlrSF":     "first_flr_sf",
    "2ndFlrSF":     "second_flr_sf",
    "LowQualFinSF": "low_qual_fin_sf",
    "GrLivArea":    "gr_liv_area",
    "BsmtFullBath": "bsmt_full_bath",
    "BsmtHalfBath": "bsmt_half_bath",
    "FullBath":     "full_bath",
    "HalfBath":     "half_bath",
    "BedroomAbvGr": "bedroom_abvgr",
    "KitchenAbvGr": "kitchen_abvgr",
    "KitchenQual":  "kitchen_qual",
    "TotRmsAbvGrd": "tot_rms_abvgrd",
    "Functional":   "functional",
    "Fireplaces":   "fireplaces",
    "FireplaceQu":  "fireplace_qu",
    "GarageType":   "garage_type",
    "GarageYrBlt":  "garage_yr_blt",
    "GarageFinish": "garage_finish",
    "GarageCars":   "garage_cars",
    "GarageArea":   "garage_area",
    "GarageQual":   "garage_qual",
    "GarageCond":   "garage_cond",
    "PavedDrive":   "paved_drive",
    "WoodDeckSF":   "wood_deck_sf",
    "OpenPorchSF":  "open_porch_sf",
    "EnclosedPorch":"enclosed_porch",
    "3SsnPorch":    "three_ssn_porch",
    "ScreenPorch":  "screen_porch",
    "PoolArea":     "pool_area",
    "PoolQC":       "pool_qc",
    "Fence":        "fence",
    "MiscFeature":  "misc_feature",
    "MiscVal":      "misc_val",
    "MoSold":       "mo_sold",
    "YrSold":       "yr_sold",
    "SaleType":     "sale_type",
    "SaleCondition":"sale_condition",
    "SalePrice":    "sale_price",
}


def fetch_raw() -> pd.DataFrame:
    """Download Ames Housing from OpenML and return as a DataFrame."""
    logger.info("Fetching Ames Housing dataset from OpenML ...")
    dataset = fetch_openml(data_id=42165, as_frame=True, parser="auto")
    df = dataset.frame.copy()
    logger.info(f"Fetched {len(df)} rows, {len(df.columns)} columns")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename OpenML column names to snake_case for SQL compatibility."""
    df = df.rename(columns=COLUMN_MAP)
    # sale_price comes back as object from OpenML; coerce to int
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce").astype("Int64")
    return df


def save_raw_csv(df: pd.DataFrame) -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    logger.info(f"Raw CSV saved: {RAW_CSV}")


def load_into_db(df: pd.DataFrame) -> None:
    schema_path = Path(__file__).parent.parent / "sql" / "schema.sql"
    schema_sql = schema_path.read_text()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(schema_sql)
        conn.execute("DELETE FROM raw_housing")
        df.to_sql("raw_housing", conn, if_exists="append", index=False)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM raw_housing").fetchone()[0]
        logger.info(f"Loaded {count} rows into raw_housing")
    finally:
        conn.close()


def run() -> pd.DataFrame:
    df = fetch_raw()
    df = normalize_columns(df)
    save_raw_csv(df)
    load_into_db(df)
    return df


if __name__ == "__main__":
    run()
