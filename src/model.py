"""
Model training, evaluation, and feature importance for the housing price predictor.
Trains a baseline linear regression and a gradient boosting model, prints
comparison metrics, and returns results for use in the modeling notebook.
"""

import sqlite3
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DB_PATH = Path(__file__).parent.parent / "data" / "housing.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_rmse_scores = -cross_val_score(
        model, X_train, y_train,
        scoring="neg_root_mean_squared_error", cv=5
    )

    result = {
        "model": name,
        "test_rmse": rmse(y_test, y_pred),
        "test_r2": r2_score(y_test, y_pred),
        "cv_rmse_mean": cv_rmse_scores.mean(),
        "cv_rmse_std": cv_rmse_scores.std(),
    }

    logger.info(
        f"{name}: test RMSE=${result['test_rmse']:,.0f}, "
        f"R²={result['test_r2']:.3f}, "
        f"5-fold CV RMSE=${result['cv_rmse_mean']:,.0f} ± {result['cv_rmse_std']:,.0f}"
    )
    return result


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from a GradientBoostingRegressor."""
    # Unwrap pipeline if present
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    importances = estimator.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run(X: pd.DataFrame, y: pd.Series) -> dict:
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Baseline: Ridge regression (linear, needs scaling)
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])
    ridge_result = evaluate(ridge_pipe, X_train, X_test, y_train, y_test, "Ridge")

    # Stronger model: Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    gb_result = evaluate(gb, X_train, X_test, y_train, y_test, "GradientBoosting")

    importance_df = get_feature_importance(gb, feature_names)

    return {
        "results": [ridge_result, gb_result],
        "gb_model": gb,
        "ridge_model": ridge_pipe,
        "feature_importance": importance_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_gb": gb.predict(X_test),
        "y_pred_ridge": ridge_pipe.predict(X_test),
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import run as get_features

    X, y, _ = get_features()
    results = run(X, y)
    print("\nTop 10 features:")
    print(results["feature_importance"].head(10).to_string(index=False))
