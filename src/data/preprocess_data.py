"""
Stage 2: Data Preprocessing
Handles missing values, duplicates, outliers, and standardisation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger, log_separator
from src.utils.config_loader import load_config

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip whitespace, replace spaces with underscores."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.debug("Column names standardised.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numeric columns → median imputation
    - Categorical columns → mode imputation
    """
    n_missing_before = df.isnull().sum().sum()
    logger.info(f"Missing values before imputation: {n_missing_before}")

    for col in df.columns:
        n_null = df[col].isnull().sum()
        if n_null == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logger.debug(f"  [{col}] — {n_null} nulls filled with median={fill_val:.2f}")
        else:
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            logger.debug(f"  [{col}] — {n_null} nulls filled with mode='{fill_val}'")

    logger.info(f"Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed:
        logger.info(f"Removed {n_removed} duplicate row(s).")
    else:
        logger.info("No duplicate rows found.")
    return df


def handle_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Clip numeric columns using IQR method.
    Values beyond Q1 - threshold*IQR and Q3 + threshold*IQR are clipped.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_clipped = 0

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_clipped += n_outliers
            logger.debug(f"  [{col}] — {n_outliers} outliers clipped to [{lower:.2f}, {upper:.2f}]")

    logger.info(f"Outlier handling complete. Total values clipped: {total_clipped}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        logger.info(f"One-hot encoded columns: {cat_cols}")
    else:
        logger.info("No categorical columns to encode.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(config_path: str = "configs/config.yaml") -> str:
    log_separator(logger, "Stage 2: Data Preprocessing")
    config = load_config(config_path)

    raw_path       = config.get("paths", "raw_data")
    processed_path = config.get("paths", "processed_data")
    outlier_method = config.get("preprocessing", "outlier_method", default="iqr")
    threshold      = config.get("preprocessing", "outlier_threshold", default=1.5)

    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded dataset shape: {df.shape}")

    df = standardize_column_names(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)

    if outlier_method == "iqr":
        df = handle_outliers_iqr(df, threshold=threshold)

    df = encode_categoricals(df)

    Path(os.path.dirname(processed_path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed dataset saved → {processed_path}")
    logger.info(f"Final shape: {df.shape}")

    return processed_path


if __name__ == "__main__":
    preprocess_data()
