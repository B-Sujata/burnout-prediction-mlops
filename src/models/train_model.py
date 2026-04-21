"""
Stage 5: Model Training — Real Dataset Edition
Trains Linear Regression, Random Forest, Gradient Boosting (+ XGBoost if available).
All experiments tracked with MLflow.
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import mlflow, mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.utils.logger import get_logger, log_separator
from src.utils.config_loader import load_config

logger = get_logger(__name__)

# All features the model trains on (raw + engineered)
FEATURE_COLS = [
    # Raw psychological
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    # Raw physical
    "headache", "blood_pressure", "breathing_problem",
    # Raw environmental
    "noise_level", "living_conditions", "safety", "basic_needs",
    # Raw academic
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns",
    # Raw social
    "social_support", "peer_pressure", "extracurricular_activities", "bullying",
    # Engineered composite
    "emotional_strain", "physical_stress", "academic_pressure",
    "social_stress", "recovery_index",
    # Interaction features
    "stress_sleep_ratio", "academic_burden", "isolation_index",
    "environment_quality", "mental_load",
]
TARGET_COL = "burnout_score"


def compute_metrics(y_true, y_pred):
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
    }


def build_models(config):
    seed   = config.get("project",  "random_seed", default=42)
    rf_cfg = config.get("models",   "random_forest")
    gb_cfg = config.get("models",   "gradient_boosting")

    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators   = rf_cfg.get("n_estimators", 150),
                max_depth      = rf_cfg.get("max_depth", 12),
                min_samples_split = rf_cfg.get("min_samples_split", 4),
                min_samples_leaf  = rf_cfg.get("min_samples_leaf", 2),
                random_state   = seed, n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators = gb_cfg.get("n_estimators", 200),
                learning_rate= gb_cfg.get("learning_rate", 0.05),
                max_depth    = gb_cfg.get("max_depth", 5),
                random_state = seed,
            )),
        ]),
    }

    if XGBOOST_AVAILABLE:
        xgb_cfg = config.get("models", "xgboost")
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                n_estimators     = xgb_cfg.get("n_estimators", 200),
                learning_rate    = xgb_cfg.get("learning_rate", 0.05),
                max_depth        = xgb_cfg.get("max_depth", 6),
                subsample        = xgb_cfg.get("subsample", 0.8),
                colsample_bytree = xgb_cfg.get("colsample_bytree", 0.8),
                random_state     = seed, verbosity=0,
            )),
        ])
    else:
        logger.warning("XGBoost not installed — skipping.")

    return models


def train_all_models(config_path: str = "configs/config.yaml") -> dict:
    log_separator(logger, "Stage 5: Model Training")
    config = load_config(config_path)

    engineered_path = config.get("paths", "engineered_data")
    model_dir       = config.get("paths", "model_dir")
    best_model_path = config.get("paths", "best_model")
    results_dir     = config.get("paths", "results_dir")
    test_size       = config.get("training", "test_size", default=0.2)
    seed            = config.get("project",  "random_seed", default=42)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(engineered_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"Training on {len(available)} features | target: {TARGET_COL}")

    X = df[available].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(config.get("mlflow", "tracking_uri", default="mlruns"))
        mlflow.set_experiment(config.get("mlflow", "experiment_name", default="burnout_real"))

    models  = build_models(config)
    results = {}

    for name, pipeline in models.items():
        logger.info(f"\n▶ Training {name} ...")
        cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        logger.info(f"  CV R²: {cv.round(4)}  mean={cv.mean():.4f} ± {cv.std():.4f}")

        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics.update({"cv_r2_mean": round(cv.mean(), 4), "cv_r2_std": round(cv.std(), 4)})
        logger.info(f"  MAE={metrics['mae']}  RMSE={metrics['rmse']}  R²={metrics['r2']}")

        pkl_path = os.path.join(model_dir, f"{name.lower()}_model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline, f)

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "test_size": test_size, "n_features": len(available)})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, "model")

        results[name] = {"pipeline": pipeline, "metrics": metrics}

    # Select best model by R²
    best_name = max(results, key=lambda n: results[n]["metrics"]["r2"])
    with open(best_model_path, "wb") as f:
        pickle.dump(results[best_name]["pipeline"], f)
    logger.info(f"\n★ Best: {best_name} (R²={results[best_name]['metrics']['r2']}) → {best_model_path}")

    summary = {n: r["metrics"] for n, r in results.items()}
    summary["best_model"] = best_name
    with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
        json.dump(available, f)

    return summary

if __name__ == "__main__":
    train_all_models()
