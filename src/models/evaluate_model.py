"""
Stage 6: Model Evaluation — loads feature names from feature_names.json
so it always matches whatever the training run used.
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger, log_separator
from src.utils.config_loader import load_config

logger = get_logger(__name__)
TARGET_COL = "burnout_score"


def evaluate_model(config_path: str = "configs/config.yaml") -> dict:
    log_separator(logger, "Stage 6: Model Evaluation")
    config = load_config(config_path)

    engineered_path = config.get("paths", "engineered_data")
    model_dir       = config.get("paths", "model_dir")
    best_model_path = config.get("paths", "best_model")
    results_dir     = config.get("paths", "results_dir")
    test_size       = config.get("training", "test_size", default=0.2)
    seed            = config.get("project",  "random_seed", default=42)

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load feature names saved during training
    feat_path = os.path.join(model_dir, "feature_names.json")
    with open(feat_path) as f:
        feature_names = json.load(f)
    logger.info(f"Using {len(feature_names)} features from training run")

    df = pd.read_csv(engineered_path)
    X  = df[feature_names].values
    y  = df[TARGET_COL].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    report = {}
    model_files = sorted(Path(model_dir).glob("*_model.pkl"))
    logger.info(f"Found {len(model_files)} model(s) to evaluate")

    for model_file in model_files:
        model_name = model_file.stem.replace("_model", "")
        with open(model_file, "rb") as f:
            pipeline = pickle.load(f)
        y_pred = pipeline.predict(X_test)
        mae  = round(mean_absolute_error(y_test, y_pred), 4)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        r2   = round(r2_score(y_test, y_pred), 4)
        report[model_name] = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info(f"  {model_name:25s}  MAE={mae:7.4f}  RMSE={rmse:7.4f}  R²={r2:7.4f}")

    if Path(best_model_path).exists():
        with open(best_model_path, "rb") as f:
            best = pickle.load(f)
        residuals = y_test - best.predict(X_test)
        report["best_model_residuals"] = {
            "mean":  round(float(np.mean(residuals)), 4),
            "std":   round(float(np.std(residuals)),  4),
            "max_abs": round(float(np.max(np.abs(residuals))), 4),
        }
        logger.info(f"Best model residuals: {report['best_model_residuals']}")

    metrics_path = os.path.join(results_dir, "metrics_summary.json")
    if Path(metrics_path).exists():
        with open(metrics_path) as f:
            report["best_model_name"] = json.load(f).get("best_model", "unknown")

    eval_path = os.path.join(results_dir, "evaluation_report.json")
    with open(eval_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved → {eval_path}")
    return report

if __name__ == "__main__":
    evaluate_model()
