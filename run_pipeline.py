"""
Main Pipeline Runner
Executes all stages of the Burnout Prediction ML pipeline in order.
Can be used as an alternative to `dvc repro`.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config configs/config.yaml
    python run_pipeline.py --stages ingest preprocess  # run specific stages
"""

import argparse
import time
import sys
import os

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

from src.utils.logger import get_logger, log_separator

logger = get_logger("pipeline_runner")


STAGES = {
    "ingest":      ("src.data.ingest_data",         "ingest_data"),
    "preprocess":  ("src.data.preprocess_data",     "preprocess_data"),
    "engineer":    ("src.data.feature_engineering", "engineer_features"),
    "train":       ("src.models.train_model",       "train_all_models"),
    "evaluate":    ("src.models.evaluate_model",    "evaluate_model"),
    "visualize":   ("src.visualization.visualize_results", "visualize_results"),
}

ALL_STAGES = list(STAGES.keys())


def run_stage(stage_name: str, config_path: str) -> None:
    module_path, func_name = STAGES[stage_name]
    import importlib
    module = importlib.import_module(module_path)
    func   = getattr(module, func_name)
    func(config_path)


def run_pipeline(stages: list, config_path: str) -> None:
    log_separator(logger, "🚀 Student Burnout Prediction — Full Pipeline")
    logger.info(f"Config: {config_path}")
    logger.info(f"Stages: {stages}")

    total_start = time.time()
    results = {}

    for stage in stages:
        if stage not in STAGES:
            logger.error(f"Unknown stage: '{stage}'. Valid stages: {ALL_STAGES}")
            sys.exit(1)

        logger.info(f"\n{'─'*50}")
        logger.info(f"  Running stage: {stage.upper()}")
        logger.info(f"{'─'*50}")
        start = time.time()
        try:
            run_stage(stage, config_path)
            elapsed = round(time.time() - start, 2)
            results[stage] = f"✅ {elapsed}s"
            logger.info(f"Stage '{stage}' completed in {elapsed}s")
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            results[stage] = f"❌ FAILED ({e})"
            logger.error(f"Stage '{stage}' FAILED after {elapsed}s: {e}", exc_info=True)
            sys.exit(1)

    total_elapsed = round(time.time() - total_start, 2)

    log_separator(logger, "Pipeline Summary")
    for stage, status in results.items():
        logger.info(f"  {stage:15s} {status}")
    logger.info(f"\n  Total time: {total_elapsed}s")
    log_separator(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Burnout Prediction ML Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--stages", nargs="+", default=ALL_STAGES,
        help=f"Stages to run (default: all). Options: {ALL_STAGES}"
    )
    args = parser.parse_args()

    run_pipeline(args.stages, args.config)
