"""
Stage 1: Data Ingestion
Loads the real StressLevelDataset (Kaggle) from data/raw/ and logs dataset info.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import os
import urllib.request

def download_data():
    url = "YOUR_DATASET_LINK"
    os.makedirs("data/raw", exist_ok=True)
    urllib.request.urlretrieve(url, "data/raw/student_data.csv")

if not os.path.exists("data/raw/student_data.csv"):
    download_data()

import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger, log_separator
from src.utils.config_loader import load_config

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying", "stress_level",
]

def ingest_data(config_path: str = "configs/config.yaml") -> str:
    log_separator(logger, "Stage 1: Data Ingestion")
    config = load_config(config_path)
    raw_path = config.get("paths", "raw_data")

    if not Path(raw_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at '{raw_path}'.\n"
            "Download 'StressLevelDataset.csv' from:\n"
            "  https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis\n"
            f"and place it at: {raw_path}"
        )

    logger.info(f"Loading real dataset from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Validate expected columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing expected columns: {missing_cols}")

    logger.info(f"Dataset loaded successfully")
    logger.info(f"Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns      : {list(df.columns)}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicates   : {df.duplicated().sum()}")

    # Log per-domain stats
    logger.info(f"\nStress level distribution:\n{df['stress_level'].value_counts().sort_index().to_string()}")
    logger.info(f"\nFeature value ranges:")
    for col in df.columns:
        logger.info(f"  {col:35s} [{df[col].min():.1f} – {df[col].max():.1f}]")

    return raw_path

if __name__ == "__main__":
    ingest_data()
