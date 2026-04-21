import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from pathlib import Path
import requests

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


# ✅ CLEAN DOWNLOAD FUNCTION (no gdown, no permissions issues)
def download_from_drive(file_id, output_path):
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def ingest_data(config_path: str = "configs/config.yaml") -> str:
    log_separator(logger, "Stage 1: Data Ingestion")

    config = load_config(config_path)
    raw_path = config.get("paths", "raw_data")

    # ✅ DOWNLOAD ONLY IF NOT EXISTS
    if not Path(raw_path).exists():
        logger.info("Dataset not found. Downloading from Google Drive...")

        file_id = "1LhdZaDJAJQbFcIL9md3nk37WhGWOZb9H"
        download_from_drive(file_id, raw_path)

    # ✅ LOAD DATA
    logger.info(f"Loading dataset from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Validate columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing expected columns: {missing_cols}")

    logger.info(f"Dataset loaded successfully")
    logger.info(f"Shape: {df.shape}")

    return raw_path


if __name__ == "__main__":
    ingest_data()