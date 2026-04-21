"""
Logger Utility
Centralized logging configuration for the Burnout Prediction project.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
import sys


def get_logger(name: str, log_dir: str = "logs", log_file: str = "project.log") -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to store log files
        log_file: Name of the log file

    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # ── File handler (DEBUG level) ──────────────────────────────────────────
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # ── Console handler (INFO level) ────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_separator(logger: logging.Logger, title: str = "") -> None:
    """Log a visual separator for readability."""
    sep = "=" * 60
    if title:
        logger.info(f"{sep}")
        logger.info(f"  {title.upper()}")
        logger.info(f"{sep}")
    else:
        logger.info(sep)
