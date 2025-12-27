import logging
import os
import sys
from datetime import datetime


def get_logger(logger_name: str = "predictive_maintenance") -> logging.Logger:

    # Create logs directory if it doesn't exist
    logs_dir = "artifacts/logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File handler
    log_file = os.path.join(logs_dir, f"{logger_name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
