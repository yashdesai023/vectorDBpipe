import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs/", level: str = "INFO") -> logging.Logger:
    """
    Configure a consistent logger for the entire pipeline.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # Avoid duplicate log handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
