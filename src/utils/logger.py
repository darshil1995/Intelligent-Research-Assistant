import logging
import sys
from src.utils.config import BASE_DIR

# Create a logs directory if it doesn't exist
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str):
    """
    Standardized logger for the entire project.
    Outputs to both Console and a log file.
    """
    logger = logging.getLogger(name)

    # Only configure if the logger doesn't have handlers already
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Format: [Timestamp] [Level] [Module] [Function] - Message
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 2. Console Handler (Standard Output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 3. File Handler (Persistent Log)
        file_handler = logging.FileHandler(LOG_DIR / "app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger