import logging
import sys
from config import LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes to both the console and a log file.
    Call this at the top of every module:
        from utils.logger import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — don't add duplicate handlers
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handler — DEBUG and above (full detail)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
