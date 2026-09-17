import logging
from typing import Optional


def setup_logger(
    name: str, level: int = logging.INFO, fmt: Optional[str] = None
) -> logging.Logger:
    """Create and return a configured logger for the project.

    Lightweight helper used throughout the codebase.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = fmt or "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
