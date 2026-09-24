"""Project logging: colour console output via rich, plus a rotating log file.

Usage:
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

LOG_DIR = Path("logs")  # git-ignored
LOG_FILE = "dino_ai.log"
MAX_BYTES = 10 * 1024**2  # rotate at 10MB
BACKUP_COUNT = 5  # keep dino_ai.log.1 .. .5
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_handlers: list[logging.Handler] = []


def setup_logging(level: int = logging.INFO, log_dir: Optional[Path] = None) -> None:
    """Attach console and file handlers to the root logger (replaces earlier ones)"""
    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()
    _handlers.clear()

    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    console = RichHandler(rich_tracebacks=True, show_path=False)
    file = RotatingFileHandler(
        log_dir / LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    file.setFormatter(logging.Formatter(FILE_FORMAT))
    _handlers.extend([console, file])
    for handler in _handlers:
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Logger for a module; sets up logging with defaults on first use"""
    if not _handlers:
        setup_logging()
    return logging.getLogger(name)
