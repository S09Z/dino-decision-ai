"""Tests for project logging"""

import logging
from logging.handlers import RotatingFileHandler

import pytest
from rich.logging import RichHandler

from src.utils import logger as logger_module
from src.utils.logger import (
    BACKUP_COUNT,
    LOG_FILE,
    MAX_BYTES,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def isolated_logging(tmp_path, monkeypatch):
    """Log into tmp_path and remove our handlers afterwards"""
    monkeypatch.setattr(logger_module, "LOG_DIR", tmp_path)
    root = logging.getLogger()
    level = root.level
    yield
    for handler in logger_module._handlers:
        root.removeHandler(handler)
        handler.close()
    logger_module._handlers.clear()
    root.setLevel(level)


def our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h in logger_module._handlers]


def test_setup_adds_rich_console_and_rotating_file(tmp_path):
    setup_logging(log_dir=tmp_path)

    console, file = our_handlers()
    assert isinstance(console, RichHandler)
    assert isinstance(file, RotatingFileHandler)
    assert file.baseFilename == str(tmp_path / LOG_FILE)
    assert file.maxBytes == MAX_BYTES
    assert file.backupCount == BACKUP_COUNT


def test_file_log_uses_structured_format(tmp_path):
    setup_logging(log_dir=tmp_path)

    get_logger("src.training").warning("buffer full")

    line = (tmp_path / LOG_FILE).read_text().strip()
    timestamp, level, name, message = line.split(" | ")
    assert timestamp[:4].isdigit()
    assert level.strip() == "WARNING"
    assert name == "src.training"
    assert message == "buffer full"


def test_setup_twice_does_not_duplicate_handlers(tmp_path):
    setup_logging(log_dir=tmp_path)
    setup_logging(log_dir=tmp_path)

    assert len(our_handlers()) == 2


def test_get_logger_sets_up_logging_on_first_use(tmp_path):
    log = get_logger("src.env")

    assert log.name == "src.env"
    assert len(our_handlers()) == 2
    assert (tmp_path / LOG_FILE).exists()


def test_level_filters_lower_messages(tmp_path):
    setup_logging(level=logging.WARNING, log_dir=tmp_path)

    log = get_logger("src.env")
    log.info("hidden")
    log.error("shown")

    text = (tmp_path / LOG_FILE).read_text()
    assert "hidden" not in text
    assert "shown" in text
