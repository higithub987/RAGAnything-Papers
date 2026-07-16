"""Tests for configure_api_logging (Step 5c): the `api` logger becomes an INFO,
non-propagating logger with exactly one stderr handler, idempotently.

The `api` logger is process-global, so a fixture snapshots and restores its state so
these tests don't leak logging config into the rest of the suite.
"""

import logging

import pytest

from api.logging_setup import configure_api_logging


@pytest.fixture
def clean_api_logger():
    logger = logging.getLogger("api")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers = []
    logger.propagate = True
    logger.setLevel(logging.WARNING)
    try:
        yield logger
    finally:
        logger.handlers, logger.level, logger.propagate = saved


def _marked_handlers(logger):
    return [h for h in logger.handlers if getattr(h, "_api_logging_handler", False)]


def test_configures_level_handler_and_propagation(clean_api_logger):
    logger = clean_api_logger
    result = configure_api_logging()
    assert result is logger
    assert logger.level == logging.INFO
    assert logger.propagate is False
    handlers = _marked_handlers(logger)
    assert len(handlers) == 1
    assert handlers[0].level == logging.INFO


def test_is_idempotent(clean_api_logger):
    logger = clean_api_logger
    configure_api_logging()
    configure_api_logging()
    assert len(_marked_handlers(logger)) == 1  # no duplicate handler on re-call


def test_respects_custom_level(clean_api_logger):
    logger = clean_api_logger
    configure_api_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG
    assert _marked_handlers(logger)[0].level == logging.DEBUG
