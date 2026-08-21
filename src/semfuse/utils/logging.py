"""Logging utilities."""

from __future__ import annotations

import logging

_SEMFUSE_LOGGER_NAME = "semfuse"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the semfuse namespace, configuring the root once."""
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        root = logging.getLogger(_SEMFUSE_LOGGER_NAME)
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            root.addHandler(handler)
            root.setLevel(logging.WARNING)
        _configured = True
    return logger


def set_log_level(level: int | str) -> None:
    """Set the log level for the semfuse namespace."""
    logging.getLogger(_SEMFUSE_LOGGER_NAME).setLevel(level)
