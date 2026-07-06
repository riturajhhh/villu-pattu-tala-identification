"""
utils/logger.py
===============
Structured logging for the Villu Pattu Tala Identification System.
Provides color console output and rotating file logging.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ANSI colour codes for console output
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_COLOURS = {
    "DEBUG":    "\033[94m",   # Blue
    "INFO":     "\033[92m",   # Green
    "WARNING":  "\033[93m",   # Yellow
    "ERROR":    "\033[91m",   # Red
    "CRITICAL": "\033[95m",   # Magenta
}


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colour codes to the level name."""

    _FMT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = _COLOURS.get(record.levelname, _RESET)
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        formatter = logging.Formatter(self._FMT, datefmt=self._DATE_FMT)
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    _FMT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        formatter = logging.Formatter(self._FMT, datefmt=self._DATE_FMT)
        return formatter.format(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "villu_pattu",
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Create and configure a named logger.

    Parameters
    ----------
    name:
        Logger name (use ``__name__`` in each module for hierarchy).
    level:
        Logging level string: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
    log_file:
        Optional path to a rotating log file.  If ``None``, file logging is
        disabled.
    max_bytes:
        Maximum size of each log file before rotation (default 10 MB).
    backup_count:
        Number of rotated backup files to keep.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if setup_logger is called more than once
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # --- Console handler (coloured) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_ColourFormatter())
    logger.addHandler(console_handler)

    # --- File handler (plain, rotating) ---
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_PlainFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "villu_pattu") -> logging.Logger:
    """Return an existing logger by name (must have been set up first).

    If the logger doesn't exist yet, a basic console-only logger is returned.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# ---------------------------------------------------------------------------
# Project-level root logger initialised from config if available
# ---------------------------------------------------------------------------

def init_from_config() -> logging.Logger:
    """Initialise the root project logger using ``config/config.yaml``."""
    try:
        from utils.config_loader import get_config  # type: ignore
        cfg = get_config()
        return setup_logger(
            name="villu_pattu",
            level=cfg.logging.level,
            log_file=cfg.logging.log_file,
            max_bytes=cfg.logging.max_bytes,
            backup_count=cfg.logging.backup_count,
        )
    except Exception:
        # Fallback to defaults — config may not be available in all contexts
        return setup_logger(name="villu_pattu")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log = setup_logger("villu_pattu.test", level="DEBUG")
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")
