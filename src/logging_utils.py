"""Logging utilities for backtesting and model training."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_logger: Optional[logging.Logger] = None
_wandb_available = False
_wandb = None


def _init_wandb() -> bool:
    """Detect if wandb is available and configured."""
    global _wandb_available, _wandb
    try:
        import wandb
        _wandb = wandb
        _wandb_available = True
        return True
    except ImportError:
        _wandb_available = False
        return False


def get_logger(name: str = "backtest") -> logging.Logger:
    """Get or create a logger instance with standardized formatting."""
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicates
    _logger.handlers.clear()

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    # Detect W&B availability on first logger creation
    _init_wandb()

    return _logger


def log_info(message: str) -> None:
    """Log an info-level message."""
    logger = get_logger()
    logger.info(message)


def log_debug(message: str) -> None:
    """Log a debug-level message."""
    logger = get_logger()
    logger.debug(message)


def log_warning(message: str) -> None:
    """Log a warning-level message."""
    logger = get_logger()
    logger.warning(message)


def log_error(message: str) -> None:
    """Log an error-level message."""
    logger = get_logger()
    logger.error(message)


def wandb_log(metrics: dict) -> None:
    """Log metrics to Weights & Biases if available and initialized."""
    global _wandb_available, _wandb
    if not _wandb_available or _wandb is None:
        return
    try:
        _wandb.log(metrics)
    except Exception as e:
        log_debug(f"W&B logging failed (non-critical): {e}")


def wandb_init(project: str, name: str, config: dict) -> None:
    """Initialize Weights & Biases run if available."""
    global _wandb_available, _wandb
    if not _wandb_available or _wandb is None:
        return
    try:
        _wandb.init(project=project, name=name, config=config)
        log_info("Weights & Biases initialized")
    except Exception as e:
        log_warning(f"W&B initialization failed (continuing without): {e}")


def wandb_finish() -> None:
    """Finish Weights & Biases run if active."""
    global _wandb_available, _wandb
    if not _wandb_available or _wandb is None:
        return
    try:
        _wandb.finish()
    except Exception:
        pass
