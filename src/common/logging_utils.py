#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logging_utils.py

Centralized logging setup for the DDI-AS project.
Provides a consistent, quiet-but-informative console output and a
structured log file under results/logs/.
"""
from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path


def create_logger(name: str, log_dir: Path, level: int = logging.INFO) -> Logger:
    """Create and configure a logger with both file and stream handlers.

    Parameters
    ----------
    name : str
        Logger name.
    log_dir : Path
        Directory where the log file will be written.
    level : int, optional
        Logging level for the logger and handlers, by default logging.INFO.

    Returns
    -------
    Logger
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.debug("Logger initialized: %s", log_file)
    return logger