"""
Central logging configuration for recite.

Configures loguru so all application logs go to logs/ under project root.
Entry point (clintrialm.py) calls configure_logging() at startup.
"""
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from recite.utils.path_loader import get_project_root


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    app_name: str = "clintrialm",
    also_stderr: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> Path:
    """
    Configure loguru to write to logs/ (and optionally stderr).

    Creates log_dir if missing. Removes default handler and adds a file handler
    to log_dir / {app_name}.log with rotation and retention. If also_stderr,
    adds stderr handler at the same level.

    Args:
        level: Log level (e.g. "INFO", "DEBUG").
        log_dir: Directory for log files. If None, uses get_project_root() / "logs".
        app_name: Base name for log file (e.g. "clintrialm" -> clintrialm.log).
        also_stderr: If True, also log to stderr.
        rotation: Log rotation size (e.g. "10 MB").
        retention: Log retention (e.g. "7 days").

    Returns:
        Path to the log directory (for tests/callers).
    """
    if log_dir is None:
        log_dir = get_project_root() / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{app_name}.log"

    logger.remove()
    logger.add(
        str(log_file),
        level=level.upper(),
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    if also_stderr:
        logger.add(
            sys.stderr,
            level=level.upper(),
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>: <level>{message}</level>",
        )

    return log_dir
