"""Central logging configuration for recite."""
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from recite.utils.path_loader import get_project_root


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    app_name: str = "recite",
    also_stderr: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> Path:
    """Configure loguru to write to logs/ and optionally stderr."""
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
