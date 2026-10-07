"""Logging configuration: console summary plus a detailed log file per run."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

_HANDLER_TAG = "_benchmark_handler"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    log_dir: Union[str, Path],
    level: str = "INFO",
    run_id: Optional[str] = None,
) -> Path:
    """Configure root logging and return the path of the log file.

    Safe to call repeatedly (e.g. re-running a notebook cell): handlers added by
    a previous call are replaced rather than duplicated. The console shows
    messages at ``level``; the log file always records DEBUG detail.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"benchmark_{run_id}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(level.upper())
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    for handler in (console, file_handler):
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    logging.getLogger(__name__).info("Logging to %s", log_file)
    return log_file
