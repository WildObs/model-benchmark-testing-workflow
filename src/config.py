"""Configuration for a benchmark run."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class BenchmarkConfig:
    """User preferences for a benchmark run.

    Relative paths are resolved against the project root, so the same config
    works from the notebook (scripts/) and from the command line.
    """

    input_camtrap_folder_name: str
    data_source_location: str = ""
    data_collation_process: str = ""
    thresholds: List[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    export_errors: bool = True

    # Report formatting
    template: str = "full"  # template name in templates_path, or a path to a .md file
    exclude_sections: List[str] = field(default_factory=list)

    # Folders
    input_path: Union[str, Path] = "Input_Data"
    output_path: Union[str, Path] = "Output_Reports"
    templates_path: Union[str, Path] = "templates"
    log_path: Union[str, Path] = "logs"
    log_level: str = "INFO"

    def __post_init__(self):
        for name in ("input_path", "output_path", "templates_path", "log_path"):
            setattr(self, name, _resolve(getattr(self, name)))
        self.thresholds = sorted({float(t) for t in self.thresholds})
        self.log_level = self.log_level.upper()
        self.validate()

    def validate(self) -> None:
        if not self.input_camtrap_folder_name:
            raise ValueError("input_camtrap_folder_name must be set.")
        if not self.thresholds:
            raise ValueError("At least one confidence threshold is required.")
        bad = [t for t in self.thresholds if not 0.0 <= t <= 1.0]
        if bad:
            raise ValueError(f"Confidence thresholds must be between 0 and 1, got: {bad}")
        if not isinstance(logging.getLevelName(self.log_level), int):
            raise ValueError(f"Invalid log_level: {self.log_level}")

    @property
    def camtrap_folder(self) -> Path:
        return Path(self.input_path) / self.input_camtrap_folder_name


def _resolve(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
