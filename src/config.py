"""Configuration for a benchmark run."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class BenchmarkConfig:
    """User preferences for a benchmark run.

    Relative paths are resolved against the project root, so the same config
    works from the notebook (scripts/) and from the command line.
    """

    test_set_name: str = ""
    input_camtrap_folder_name: str = ""
    data_source_location: str = ""
    data_collation_process: str = ""
    thresholds: List[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    export_errors: bool = False  # optional: save a CSV of misclassified images
    lookup_taxonomy: bool = False  # look up full taxonomy for each species using ALA galah (needs internet)
    clear_existing_outputs: bool = False  # clear existing reports/outputs in the set folder before generating new ones
    clear_existing: bool = False  # alias for clear_existing_outputs

    # Which observations to analyse: "event" (default; works for all exports) or "media" (bounding-box
    # observations, only used if every export in the set contains them - otherwise reverts to "event")
    observation_level: str = "event"

    # Report formatting
    template: str = "full"  # template name in templates_path, or a path to a .md file
    exclude_sections: List[str] = field(default_factory=list)
    # Report only these species (names as shown in the report); empty means all species tested
    species_subset: List[str] = field(default_factory=list)

    # Only evaluate these export folders (names inside the test set folder); empty means all exports
    export_subset: List[str] = field(default_factory=list)

    # Let a higher-ranking input label (e.g. an order) match model predictions of lower-ranking
    # taxa within it when calculating recall, precision and F1 (uses ALA galah; needs internet)
    allow_higher_taxon_matching: bool = False

    # Acceptable substitutions per input label, e.g. {"Lagomorpha": ["Leporidae"]}; any taxonomic rank
    custom_matches: Dict[str, List[str]] = field(default_factory=dict)

    # Folders
    input_path: Union[str, Path] = "Input_Data"
    output_path: Union[str, Path] = "Output_Reports"
    templates_path: Union[str, Path] = "templates"
    log_path: Union[str, Path] = "logs"
    log_level: str = "INFO"

    def __post_init__(self):
        if not self.test_set_name and self.input_camtrap_folder_name:
            self.test_set_name = self.input_camtrap_folder_name
        elif not self.input_camtrap_folder_name and self.test_set_name:
            self.input_camtrap_folder_name = self.test_set_name

        if self.clear_existing and not self.clear_existing_outputs:
            self.clear_existing_outputs = self.clear_existing
        elif self.clear_existing_outputs and not self.clear_existing:
            self.clear_existing = self.clear_existing_outputs

        for name in ("input_path", "output_path", "templates_path", "log_path"):
            setattr(self, name, _resolve(getattr(self, name)))
        self.thresholds = sorted({float(t) for t in self.thresholds})
        self.custom_matches = {
            str(k).strip(): [str(s).strip() for s in v if str(s).strip()]
            for k, v in self.custom_matches.items()
            if str(k).strip() and any(str(s).strip() for s in v)
        }
        self.export_subset = [str(s).strip() for s in self.export_subset if str(s).strip()]
        self.species_subset = [str(s).strip() for s in self.species_subset if str(s).strip()]
        self.log_level = self.log_level.upper()
        self.observation_level = str(self.observation_level).strip().lower()
        self.validate()

    def validate(self) -> None:
        if not self.test_set_name and not self.input_camtrap_folder_name:
            raise ValueError("test_set_name must be set.")
        if not self.thresholds:
            raise ValueError("At least one confidence threshold is required.")
        bad = [t for t in self.thresholds if not 0.0 <= t <= 1.0]
        if bad:
            raise ValueError(f"Confidence thresholds must be between 0 and 1, got: {bad}")
        if not isinstance(logging.getLevelName(self.log_level), int):
            raise ValueError(f"Invalid log_level: {self.log_level}")
        if self.observation_level not in ("event", "media"):
            raise ValueError(f"observation_level must be 'event' or 'media', got {self.observation_level!r}")

    @property
    def set_name(self) -> str:
        return self.test_set_name or self.input_camtrap_folder_name

    @property
    def set_folder(self) -> Path:
        return Path(self.input_path) / self.set_name

    @property
    def camtrap_folder(self) -> Path:
        return self.set_folder

    @property
    def set_output_folder(self) -> Path:
        return Path(self.output_path) / self.set_name


def _resolve(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
