"""End-to-end benchmark run: load data, compute metrics, write outputs."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import BenchmarkConfig
from .data_loader import load_benchmark_table
from .logging_setup import setup_logging
from .metrics import BenchmarkMetrics, compute_metrics
from .model_comparison import create_model_comparison_report
from .report import export_misclassified, write_html_report

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """Results for a single Camtrap DP export evaluation."""

    model_name: str
    metrics: BenchmarkMetrics
    report_path: Path
    misclassified_path: Optional[Path]
    log_path: Path


@dataclass
class BenchmarkBatchRun:
    """Results for a batch of Camtrap DP evaluations in a test set."""

    set_name: str
    runs: List[BenchmarkRun]
    comparison_report_path: Optional[Path] = None
    log_path: Optional[Path] = None

    @property
    def report_path(self) -> Path:
        """Comparison report if generated; otherwise the single model evaluation report."""
        if self.comparison_report_path:
            return self.comparison_report_path
        if self.runs:
            return self.runs[0].report_path
        raise ValueError("No reports were generated.")

    @property
    def report_paths(self) -> List[Path]:
        """All report paths (evaluation reports + comparison report if generated)."""
        paths = [r.report_path for r in self.runs]
        if self.comparison_report_path:
            paths.append(self.comparison_report_path)
        return paths

    @property
    def model_names(self) -> List[str]:
        return [r.model_name for r in self.runs]

    # Single-run convenience / backward-compatibility properties
    @property
    def model_name(self) -> str:
        return ", ".join(self.model_names)

    @property
    def misclassified_path(self) -> Optional[Path]:
        return self.runs[0].misclassified_path if self.runs else None

    @property
    def metrics(self) -> BenchmarkMetrics:
        if self.runs:
            return self.runs[0].metrics
        raise ValueError("No runs available.")


def discover_export_folders(folder: Path) -> List[Path]:
    """Find all Camtrap DP export folders inside a test set folder.

    If the folder directly contains 'observations.csv', it is treated as a single export.
    Otherwise, scans for subdirectories containing 'observations.csv'.
    """
    folder = Path(folder)
    if not folder.is_dir():
        siblings = (
            sorted(p.name for p in folder.parent.iterdir() if p.is_dir() and not p.name.startswith("."))
            if folder.parent.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"Test set folder not found: {folder}. Folders available in {folder.parent}: {siblings}"
        )

    # Check if the folder itself is directly a Camtrap DP export
    if (folder / "observations.csv").is_file():
        return [folder]

    # Find subdirectories containing observations.csv
    child_dirs = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    export_dirs = [p for p in child_dirs if (p / "observations.csv").is_file()]

    if not export_dirs:
        if child_dirs:
            raise FileNotFoundError(
                f"No Camtrap DP export folders (containing observations.csv) found in {folder}. "
                f"Subdirectories found: {[p.name for p in child_dirs]}"
            )
        raise FileNotFoundError(f"No Camtrap DP export folders found in {folder}.")

    return export_dirs


def run_benchmark(config: BenchmarkConfig) -> BenchmarkBatchRun:
    """Run the full workflow for a test set containing one or more Camtrap DP exports."""
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(config.log_path, config.log_level, run_id)
    start = time.perf_counter()

    try:
        logger.info("Starting benchmark run %s for set '%s'", run_id, config.set_name)
        logger.debug("Config: %s", config)
        Path(config.input_path).mkdir(parents=True, exist_ok=True)
        set_output_dir = config.set_output_folder
        set_output_dir.mkdir(parents=True, exist_ok=True)

        export_folders = discover_export_folders(config.set_folder)
        logger.info(
            "Found %d Camtrap DP export folder(s) in set '%s'",
            len(export_folders),
            config.set_name,
        )

        runs: List[BenchmarkRun] = []
        for export_folder in export_folders:
            logger.info("Processing export: %s", export_folder.name)
            merged, model_name, species_display, image_counts = load_benchmark_table(export_folder)
            metrics = compute_metrics(merged, config.thresholds)

            misclassified_path = None
            if config.export_errors:
                misclassified_path = export_misclassified(
                    config, merged, model_name, run_id, output_dir=set_output_dir
                )

            report_path = write_html_report(
                config,
                model_name,
                species_display,
                metrics,
                len(merged),
                image_counts,
                now.strftime("%Y-%m-%d %H:%M:%S"),
                run_id,
                output_dir=set_output_dir,
            )
            runs.append(BenchmarkRun(model_name, metrics, report_path, misclassified_path, log_path))

        comparison_report_path = None
        if len(runs) > 1:
            logger.info(
                "Generating comparison report for %d models in set '%s'",
                len(runs),
                config.set_name,
            )
            comparison_report_path = create_model_comparison_report(
                reports_folder=[r.report_path for r in runs],
                output_folder=set_output_dir,
                run_id=run_id,
            )
        else:
            logger.info(
                "Set '%s' contains only 1 model (%s); comparison report skipped.",
                config.set_name,
                runs[0].model_name,
            )

    except Exception:
        logger.exception("Benchmark run %s failed", run_id)
        raise

    logger.info("Finished set '%s' in %.1fs", config.set_name, time.perf_counter() - start)
    return BenchmarkBatchRun(
        set_name=config.set_name,
        runs=runs,
        comparison_report_path=comparison_report_path,
        log_path=log_path,
    )
