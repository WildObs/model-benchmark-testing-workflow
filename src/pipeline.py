"""End-to-end benchmark run: load data, compute metrics, write outputs."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import BenchmarkConfig
from .data_loader import load_benchmark_table
from .logging_setup import setup_logging
from .metrics import BenchmarkMetrics, compute_metrics
from .report import export_misclassified, write_html_report

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    model_name: str
    metrics: BenchmarkMetrics
    report_path: Path
    misclassified_path: Optional[Path]
    log_path: Path


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    """Run the full workflow for one Camtrap DP export."""
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(config.log_path, config.log_level, run_id)
    start = time.perf_counter()

    try:
        logger.info("Starting benchmark run %s", run_id)
        logger.debug("Config: %s", config)
        Path(config.input_path).mkdir(parents=True, exist_ok=True)
        Path(config.output_path).mkdir(parents=True, exist_ok=True)

        merged, model_name, species_display, image_counts = load_benchmark_table(config.camtrap_folder)
        metrics = compute_metrics(merged, config.thresholds)

        misclassified_path = None
        if config.export_errors:
            misclassified_path = export_misclassified(config, merged, model_name, run_id)

        report_path = write_html_report(
            config, model_name, species_display, metrics, len(merged), image_counts,
            now.strftime("%Y-%m-%d %H:%M:%S"), run_id,
        )
    except Exception:
        logger.exception("Benchmark run %s failed", run_id)
        raise

    logger.info("Finished in %.1fs", time.perf_counter() - start)
    return BenchmarkRun(model_name, metrics, report_path, misclassified_path, log_path)
