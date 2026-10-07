"""Assemble and save the HTML report and the misclassified-images export."""

import logging
from pathlib import Path

import pandas as pd

from .config import BenchmarkConfig
from .report_components import build_placeholders
from .metrics import BenchmarkMetrics
from .templating import load_style, render_template, resolve_template

logger = logging.getLogger(__name__)

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap">'
)
MATHJAX = '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'


def write_html_report(
    config: BenchmarkConfig,
    model_name: str,
    species_display: list,
    metrics: BenchmarkMetrics,
    image_count: int,
    timestamp: str,
    run_id: str,
) -> Path:
    template_path = resolve_template(config.template, config.templates_path)
    logger.info("Rendering report with template '%s'", template_path.name)

    placeholders = build_placeholders(config, model_name, species_display, metrics, image_count, timestamp)
    body = render_template(template_path, placeholders, config.exclude_sections)
    style = load_style(template_path, config.templates_path)

    document = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>Model Evaluation Report - {model_name}</title>"
        f"{FONTS}<style>{style}</style>{MATHJAX}</head>"
        f'<body><div class="report-container">{body}</div></body></html>'
    )

    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    report_path = Path(config.output_path) / f"Model_Benchmark_Test_Report_{model_name}_{template_path.stem}_{run_id}.html"
    report_path.write_text(document, encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return report_path


def export_misclassified(config: BenchmarkConfig, merged: pd.DataFrame, model_name: str, run_id: str) -> Path:
    errors = merged[merged["true_species"] != merged["pred_species"]]
    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    path = Path(config.output_path) / f"Misclassified_Images_{model_name}_{run_id}.csv"
    errors.to_csv(path, index=False)
    logger.info("Exported %d misclassified images to %s", len(errors), path)
    return path
