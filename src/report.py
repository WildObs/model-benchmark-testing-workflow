"""Assemble and save the HTML report and the misclassified-images export."""

import base64
import logging
from pathlib import Path

import pandas as pd

from .config import BenchmarkConfig
from .report_components import build_placeholders
from .metrics import BenchmarkMetrics
from .templating import load_style, render_template, resolve_template

logger = logging.getLogger(__name__)

FONT_FILES = (
    ("Poppins-Regular.woff2", "normal", 400),
    ("Poppins-SemiBold.woff2", "normal", 600),
    ("Poppins-Italic.woff2", "italic", 400),
)


def embedded_font_css(templates_dir) -> str:
    """@font-face rules with Poppins embedded as base64, so reports render the same offline."""
    rules = []
    for filename, style, weight in FONT_FILES:
        path = Path(templates_dir) / "fonts" / filename
        if not path.is_file():
            logger.warning("Font file not found: %s; falling back to system fonts", path)
            continue
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            f"@font-face{{font-family:'Poppins';font-style:{style};font-weight:{weight};"
            f"font-display:swap;src:url(data:font/woff2;base64,{data}) format('woff2');}}"
        )
    return "".join(rules)

MATHJAX = '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'


def write_html_report(
    config: BenchmarkConfig,
    model_name: str,
    species_display: list,
    metrics: BenchmarkMetrics,
    image_count: int,
    image_counts: dict,
    timestamp: str,
    run_id: str,
) -> Path:
    template_path = resolve_template(config.template, config.templates_path)
    logger.info("Rendering report with template '%s'", template_path.name)

    placeholders = build_placeholders(
        config, model_name, species_display, metrics, image_count, image_counts, timestamp
    )
    body = render_template(template_path, placeholders, config.exclude_sections)
    style = load_style(template_path, config.templates_path)

    document = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>Model Evaluation Report - {model_name}</title>"
        f"<style>{embedded_font_css(config.templates_path)}{style}</style>{MATHJAX}</head>"
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
