"""Compare per-species optimal-threshold F1 scores from benchmark reports."""

import logging
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from .report import embedded_font_css
from .templating import Placeholder, load_style, render_template

logger = logging.getLogger(__name__)

REPORT_PREFIX = "WildObs_CV_Model_Evaluation_Report_"
COMPARISON_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "model_comparison.md"


class _EvaluationReportParser(HTMLParser):
    """Extract the model name and optimal-threshold F1 table from a report."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.model_name = ""
        self._in_strong = False
        self._strong_parts: List[str] = []
        self._expect_model_name = False
        self._in_section_heading = False
        self._section_heading_parts: List[str] = []
        self._in_best_threshold_section = False
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._table_rows: List[List[str]] = []
        self._tables: List[List[List[str]]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "strong":
            self._in_strong = True
            self._strong_parts = []
        elif tag == "h2":
            self._in_section_heading = True
            self._section_heading_parts = []
        elif tag == "table" and self._in_best_threshold_section:
            self._in_table = True
            self._table_rows = []
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._table_rows.append([])
        elif tag in ("th", "td") and self._in_row:
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag):
        if tag == "strong":
            self._in_strong = False
            label = " ".join("".join(self._strong_parts).split()).lower()
            if label == "computer vision (cv) model tested:":
                self._expect_model_name = True
        elif tag == "h2":
            heading = " ".join("".join(self._section_heading_parts).split()).lower()
            self._in_best_threshold_section = heading == "optimal confidence threshold by species"
            self._in_section_heading = False
        elif tag in ("th", "td") and self._in_cell:
            self._table_rows[-1].append(" ".join("".join(self._cell_parts).split()))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            self._in_row = False
        elif tag == "table" and self._in_table:
            self._tables.append(self._table_rows)
            self._in_table = False
            self._in_best_threshold_section = False

    def handle_data(self, data):
        if self._in_strong:
            self._strong_parts.append(data)
        elif self._expect_model_name and not self.model_name and data.strip():
            self.model_name = data.strip()
            self._expect_model_name = False
        if self._in_section_heading:
            self._section_heading_parts.append(data)
        if self._in_cell:
            self._cell_parts.append(data)

    def optimal_f1_scores(self, report_path: Path) -> Tuple[str, Dict[str, float]]:
        if not self.model_name:
            raise ValueError(f"Could not find the model name in report: {report_path}")

        for rows in self._tables:
            if not rows:
                continue
            headers = [re.sub(r"\s+", "_", value.strip().lower()) for value in rows[0]]
            if "species" not in headers or "f1_score" not in headers:
                continue
            species_index = headers.index("species")
            f1_index = headers.index("f1_score")
            scores = {}
            for row in rows[1:]:
                if len(row) <= max(species_index, f1_index):
                    raise ValueError(f"Malformed optimal-threshold table in report: {report_path}")
                species = row[species_index]
                if not species:
                    continue
                try:
                    score = float(row[f1_index])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid F1 score for '{species}' in report: {report_path}"
                    ) from exc
                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"F1 score for '{species}' must be between 0 and 1 in report: {report_path}"
                    )
                if species in scores:
                    raise ValueError(f"Duplicate species '{species}' in report: {report_path}")
                scores[species] = score
            return self.model_name, scores

        raise ValueError(
            f"Could not find the 'Optimal Confidence Threshold by Species' F1 table in report: "
            f"{report_path}"
        )


def _read_evaluation_report(report_path: Path) -> Tuple[str, Dict[str, float]]:
    parser = _EvaluationReportParser()
    parser.feed(report_path.read_text(encoding="utf-8"))
    return parser.optimal_f1_scores(report_path)


def _build_summary(reports: List[Tuple[str, Dict[str, float]]]) -> pd.DataFrame:
    model_names = [model_name for model_name, _ in reports]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Each report must have a unique model name.")

    all_species = sorted({species for _, scores in reports for species in scores})
    rows = []
    for species in all_species:
        available_scores = {
            model_name: scores[species]
            for model_name, scores in reports
            if species in scores
        }
        highest_score = max(available_scores.values())
        best_models = [
            model_name for model_name, score in available_scores.items()
            if score == highest_score
        ]
        row = {"Species": species}
        for model_name, scores in reports:
            row[f"f1 score {model_name}"] = scores.get(species)
        row["Best performing model"] = best_models[0] if len(best_models) == 1 else "N/A"
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=["Species"]
        + [f"f1 score {model_name}" for model_name in model_names]
        + ["Best performing model"],
    )


def create_model_comparison_report(
    reports_folder: Union[str, Path], output_folder: Union[str, Path]
) -> Path:
    """Create an HTML comparison report from evaluation reports in a folder."""
    reports_dir = Path(reports_folder).expanduser().resolve()
    if not reports_dir.is_dir():
        raise NotADirectoryError(f"Evaluation reports folder does not exist: {reports_dir}")

    report_paths = sorted(reports_dir.glob(f"{REPORT_PREFIX}*.html"))
    if len(report_paths) < 2:
        raise ValueError(
            f"Expected at least two evaluation reports named "
            f"'{REPORT_PREFIX}*.html' in {reports_dir}, found {len(report_paths)}."
        )

    reports = [_read_evaluation_report(path) for path in report_paths]
    summary = _build_summary(reports)
    summary_html = summary.to_html(
        index=False, classes="styled-table", border=0, na_rep="N/A", escape=True
    )
    placeholders = {"Comparison_Summary_Table": Placeholder(summary_html, "html")}
    template = COMPARISON_TEMPLATE
    body = render_template(template, placeholders)
    templates_dir = template.parent
    style = load_style(template, templates_dir)
    fonts = embedded_font_css(templates_dir)

    output_dir = Path(output_folder).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"WildObs_CV_Model_Evaluation_Comparison_Report_{timestamp}.html"
    document = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>WildObs: CV Model Evaluation Comparison Report</title>"
        f"<style>{fonts}{style}</style></head>"
        f'<body><div class="report-container">{body}</div></body></html>'
    )
    report_path.write_text(document, encoding="utf-8")
    logger.info("Model comparison report saved to %s", report_path)
    return report_path
