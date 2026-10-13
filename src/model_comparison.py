"""Compare per-species optimal-threshold F1 scores from benchmark reports."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .report import embedded_font_css, embedded_logo_html
from .templating import Placeholder, load_style, render_template

logger = logging.getLogger(__name__)

REPORT_PREFIX = "WildObs_CV_Model_Evaluation_Report_"
COMPARISON_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "model_comparison.md"


DETAIL_LABELS = {
    "computer vision (cv) model tested:": "model_name",
    "timestamp of report:": "timestamp",
    "source location of test images:": "source_location",
}


@dataclass
class EvaluationReport:
    model_name: str
    timestamp: str
    source_location: str
    scores: Dict[str, float]
    species_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class _EvaluationReportParser(HTMLParser):
    """Extract the test details, optimal-threshold F1 table, and species metadata from a report."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.model_name = ""
        self.timestamp = ""
        self.source_location = ""
        self._in_strong = False
        self._strong_parts: List[str] = []
        self._pending_detail = ""
        self._in_section_heading = False
        self._section_heading_parts: List[str] = []
        self._in_best_threshold_section = False
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._table_rows: List[List[str]] = []
        self._tables: List[List[List[str]]] = []
        self._best_threshold_tables: List[List[List[str]]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "strong":
            self._in_strong = True
            self._strong_parts = []
        elif tag == "h2":
            self._in_section_heading = True
            self._section_heading_parts = []
        elif tag == "table":
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
            self._pending_detail = DETAIL_LABELS.get(label, "")
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
            if self._in_best_threshold_section:
                self._best_threshold_tables.append(self._table_rows)
                self._in_best_threshold_section = False
            self._tables.append(self._table_rows)
            self._in_table = False

    def handle_data(self, data):
        if self._in_strong:
            self._strong_parts.append(data)
        elif self._pending_detail and data.strip():
            if not getattr(self, self._pending_detail):
                setattr(self, self._pending_detail, data.strip())
            self._pending_detail = ""
        if self._in_section_heading:
            self._section_heading_parts.append(data)
        if self._in_cell:
            self._cell_parts.append(data)

    def optimal_f1_scores(self, report_path: Path) -> Tuple[str, Dict[str, float]]:
        if not self.model_name:
            raise ValueError(f"Could not find the model name in report: {report_path}")

        candidate_tables = self._best_threshold_tables if self._best_threshold_tables else self._tables

        for rows in candidate_tables:
            if not rows:
                continue
            headers = [re.sub(r"\s+", "_", value.strip().lower()) for value in rows[0]]
            if "species" not in headers or "f1_score" not in headers:
                continue
            if "true_positives" in headers and len(candidate_tables) > 1:
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

    def extract_species_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Extract species metadata (image count, ALA taxonomy) from the report."""
        for rows in self._tables:
            if not rows:
                continue
            headers = [re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_") for col in rows[0]]
            species_idx = None
            for idx, h in enumerate(headers):
                if h in ("species", "species_or_label", "species_label") or h.startswith("species"):
                    species_idx = idx
                    break
            if species_idx is None:
                continue

            num_images_idx = next(
                (i for i, h in enumerate(headers) if h in ("number_of_images", "image_count", "images", "num_images")),
                None,
            )
            matched_sci_idx = next(
                (i for i, h in enumerate(headers) if h in ("matched_scientific_name_ala", "matched_scientific_name")),
                None,
            )
            rank_idx = next(
                (i for i, h in enumerate(headers) if h in ("taxonomic_rank", "rank")),
                None,
            )
            common_name_idx = next(
                (i for i, h in enumerate(headers) if h in ("common_name", "vernacular_name")),
                None,
            )

            if all(idx is None for idx in (num_images_idx, matched_sci_idx, rank_idx, common_name_idx)):
                continue

            metadata: Dict[str, Dict[str, Any]] = {}
            for row in rows[1:]:
                if len(row) <= species_idx:
                    continue
                species = row[species_idx].strip()
                if not species:
                    continue

                num_img: Union[int, str] = ""
                if num_images_idx is not None and len(row) > num_images_idx:
                    val = row[num_images_idx].strip()
                    if val.isdigit():
                        num_img = int(val)
                    elif val:
                        num_img = val

                matched_sci = ""
                if matched_sci_idx is not None and len(row) > matched_sci_idx:
                    matched_sci = row[matched_sci_idx].strip()

                rank = ""
                if rank_idx is not None and len(row) > rank_idx:
                    rank = row[rank_idx].strip()

                common_name = ""
                if common_name_idx is not None and len(row) > common_name_idx:
                    common_name = row[common_name_idx].strip()

                metadata[species] = {
                    "Number of images": num_img,
                    "Matched scientific name (ALA)": matched_sci,
                    "Taxonomic Rank": rank,
                    "Common name": common_name,
                }
            return metadata
        return {}


def _read_evaluation_report(report_path: Path) -> EvaluationReport:
    parser = _EvaluationReportParser()
    parser.feed(report_path.read_text(encoding="utf-8"))
    model_name, scores = parser.optimal_f1_scores(report_path)
    metadata = parser.extract_species_metadata()
    return EvaluationReport(
        model_name,
        parser.timestamp or "Not specified",
        parser.source_location or "Not specified",
        scores,
        metadata,
    )


def _unique_joined(values: List[str]) -> str:
    return ", ".join(dict.fromkeys(values))


def _build_test_details(reports: List[EvaluationReport]) -> Dict[str, str]:
    """Aggregate the test details of every report for the template placeholders."""
    return {
        "Report_Timestamp": _unique_joined(sorted(r.timestamp for r in reports)),
        "Model_Names": _unique_joined([r.model_name for r in reports]),
        "Data_Source_Location": _unique_joined([r.source_location for r in reports]),
    }


def _build_summary(
    reports: Sequence[
        Union[
            EvaluationReport,
            Tuple[str, Dict[str, float]],
            Tuple[str, Dict[str, float], Dict[str, Dict[str, Any]]],
        ]
    ]
) -> pd.DataFrame:
    normalized_reports: List[Tuple[str, Dict[str, float], Dict[str, Dict[str, Any]]]] = []
    for r in reports:
        if isinstance(r, EvaluationReport):
            normalized_reports.append((r.model_name, r.scores, r.species_metadata))
        elif isinstance(r, (list, tuple)):
            if len(r) == 2:
                normalized_reports.append((r[0], r[1], {}))
            elif len(r) == 3:
                normalized_reports.append((r[0], r[1], r[2]))
            else:
                raise ValueError(f"Invalid report tuple: {r}")
        else:
            raise TypeError(f"Unsupported report type: {type(r)}")

    model_names = [model_name for model_name, _, _ in normalized_reports]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Each report must have a unique model name.")

    # Aggregate species metadata across reports (prefer non-empty values)
    aggregated_metadata: Dict[str, Dict[str, Any]] = {}
    for _, _, meta in normalized_reports:
        for species, fields in meta.items():
            if species not in aggregated_metadata:
                aggregated_metadata[species] = dict(fields)
            else:
                for k, v in fields.items():
                    current_val = aggregated_metadata[species].get(k, "")
                    if (current_val == "" or current_val == 0) and (v != "" and v is not None):
                        aggregated_metadata[species][k] = v

    all_species = sorted({species for _, scores, _ in normalized_reports for species in scores})
    rows = []
    for species in all_species:
        available_scores = {
            model_name: scores[species]
            for model_name, scores, _ in normalized_reports
            if species in scores
        }
        highest_score = max(available_scores.values())
        best_models = [
            model_name for model_name, score in available_scores.items()
            if score == highest_score
        ]

        meta = aggregated_metadata.get(species)
        if meta is None:
            sp_lower = species.lower()
            for s_name, s_meta in aggregated_metadata.items():
                if s_name.lower() == sp_lower:
                    meta = s_meta
                    break
        if meta is None:
            meta = {}

        row = {
            "Species": species,
            "Number of images": meta.get("Number of images", ""),
            "Matched scientific name (ALA)": meta.get("Matched scientific name (ALA)", ""),
            "Taxonomic Rank": meta.get("Taxonomic Rank", ""),
            "Common name": meta.get("Common name", ""),
        }
        for model_name, scores, _ in normalized_reports:
            row[f"f1 score {model_name}"] = scores.get(species)
        row["Best performing model"] = best_models[0] if len(best_models) == 1 else "N/A"
        rows.append(row)

    columns = (
        ["Species", "Number of images", "Matched scientific name (ALA)", "Taxonomic Rank", "Common name"]
        + [f"f1 score {model_name}" for model_name in model_names]
        + ["Best performing model"]
    )
    return pd.DataFrame(rows, columns=columns)


def create_model_comparison_report(
    reports_folder: Union[str, Path, Sequence[Union[str, Path]]],
    output_folder: Union[str, Path],
    run_id: Optional[str] = None,
) -> Path:
    """Create an HTML comparison report from evaluation reports.

    reports_folder can be:
    - A folder containing evaluation reports (WildObs_CV_Model_Evaluation_Report_*.html)
    - A list/sequence of Paths to specific evaluation report files.
    """
    if isinstance(reports_folder, (list, tuple, set)):
        report_paths = [Path(p).expanduser().resolve() for p in reports_folder]
        for p in report_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Evaluation report file not found: {p}")
        source_desc = "provided"
    else:
        reports_dir = Path(reports_folder).expanduser().resolve()
        if not reports_dir.is_dir():
            raise NotADirectoryError(f"Evaluation reports folder does not exist: {reports_dir}")
        report_paths = sorted(reports_dir.glob(f"{REPORT_PREFIX}*.html"))
        source_desc = f"in {reports_dir}"

    if len(report_paths) < 2:
        raise ValueError(
            f"Expected at least two evaluation reports named "
            f"'{REPORT_PREFIX}*.html' {source_desc}, found {len(report_paths)}."
        )

    reports = [_read_evaluation_report(path) for path in report_paths]
    summary = _build_summary(reports)
    summary_html = summary.to_html(
        index=False, classes="styled-table", border=0, na_rep="N/A", escape=True
    )
    placeholders = {"Comparison_Summary_Table": Placeholder(summary_html, "html")}
    placeholders.update({name: Placeholder(value) for name, value in _build_test_details(reports).items()})
    template = COMPARISON_TEMPLATE
    body = render_template(template, placeholders)
    templates_dir = template.parent
    style = load_style(template, templates_dir)
    fonts = embedded_font_css(templates_dir)

    output_dir = Path(output_folder).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run_id if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"WildObs_CV_Model_Evaluation_Comparison_Report_{timestamp}.html"
    document = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>WildObs: CV Model Evaluation Comparison Report</title>"
        f"<style>{fonts}{style}</style></head>"
        f'<body><div class="report-container">{embedded_logo_html(templates_dir)}{body}</div></body></html>'
    )
    report_path.write_text(document, encoding="utf-8")
    logger.info("Model comparison report saved to %s", report_path)
    return report_path
