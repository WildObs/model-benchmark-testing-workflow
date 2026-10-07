import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BenchmarkConfig  # noqa: E402
from src.data_loader import extract_sequence  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.pipeline import run_benchmark  # noqa: E402
from src.report import _sanitize_filename_component  # noqa: E402
from src.templating import Placeholder, list_sections, list_templates, render_template, resolve_template  # noqa: E402
from src.utils import threshold_label  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def make_camtrap(folder: Path):
    folder.mkdir(parents=True)
    pd.DataFrame({
        "deploymentID": ["d1", "d2"],
        "deploymentTags": ["Felis catus", "Vulpes vulpes"],
    }).to_csv(folder / "deployments.csv", index=False)
    pd.DataFrame({
        "mediaID": ["m1", "m2", "m3", "m4"],
        "deploymentID": ["d1", "d1", "d2", "d2"],
        "mediaComments": [f"sequenceID:e{i}" for i in range(1, 5)],
    }).to_csv(folder / "media.csv", index=False)
    pd.DataFrame({
        "eventID": ["e1", "e2", "e3", "e4"],
        "deploymentID": ["d1", "d1", "d2", "d2"],
        "observationType": ["animal"] * 4,
        "scientificName": ["Felis catus", "Vulpes vulpes", "Vulpes vulpes", "Vulpes vulpes"],
        "classificationProbability": ["0.9", "0.8", "0.95", "0.4"],
        "classifiedBy": ["test model"] * 4,
    }).to_csv(folder / "observations.csv", index=False)


def test_count_images_per_species(tmp_path):
    from src.data_loader import count_images_per_species, load_camtrap_data

    make_camtrap(tmp_path / "export")
    _, media, deployments = load_camtrap_data(tmp_path / "export")
    assert count_images_per_species(media, deployments) == {"Felis catus": 2, "Vulpes vulpes": 2}


def test_extract_sequence():
    assert extract_sequence("foo sequenceID:abc-1 bar") == "abc-1"
    assert extract_sequence(None) is None


def test_threshold_label_avoids_collisions():
    assert threshold_label(0.5) == ">=0.5"
    assert threshold_label(0.0) == ">=0.0"
    assert threshold_label(0.05) != threshold_label(0.15)


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        BenchmarkConfig("x", thresholds=[1.5])
    with pytest.raises(ValueError):
        BenchmarkConfig("x", thresholds=[])
    config = BenchmarkConfig("x", thresholds=[0.9, 0.5, 0.5], input_path=tmp_path)
    assert config.thresholds == [0.5, 0.9]


def test_report_filename_component_replaces_spaces_and_special_characters():
    assert _sanitize_filename_component("North-West / Plot #2") == "North_West_Plot_2"
    assert _sanitize_filename_component("Model (v2)!") == "Model_v2"


def test_metrics():
    merged = pd.DataFrame({
        "true_species": ["cat", "cat", "fox", "fox"],
        "pred_species": ["cat", "fox", "fox", "fox"],
        "confidence": [0.9, 0.8, 0.95, 0.4],
    })
    metrics = compute_metrics(merged, [0.5])
    cat = metrics.results.set_index("species").loc["Cat"]
    assert (cat.true_positives, cat.false_negatives, cat.false_positives) == (1, 1, 0)
    assert cat.recall == 0.5 and cat.precision == 1.0
    fox = metrics.results.set_index("species").loc["Fox"]
    assert (fox.true_positives, fox.false_negatives, fox.false_positives) == (1, 1, 1)
    assert "threshold_value" not in metrics.results.columns


def test_template_sections_and_placeholders(tmp_path):
    template = tmp_path / "t.md"
    template.write_text(
        "<!-- description: demo -->\n# Title {Name}\n\n<!-- BEGIN:A -->\nKeep {Name}\n<!-- END:A -->\n\n"
        "<!-- BEGIN:B -->\nDrop me\n<!-- END:B -->\n\n\\[ \\text{Recall} = \\frac{TP}{TP + FN} \\]\n\n{Table}\n"
    )
    assert list_templates(tmp_path) == {"t": "demo"}
    assert list_sections(template) == ["A", "B"]
    placeholders = {"Name": Placeholder("<x>"), "Table": Placeholder("<table><tr><td>1</td></tr></table>", "html")}
    out = render_template(template, placeholders, exclude_sections=["B"])
    assert "Keep &lt;x&gt;" in out
    assert "Drop me" not in out
    assert "\\text{Recall} = \\frac{TP}{TP + FN}" in out
    assert "<table><tr><td>1</td></tr></table>" in out
    assert "description" not in out


def test_resolve_template():
    with pytest.raises(FileNotFoundError):
        resolve_template("nope", TEMPLATES)
    assert resolve_template("full", TEMPLATES).name == "full.md"
    assert set(list_templates(TEMPLATES)) >= {"full", "minimal"}


@pytest.mark.parametrize("template, has_limitations", [("full", True), ("minimal", False)])
def test_end_to_end(tmp_path, template, has_limitations):
    make_camtrap(tmp_path / "in" / "export")
    config = BenchmarkConfig(
        "export", data_source_location="North-West / Plot #2", thresholds=[0.5], template=template,
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    run = run_benchmark(config)
    run_id = run.log_path.stem.removeprefix("benchmark_")
    assert run.report_path.name == (
        f"WildObs_CV_Model_Evaluation_Report_North_West_Plot_2_test_model_{run_id}.html"
    )
    report = run.report_path.read_text(encoding="utf-8")
    assert ("Limitations" in report) is has_limitations
    assert "Felis catus" in report and "test model" in report
    assert "{Model_Name}" not in report
    assert run.misclassified_path.is_file()
    assert run.log_path.is_file()


def test_exclude_sections_end_to_end(tmp_path):
    make_camtrap(tmp_path / "in" / "export")
    config = BenchmarkConfig(
        "export", thresholds=[0.5], exclude_sections=["Limitations", "Full_Confusion_Matrices"],
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    report = run_benchmark(config).report_path.read_text(encoding="utf-8")
    assert "Limitations" not in report
    assert "Full Confusion Matrices" not in report
    assert "Purpose and intended use" in report


def test_taxonomy_table_blank_for_no_match():
    from src.taxonomy import build_taxonomy_table

    def fake_search(taxa):
        if taxa == ["Macropus rufus"]:
            return pd.DataFrame({"scientificName": ["Osphranter rufus"], "family": ["Macropodidae"], "issues": ["noIssue"]})
        if taxa == ["Boom"]:
            raise RuntimeError("network down")
        return pd.DataFrame()

    table = build_taxonomy_table(["Macropus rufus", "Blank", "Boom"], image_counts={"Macropus rufus": 12}, search_taxa=fake_search)
    assert list(table["Species (or label)"]) == ["Macropus rufus", "Blank", "Boom"]
    assert table.loc[0, "Matched scientific name (ALA)"] == "Osphranter rufus"
    assert table.loc[0, "Family"] == "Macropodidae"
    assert list(table["Number of images"]) == [12, 0, 0]
    assert table.loc[0, "Kingdom"] == ""
    assert (table.loc[1:, "Matched scientific name (ALA)"] == "").all()


def test_end_to_end_with_taxonomy(tmp_path, monkeypatch):
    import src.report_components as rc

    monkeypatch.setattr(
        rc, "build_taxonomy_table",
        lambda labels, counts: pd.DataFrame({"Species (or label)": labels, "Family": ["TestFamily"] * len(labels)}),
    )
    make_camtrap(tmp_path / "in" / "export")
    config = BenchmarkConfig(
        "export", thresholds=[0.5], lookup_taxonomy=True,
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    report = run_benchmark(config).report_path.read_text(encoding="utf-8")
    assert "TestFamily" in report
