import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BenchmarkConfig  # noqa: E402
from src.data_loader import extract_sequence  # noqa: E402
from src.metrics import compute_animal_recall, compute_metrics  # noqa: E402
from src.model_comparison import (  # noqa: E402
    EvaluationReport,
    _EvaluationReportParser,
    _build_model_performance_summary,
    _build_summary,
    create_model_comparison_report,
)
from src.pipeline import run_benchmark  # noqa: E402
from src.report import _sanitize_filename_component  # noqa: E402
from src.templating import Placeholder, list_sections, list_templates, render_template, resolve_template  # noqa: E402
from src.utils import threshold_label  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def make_camtrap(folder: Path, model_name: str = "test model"):
    folder.mkdir(parents=True, exist_ok=True)
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
        "classifiedBy": [model_name] * 4,
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
        "mediaID": ["m1", "m2", "m3", "m4"],
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
    assert "<math" in out and "<mfrac>" in out and "Recall" in out
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
        export_errors=True,
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    run = run_benchmark(config)
    run_id = run.log_path.stem.removeprefix("benchmark_")
    assert run.report_path.name == (
        f"WildObs_CV_Model_Evaluation_Report_North_West_Plot_2_test_model_{run_id}.html"
    )
    report = run.report_path.read_text(encoding="utf-8")
    assert ("Limitations" in report) is has_limitations
    assert report.index('class="report-logo"') < report.index("<h1>")
    assert "Felis catus" in report and "test model" in report
    assert "{Model_Name}" not in report
    assert run.misclassified_path.is_file()
    assert run.log_path.is_file()


def test_species_subset_end_to_end(tmp_path):
    make_camtrap(tmp_path / "in" / "export")
    config = BenchmarkConfig(
        "export", thresholds=[0.5], species_subset=["Felis catus"], export_errors=True,
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    run = run_benchmark(config)
    report = run.report_path.read_text(encoding="utf-8")
    assert "Felis catus" in report
    # Vulpes vulpes is still listed as a (mis)prediction, but has no results of its own
    assert set(run.metrics.results["species"]) == {"Felis catus"}
    assert set(run.metrics.single_row_matrices) == {"Felis catus"}


def test_species_subset_unknown_species_fails(tmp_path):
    make_camtrap(tmp_path / "in" / "export")
    config = BenchmarkConfig(
        "export", thresholds=[0.5], species_subset=["Nothing here"],
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    with pytest.raises(ValueError, match="None of the selected species"):
        run_benchmark(config)


def test_higher_taxon_matching_metrics():
    from src.taxonomy import build_match_map

    taxa = {
        "lagomorpha": {"scientificName": "LAGOMORPHA", "rank": "order", "kingdom": "Animalia", "order": "Lagomorpha"},
        "oryctolagus cuniculus": {"scientificName": "Oryctolagus cuniculus", "rank": "species",
                                  "order": "Lagomorpha", "genus": "Oryctolagus", "species": "Oryctolagus cuniculus"},
        "vulpes vulpes": {"scientificName": "Vulpes vulpes", "rank": "species", "order": "Carnivora",
                          "species": "Vulpes vulpes"},
    }

    def fake_search(taxa=None):
        return pd.DataFrame([taxa_map[taxa[0]]]) if taxa[0] in taxa_map else pd.DataFrame()

    taxa_map = taxa
    merged = pd.DataFrame({
        "true_species": ["lagomorpha", "lagomorpha", "vulpes vulpes", "oryctolagus cuniculus"],
        "pred_species": ["oryctolagus cuniculus", "vulpes vulpes", "vulpes vulpes", "oryctolagus cuniculus"],
        "confidence": [0.9] * 4,
        "mediaID": list("abcd"),
    })
    match_map = build_match_map(sorted(merged.true_species.unique()), sorted(merged.pred_species.unique()), fake_search)
    assert match_map == {"lagomorpha": {"oryctolagus cuniculus"}}

    def row(metrics, name):
        return metrics.results.query("species == @name").iloc[0]

    exact = compute_metrics(merged, [0.5])
    assert row(exact, "Lagomorpha").recall == 0
    loose = compute_metrics(merged, [0.5], match_map)
    assert (row(loose, "Lagomorpha").recall, row(loose, "Lagomorpha").precision) == (0.5, 1.0)
    # the specific species is unaffected, and the confusion matrix stays exact
    assert row(loose, "Oryctolagus cuniculus").false_positives == row(exact, "Oryctolagus cuniculus").false_positives == 1
    assert loose.full_confusion_matrices[">=0.5"].equals(exact.full_confusion_matrices[">=0.5"])


def test_export_subset_end_to_end(tmp_path):
    make_camtrap(tmp_path / "in" / "set" / "a", "model a")
    make_camtrap(tmp_path / "in" / "set" / "b", "model b")
    kwargs = dict(thresholds=[0.5], input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs")
    both = run_benchmark(BenchmarkConfig("set", **kwargs))
    assert both.model_names == ["model a", "model b"] and both.comparison_report_path
    one = run_benchmark(BenchmarkConfig("set", export_subset=["b"], **kwargs))
    assert one.model_names == ["model b"] and one.comparison_report_path is None
    with pytest.raises(ValueError, match="not found"):
        run_benchmark(BenchmarkConfig("set", export_subset=["zzz"], **kwargs))


def test_custom_substitutions():
    from src.taxonomy import build_custom_match_map, merge_match_maps
    from src.ui import parse_substitutions

    assert parse_substitutions(" Leporidae, Lepus europaeus ;,") == ["Leporidae", "Lepus europaeus"]
    taxa_map = {
        "lepus europaeus": {"scientificName": "Lepus europaeus", "rank": "species", "family": "Leporidae",
                            "genus": "Lepus", "species": "Lepus europaeus"},
        "oryctolagus cuniculus": {"scientificName": "Oryctolagus cuniculus", "rank": "species", "family": "Leporidae",
                                  "genus": "Oryctolagus", "species": "Oryctolagus cuniculus"},
        "vulpes vulpes": {"scientificName": "Vulpes vulpes", "rank": "species", "family": "Canidae",
                          "species": "Vulpes vulpes"},
        "leporidae": {"scientificName": "LEPORIDAE", "rank": "family", "family": "Leporidae"},
    }

    def fake_search(taxa=None):
        return pd.DataFrame([taxa_map[taxa[0]]]) if taxa[0] in taxa_map else pd.DataFrame()

    preds = ["lepus europaeus", "oryctolagus cuniculus", "vulpes vulpes", "blank"]
    got = build_custom_match_map({"Sylvilagus": ["Leporidae"], "Cat": ["Vulpes vulpes"]}, preds, fake_search)
    assert got == {"sylvilagus": {"lepus europaeus", "oryctolagus cuniculus"}, "cat": {"vulpes vulpes"}}
    assert merge_match_maps({"a": {"x"}}, {"a": {"y"}}, None) == {"a": {"x", "y"}}

    merged = pd.DataFrame({
        "true_species": ["sylvilagus"] * 2, "pred_species": ["lepus europaeus", "vulpes vulpes"],
        "confidence": [0.9, 0.9], "mediaID": ["a", "b"],
    })
    result = compute_metrics(merged, [0.5], got).results.iloc[0]
    assert result.recall == 0.5


def test_custom_substitutions_end_to_end(tmp_path):
    make_camtrap(tmp_path / "in" / "export")
    config = BenchmarkConfig(
        "export", thresholds=[0.5], custom_matches={"Felis catus": ["Vulpes vulpes"]},
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    run = run_benchmark(config)
    cat = run.metrics.results.query("species == 'Felis catus'").iloc[0]
    assert cat.recall == 1.0


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


def make_mixed_camtrap(folder: Path, model_name: str = "test model"):
    """Camtrap export with event rows plus media rows that deliberately disagree with them."""
    make_camtrap(folder, model_name)
    obs = pd.read_csv(folder / "observations.csv", dtype=str)
    obs["observationLevel"] = "event"
    obs["mediaID"] = pd.NA
    media_rows = pd.DataFrame({
        "eventID": ["e1", "e2", "e3"],
        "deploymentID": ["d1", "d1", "d2"],
        "mediaID": ["m1", "m2", "m3"],
        "observationLevel": "media",
        "observationType": "animal",
        "scientificName": ["Vulpes vulpes"] * 3,
        "classificationProbability": ["0.9"] * 3,
        "classifiedBy": model_name,
    })
    pd.concat([obs, media_rows], ignore_index=True).to_csv(folder / "observations.csv", index=False)


def test_observation_level_selection(tmp_path):
    from src.data_loader import load_benchmark_table

    mixed = tmp_path / "mixed"
    make_mixed_camtrap(mixed)
    event, *_ = load_benchmark_table(mixed, "event")
    media, *_ = load_benchmark_table(mixed, "media")
    assert len(event) == 4 and set(event["pred_species"]) == {"felis catus", "vulpes vulpes"}
    # m4 has no media-level observation: kept with its true label and scored as a blank prediction
    assert len(media) == 4 and set(media["pred_species"]) == {"vulpes vulpes", "blank"}
    assert media.set_index("mediaID").loc["m4", "true_species"] == "vulpes vulpes"
    with pytest.raises(ValueError):
        load_benchmark_table(mixed, "both")


def test_media_level_reverts_to_event_when_any_export_lacks_it(tmp_path):
    from src.data_loader import resolve_observation_level

    legacy, mixed = tmp_path / "legacy", tmp_path / "mixed"
    make_camtrap(legacy)
    make_mixed_camtrap(mixed)
    assert resolve_observation_level("media", [mixed]) == "media"
    assert resolve_observation_level("media", [mixed, legacy]) == "event"
    assert resolve_observation_level("event", [mixed]) == "event"


def test_observation_level_config_and_pipeline(tmp_path):
    with pytest.raises(ValueError):
        BenchmarkConfig("x", observation_level="both")
    assert BenchmarkConfig("x").observation_level == "event"

    make_mixed_camtrap(tmp_path / "in" / "set" / "a")
    make_camtrap(tmp_path / "in" / "set" / "b", model_name="other model")
    config = BenchmarkConfig(
        "set", thresholds=[0.5], observation_level="media",
        input_path=tmp_path / "in", output_path=tmp_path / "out", log_path=tmp_path / "logs",
    )
    run = run_benchmark(config)
    # Reverted to event for both exports: model "test model" scores cat 1 of 2 as in the event data
    cat = run.runs[0].metrics.results.set_index("species").loc["Felis catus"]
    assert cat.true_positives == 1


def test_ui_helpers(tmp_path):
    from src.ui import effective_observation_level, list_test_sets, make_config, parse_thresholds, summarise_set

    assert parse_thresholds("0.5, 0.7;0.9") == [0.5, 0.7, 0.9]
    for bad in ("", "abc", "1.5"):
        with pytest.raises(ValueError):
            parse_thresholds(bad)

    make_mixed_camtrap(tmp_path / "in" / "set" / "a")
    make_camtrap(tmp_path / "in" / "set" / "b", model_name="old model")
    (tmp_path / "in" / "empty").mkdir()
    assert list_test_sets(tmp_path / "in") == ["set"]

    summaries = summarise_set(tmp_path / "in" / "set")
    by_model = {s.model: s for s in summaries}
    assert by_model["test model"].levels == ["event", "media"] and by_model["test model"].images == 4
    assert by_model["old model"].levels == ["event"]
    assert effective_observation_level("media", summaries) == "event"
    assert effective_observation_level("media", summaries[:1]) == "media"

    with pytest.raises(ValueError):
        make_config("", "", "", "0.5", "event", "full", [], False, False, False)
    config = make_config("set", " here ", "", "0.5, 0.8", "media", "full", ["Limitations"], True, False, True,
                         input_path=str(tmp_path / "in"))
    assert config.thresholds == [0.5, 0.8] and config.data_source_location == "here"
    assert config.observation_level == "media" and config.clear_existing_outputs is True


def test_ui_form_runs(tmp_path):
    ipywidgets = pytest.importorskip("ipywidgets")
    from src.ui import build_form

    root = tmp_path / "proj"
    make_camtrap(root / "Input_Data" / "set" / "export")
    (root / "templates").mkdir(parents=True)
    for src_file in TEMPLATES.iterdir():
        if src_file.is_file():
            (root / "templates" / src_file.name).write_bytes(src_file.read_bytes())

    form = build_form(root)

    def find(widget_type, **attrs):
        stack = list(form.children)
        while stack:
            item = stack.pop()
            if isinstance(item, widget_type) and all(getattr(item, k) == v for k, v in attrs.items()):
                return item
            stack.extend(getattr(item, "children", ()))
        raise AssertionError(f"{widget_type} {attrs} not found")

    test_set = find(ipywidgets.Dropdown, description="Test set folder")
    assert test_set.value is None
    find(ipywidgets.Button, description="Run evaluation").click()
    assert "Choose a test set" in [c for c in form.children if isinstance(c, ipywidgets.HTML)][-1].value
    test_set.value = "set"

    thresholds = find(ipywidgets.Dropdown, description="Confidence thresholds to include")
    assert thresholds.value == "0.5, 0.6, 0.7, 0.8, 0.9"
    assert [label for label, _ in thresholds.options][0] == ">0.5" and len(thresholds.options) == 4
    assert "model_comparison" not in find(ipywidgets.Dropdown, description="Report template").options
    assert find(ipywidgets.RadioButtons, description="Taxonomy matching preferences").value == "exact"
    assert find(ipywidgets.RadioButtons, description="Observation Levels to include in the analysis").options[0][0] == "Event-based (recommended)"
    toggle = find(ipywidgets.Checkbox, description="Specify acceptable substitutions for each input species")
    texts = [find(ipywidgets.Text, description=n) for n in ("Felis catus", "Vulpes vulpes")]
    assert not toggle.value and len(texts) == 2
    model_mode = find(ipywidgets.RadioButtons, description="Model subset")
    files = find(ipywidgets.SelectMultiple, description="Input files to include")
    assert model_mode.value == "all" and files.layout.display == "none"
    assert files.options == (("test model (export)", "export"),)
    model_mode.value = "subset"
    assert files.layout.display == "block"
    find(ipywidgets.Button, description="Run evaluation").click()
    assert "Select at least one input file" in [c for c in form.children if isinstance(c, ipywidgets.HTML)][-1].value
    model_mode.value = "all"
    mode = find(ipywidgets.RadioButtons, description="Species subset")
    species = find(ipywidgets.SelectMultiple, description="Species to include")
    assert species.options == ("Felis catus", "Vulpes vulpes") and species.layout.display == "none"
    mode.value = "subset"
    assert species.layout.display == "block"
    find(ipywidgets.Button, description="Run evaluation").click()
    assert "Select at least one species" in [c for c in form.children if isinstance(c, ipywidgets.HTML)][-1].value
    mode.value = "all"
    advanced = next(c for c in form.children if isinstance(c, ipywidgets.Accordion) and c.titles == ("Advanced",))
    log_box = next(c for c in form.children if isinstance(c, ipywidgets.Accordion) and c.titles == ("Log messages",))
    assert advanced.selected_index is None and log_box.selected_index is None
    export_errors = find(ipywidgets.Checkbox, description="Export misclassified images csv")
    taxonomy = find(ipywidgets.Checkbox, description="Look up taxonomy (needs internet and galah)")
    clear = find(ipywidgets.Checkbox, description="Clear existing outputs for this set first")
    assert (export_errors.value, taxonomy.value, clear.value) == (False, True, True)
    assert {export_errors, taxonomy, clear} <= set(advanced.children[0].children)

    status = [c for c in form.children if isinstance(c, ipywidgets.HTML)][-1]
    taxonomy.value = False  # avoids a network lookup in the test
    thresholds.value = "0.2, 0.4, 0.6, 0.8"
    find(ipywidgets.Button, description="Run evaluation").click()
    assert "Done" in status.value
    assert list((root / "Output_Reports" / "set").glob("*.html"))
    reports = [c for c in form.children if isinstance(c, ipywidgets.VBox) and c.children][-1]
    accordion = reports.children[0].children[1]
    n_reports = len(list((root / "Output_Reports" / "set").glob("*.html")))
    assert len(accordion.children) == n_reports and accordion.selected_index is None
    accordion.selected_index = 0  # lazily renders the first report (needs a live kernel to capture output)


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


def _write_evaluation_report(
    path, model_name, f1_rows, timestamp="2026-10-08 10:14:05", location="Edgbaston Reserve, QLD"
):
    rows = "".join(
        f"<tr><td>{species}</td><td>&gt;=0.5</td><td>0.8</td><td>0.8</td><td>{score}</td></tr>"
        for species, score in f1_rows
    )
    path.write_text(
        "<html><head><title>Model Evaluation Report</title></head><body>"
        f"<p><strong>Timestamp of report:</strong> {timestamp}</p>"
        f"<p><strong>Computer Vision (CV) model tested:</strong> {model_name}</p>"
        f"<p><strong>Source location of test images:</strong> {location}</p>"
        "<h2>Optimal Confidence Threshold by Species</h2>"
        "<table><thead><tr><th>species</th><th>model_confidence</th><th>recall</th>"
        f"<th>precision</th><th>f1_score</th></tr></thead><tbody>{rows}</tbody></table>"
        "</body></html>",
        encoding="utf-8",
    )


def test_add_navigation_links_every_h2():
    from src.templating import add_navigation

    body = "<h1>T</h1><h2>Intro</h2><p>x</p><h2>Results &amp; more</h2><p>y</p>"
    result = add_navigation(body, "k")
    assert result.startswith('<nav class="report-nav"')
    assert result.count("<a href=") == 2
    assert "Results &amp; more</a>" in result
    ids = [part.split('"')[0] for part in result.split('<h2 id="')[1:]]
    assert len(ids) == 2 and all(f'href="#{i}"' in result for i in ids)
    assert add_navigation("<h2>Only</h2>", "k") == "<h2>Only</h2>"


def test_compute_animal_recall_counts_each_image_once():
    merged = pd.DataFrame({
        "mediaID": ["m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7"],
        "true_species": ["deer"] * 6 + ["blank", "human"],
        "pred_species": ["deer", "fox", "blank", "blank", "unclassified", "deer", "blank", "blank"],
    })
    recall = compute_animal_recall(merged)
    assert (recall.animal_images, recall.blank, recall.unclassified) == (5, 2, 1)


def test_compute_animal_recall_counts_blank_and_unclassified():
    merged = pd.DataFrame({
        "mediaID": [f"m{i}" for i in range(9)],
        "true_species": ["deer"] * 6 + ["blank", "blank", "human"],
        "pred_species": ["deer", "deer", "blank", "blank", "unclassified", "fox", "blank", "deer", "blank"],
    })
    recall = compute_animal_recall(merged)
    assert (recall.animal_images, recall.blank, recall.unclassified) == (6, 2, 1)
    assert recall.detector_recall == pytest.approx(0.6667, abs=1e-4)
    assert recall.classifier_recall == pytest.approx(0.75)
    assert recall.overall_recall == pytest.approx(0.5)


def test_compute_animal_recall_without_animals_is_none():
    recall = compute_animal_recall(pd.DataFrame({"mediaID": ["m1"], "true_species": ["blank"], "pred_species": ["blank"]}))
    assert recall.detector_recall is None and recall.overall_recall is None


def test_model_comparison_report_includes_animal_recall_section(tmp_path):
    from src.metrics import AnimalRecall

    reports = []
    for name in ("A", "B"):
        path = tmp_path / f"WildObs_CV_Model_Evaluation_Report_{name}.html"
        _write_evaluation_report(path, name, [("Cat", "0.85")])
        reports.append(path)
    recall = {"A": AnimalRecall(10, 2, 1), "B": AnimalRecall(10, 0, 0)}
    out = create_model_comparison_report(reports, tmp_path / "out", animal_recall=recall)
    content = out.read_text(encoding="utf-8")
    assert "Animal Recall" in content
    assert "Detector Model Animal Recall" in content
    assert "0.8000" in content


def test_model_comparison_extracts_optimal_threshold_f1_scores(tmp_path):
    report_path = tmp_path / "evaluation.html"
    _write_evaluation_report(report_path, "Camera Model A", [("Cat", "0.8123")])

    parser = _EvaluationReportParser()
    parser.feed(report_path.read_text(encoding="utf-8"))

    assert parser.optimal_f1_scores(report_path) == ("Camera Model A", {"Cat": 0.8123})


def test_model_comparison_summary_handles_ties_and_missing_species():
    summary = _build_summary([
        ("Model A", {"Cat": 0.8, "Fox": 0.5}),
        ("Model B", {"Cat": 0.8, "Fox": 0.7}),
        ("Model C", {"Cat": 0.7, "Owl": 0.9}),
    ])

    cat = summary.set_index("Species").loc["Cat"]
    fox = summary.set_index("Species").loc["Fox"]
    owl = summary.set_index("Species").loc["Owl"]
    assert cat["Best performing model"] == "N/A"
    assert fox["Best performing model"] == "Model B"
    assert owl["Best performing model"] == "Model C"
    assert pd.isna(owl["f1 score Model A"])
    assert list(summary.columns) == [
        "Species",
        "Number of images",
        "Matched scientific name (ALA)",
        "Taxonomic Rank",
        "Common name",
        "f1 score Model A",
        "f1 score Model B",
        "f1 score Model C",
        "Best performing model",
    ]


def test_model_comparison_report_reads_folder_and_uses_shared_style(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_evaluation_report(
        reports_dir / "WildObs_CV_Model_Evaluation_Report_Location_Model_A_20261008.html",
        "Model A",
        [("Cat", "0.8")],
        timestamp="2026-10-08 10:20:00",
    )
    _write_evaluation_report(
        reports_dir / "WildObs_CV_Model_Evaluation_Report_Location_Model_B_20261008.html",
        "Model B",
        [("Cat", "0.9")],
    )

    report_path = create_model_comparison_report(str(reports_dir), str(tmp_path / "out"))
    report = report_path.read_text(encoding="utf-8")

    assert report_path.name.startswith("WildObs_CV_Model_Evaluation_Comparison_Report_")
    assert "f1 score Model A" in report and "f1 score Model B" in report
    assert "Model B" in report and "0.9" in report
    assert "font-family:'Poppins'" in report
    assert "styled-table" in report
    assert 'class="report-logo"' in report and "data:image/avif;base64," in report
    assert report.index("report-logo") < report.index("<h1>")
    assert "<strong>Timestamps of input evaluation reports:</strong> 2026-10-08 10:14:05, 2026-10-08 10:20:00" in report
    assert "<strong>Computer Vision (CV) models tested:</strong> Model A, Model B" in report
    assert "<strong>Source location of test images:</strong> Edgbaston Reserve, QLD</p>" in report
    assert "<th>Number of images</th>" in report
    assert "<th>Matched scientific name (ALA)</th>" in report
    assert "<th>Taxonomic Rank</th>" in report
    assert "<th>Common name</th>" in report
    assert "Model performance summary" in report
    assert "<th>Average Recall</th>" in report
    assert "<th>Average Precision</th>" in report
    assert "<th>Average F1 Score</th>" in report
    assert "0.8000" in report


def test_build_model_performance_summary_computes_averages():
    rep_a = EvaluationReport(
        model_name="Model A",
        timestamp="2026-10-08 10:00:00",
        source_location="Location A",
        scores={"Cat": 0.8, "Fox": 0.6},
        species_metadata={},
        species_metrics={
            "Cat": {"recall": 0.8, "precision": 0.8, "f1_score": 0.8},
            "Fox": {"recall": 0.6, "precision": 0.6, "f1_score": 0.6},
        },
    )
    rep_b = EvaluationReport(
        model_name="Model B",
        timestamp="2026-10-08 10:00:00",
        source_location="Location A",
        scores={"Cat": 0.9, "Fox": 0.7},
        species_metadata={},
        species_metrics={
            "Cat": {"recall": 1.0, "precision": 0.8, "f1_score": 0.9},
            "Fox": {"recall": 0.8, "precision": 0.6, "f1_score": 0.7},
        },
    )
    df = _build_model_performance_summary([rep_a, rep_b])
    assert list(df.columns) == ["Model", "Average Recall", "Average Precision", "Average F1 Score"]
    assert len(df) == 2
    row_a = df[df["Model"] == "Model A"].iloc[0]
    assert row_a["Average Recall"] == 0.7
    assert row_a["Average Precision"] == 0.7
    assert row_a["Average F1 Score"] == 0.7

    row_b = df[df["Model"] == "Model B"].iloc[0]
    assert row_b["Average Recall"] == 0.9
    assert row_b["Average Precision"] == 0.7
    assert row_b["Average F1 Score"] == 0.8


def test_build_model_performance_summary_handles_missing_metrics():
    # If recall and precision are not available, average recall and precision are None, f1 is computed
    rep = ("Model Solo", {"Cat": 0.8, "Fox": 0.6})
    df = _build_model_performance_summary([rep])
    assert list(df.columns) == ["Model", "Average Recall", "Average Precision", "Average F1 Score"]
    row = df.iloc[0]
    assert row["Model"] == "Model Solo"
    assert row["Average Recall"] is None
    assert row["Average Precision"] is None
    assert row["Average F1 Score"] == 0.7


def test_model_comparison_extracts_and_populates_species_metadata(tmp_path):
    def _write_rich_report(path, model_name, f1_score):
        path.write_text(
            "<html><body>"
            f"<p><strong>Computer Vision (CV) model tested:</strong> {model_name}</p>"
            "<p><strong>Timestamp of report:</strong> 2026-10-13 10:00:00</p>"
            "<p><strong>Source location of test images:</strong> Test Location</p>"
            "<p><strong>Species (or labels) evaluated:</strong></p>"
            '<table class="styled-table">'
            "<thead><tr>"
            "<th>Species (or label)</th><th>Number of images</th>"
            "<th>Matched scientific name (ALA)</th><th>Taxonomic Rank</th><th>Common name</th>"
            "</tr></thead><tbody>"
            "<tr><td>Felis catus</td><td>25</td><td>Felis catus</td><td>species</td><td>Cat</td></tr>"
            "<tr><td>Vulpes vulpes</td><td>10</td><td>Vulpes vulpes</td><td>species</td><td>Red Fox</td></tr>"
            "</tbody></table>"
            "<h2>Optimal Confidence Threshold by Species</h2>"
            "<table><thead><tr><th>species</th><th>f1_score</th></tr></thead><tbody>"
            f"<tr><td>Felis catus</td><td>{f1_score}</td></tr>"
            "<tr><td>Vulpes vulpes</td><td>0.75</td></tr>"
            "</tbody></table>"
            "</body></html>",
            encoding="utf-8",
        )

    r1 = tmp_path / "WildObs_CV_Model_Evaluation_Report_Loc_M1_20261013.html"
    r2 = tmp_path / "WildObs_CV_Model_Evaluation_Report_Loc_M2_20261013.html"
    _write_rich_report(r1, "Model 1", 0.90)
    _write_rich_report(r2, "Model 2", 0.85)

    report_path = create_model_comparison_report([r1, r2], tmp_path / "out")
    content = report_path.read_text(encoding="utf-8")

    assert "<td>Felis catus</td>" in content
    assert "<td>25</td>" in content
    assert "<td>Cat</td>" in content
    assert "<td>Vulpes vulpes</td>" in content
    assert "<td>10</td>" in content
    assert "<td>Red Fox</td>" in content
    assert "<td>0.9</td>" in content or "<td>0.90" in content
    assert "<td>Model 1</td>" in content


def test_batch_run_multiple_models_in_set(tmp_path):
    set_name = "Wychitella_2026_09"
    set_dir = tmp_path / "in" / set_name
    make_camtrap(set_dir / "export_model_1", model_name="AddaxAI Vic")
    make_camtrap(set_dir / "export_model_2", model_name="AWC 135")

    config = BenchmarkConfig(
        test_set_name=set_name,
        data_source_location="Wychitella Reserves, VIC",
        thresholds=[0.5],
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        log_path=tmp_path / "logs",
    )
    results = run_benchmark(config)

    assert results.set_name == set_name
    assert len(results.runs) == 2
    assert sorted(results.model_names) == ["AWC 135", "AddaxAI Vic"]
    assert results.comparison_report_path is not None
    assert results.comparison_report_path.is_file()
    assert results.report_path == results.comparison_report_path

    # Check output directory structure: Output_Reports / set_name
    set_out = tmp_path / "out" / set_name
    assert results.comparison_report_path.parent == set_out
    for r in results.runs:
        assert r.report_path.parent == set_out
        assert r.report_path.is_file()

    # Comparison report should contain both models, location, and performance summary table
    comparison_html = results.comparison_report_path.read_text(encoding="utf-8")
    assert "AddaxAI Vic" in comparison_html
    assert "AWC 135" in comparison_html
    assert "Wychitella Reserves, VIC" in comparison_html
    assert "Model performance summary" in comparison_html
    assert "<th>Average Recall</th>" in comparison_html
    assert "<th>Average Precision</th>" in comparison_html
    assert "<th>Average F1 Score</th>" in comparison_html


def test_batch_run_single_model_skips_comparison_report(tmp_path):
    set_name = "Edgbaston_Single_Model"
    set_dir = tmp_path / "in" / set_name
    make_camtrap(set_dir / "export_solo", model_name="Solo Model")

    config = BenchmarkConfig(
        test_set_name=set_name,
        thresholds=[0.5],
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        log_path=tmp_path / "logs",
    )
    results = run_benchmark(config)

    assert results.set_name == set_name
    assert len(results.runs) == 1
    assert results.model_names == ["Solo Model"]
    assert results.comparison_report_path is None
    assert results.report_path == results.runs[0].report_path
    assert results.report_path.is_file()

    # Ensure no comparison report was saved in the output directory
    set_out = tmp_path / "out" / set_name
    comparison_files = list(set_out.glob("WildObs_CV_Model_Evaluation_Comparison_Report_*.html"))
    assert len(comparison_files) == 0


def test_discover_export_folders_errors(tmp_path):
    from src.pipeline import discover_export_folders

    # Non-existent set folder
    with pytest.raises(FileNotFoundError, match="Test set folder not found"):
        discover_export_folders(tmp_path / "non_existent")

    # Set folder exists but is empty
    empty_set = tmp_path / "empty_set"
    empty_set.mkdir()
    with pytest.raises(FileNotFoundError, match="No Camtrap DP export folders found"):
        discover_export_folders(empty_set)

    # Set folder has subdirectories without observations.csv
    invalid_set = tmp_path / "invalid_set"
    (invalid_set / "sub1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No Camtrap DP export folders .* found"):
        discover_export_folders(invalid_set)


def test_create_model_comparison_report_accepts_file_sequence(tmp_path):
    report_a = tmp_path / "WildObs_CV_Model_Evaluation_Report_Loc_Model_A_2026.html"
    report_b = tmp_path / "WildObs_CV_Model_Evaluation_Report_Loc_Model_B_2026.html"
    _write_evaluation_report(report_a, "Model A", [("Cat", "0.85")])
    _write_evaluation_report(report_b, "Model B", [("Cat", "0.92")])

    out_dir = tmp_path / "out"
    comparison = create_model_comparison_report([report_a, report_b], out_dir, run_id="test_run")
    assert comparison.name == "WildObs_CV_Model_Evaluation_Comparison_Report_test_run.html"
    assert comparison.is_file()
    content = comparison.read_text(encoding="utf-8")
    assert "Model A" in content and "Model B" in content
    assert "0.85" in content and "0.92" in content


def test_clear_output_folder(tmp_path):
    from src.pipeline import clear_output_folder

    target_dir = tmp_path / "test_clear"
    target_dir.mkdir()
    (target_dir / "report1.html").write_text("dummy", encoding="utf-8")
    (target_dir / "data.csv").write_text("dummy", encoding="utf-8")
    sub = target_dir / "sub_folder"
    sub.mkdir()
    (sub / "nested.txt").write_text("dummy", encoding="utf-8")

    removed = clear_output_folder(target_dir)
    assert removed == 3
    assert target_dir.is_dir()
    assert list(target_dir.iterdir()) == []


def test_batch_run_with_clear_existing_outputs(tmp_path):
    set_name = "Set_Clear_Test"
    set_dir = tmp_path / "in" / set_name
    make_camtrap(set_dir / "export_1", model_name="Model 1")

    out_set_dir = tmp_path / "out" / set_name
    out_set_dir.mkdir(parents=True)
    stale_report = out_set_dir / "WildObs_CV_Model_Evaluation_Report_Old_20260101.html"
    stale_report.write_text("old report", encoding="utf-8")
    stale_csv = out_set_dir / "Misclassified_Images_Old.csv"
    stale_csv.write_text("old csv", encoding="utf-8")

    # Run without clear_existing_outputs -> stale files should still be present
    config_keep = BenchmarkConfig(
        test_set_name=set_name,
        clear_existing_outputs=False,
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        log_path=tmp_path / "logs",
    )
    run_benchmark(config_keep)
    assert stale_report.is_file()
    assert stale_csv.is_file()

    # Run with clear_existing_outputs -> stale files should be cleared
    config_clear = BenchmarkConfig(
        test_set_name=set_name,
        clear_existing_outputs=True,
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        log_path=tmp_path / "logs",
    )
    result = run_benchmark(config_clear)
    assert not stale_report.exists()
    assert not stale_csv.exists()
    assert result.runs[0].report_path.is_file()


def test_export_errors_default_is_false(tmp_path):
    set_name = "Set_Export_Errors_Default"
    set_dir = tmp_path / "in" / set_name
    make_camtrap(set_dir / "export_1", model_name="Model 1")

    # Default export_errors should be False
    config = BenchmarkConfig(
        test_set_name=set_name,
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        log_path=tmp_path / "logs",
    )
    assert config.export_errors is False

    result = run_benchmark(config)
    assert result.runs[0].misclassified_path is None
    csv_files = list((tmp_path / "out" / set_name).glob("Misclassified_Images_*.csv"))
    assert len(csv_files) == 0

    # Explicitly enabling export_errors creates the CSV
    config_enabled = BenchmarkConfig(
        test_set_name=set_name,
        export_errors=True,
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        log_path=tmp_path / "logs",
    )
    result_enabled = run_benchmark(config_enabled)
    assert result_enabled.runs[0].misclassified_path is not None
    assert result_enabled.runs[0].misclassified_path.is_file()



