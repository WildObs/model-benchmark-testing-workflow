"""Point-and-click form (ipywidgets) for running the benchmark workflow from a notebook."""

import html
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import PROJECT_ROOT, BenchmarkConfig
from .data_loader import available_observation_levels, get_species_display_list
from .pipeline import discover_export_folders, run_benchmark
from .templating import list_templates

NO_SETS = "(no test sets found)"

THRESHOLD_PRESETS = [
    (">0.5", "0.5"),
    ("0.5, 0.6, 0.7, 0.8, 0.9", "0.5, 0.6, 0.7, 0.8, 0.9"),
    ("0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9", "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9"),
    ("0.2, 0.4, 0.6, 0.8", "0.2, 0.4, 0.6, 0.8"),
]
DEFAULT_THRESHOLD_PRESET = "0.5, 0.6, 0.7, 0.8, 0.9"


@dataclass
class ExportSummary:
    folder: str
    model: str
    levels: List[str]
    images: int
    species: List[str]


def parse_thresholds(text: str) -> List[float]:
    """Parse '0.5, 0.6 0.7' into a list of floats; raises ValueError with a friendly message."""
    parts = [p for p in text.replace(",", " ").replace(";", " ").split() if p]
    if not parts:
        raise ValueError("Enter at least one confidence threshold, e.g. 0.5, 0.6, 0.7")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Thresholds must be numbers between 0 and 1 separated by commas, got: {text!r}") from None
    bad = [v for v in values if not 0.0 <= v <= 1.0]
    if bad:
        raise ValueError(f"Thresholds must be between 0 and 1, got: {bad}")
    return values


def list_test_sets(input_path: Path) -> List[str]:
    """Folders inside the input path that contain at least one Camtrap DP export."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        return []
    sets = []
    for folder in sorted(p for p in input_path.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        try:
            discover_export_folders(folder)
        except FileNotFoundError:
            continue
        sets.append(folder.name)
    return sets


def summarise_set(set_folder: Path) -> List[ExportSummary]:
    """Read just enough of each export to show the user what was found."""
    summaries = []
    for export in discover_export_folders(set_folder):
        obs = pd.read_csv(export / "observations.csv", dtype=str, usecols=lambda c: c in ("observationLevel", "classifiedBy"))
        models = obs["classifiedBy"].dropna().astype(str).str.strip() if "classifiedBy" in obs else pd.Series(dtype=str)
        models = models[models != ""]
        levels = available_observation_levels(obs)
        deployments = pd.read_csv(export / "deployments.csv", dtype=str)
        media_rows = sum(1 for _ in open(export / "media.csv", encoding="utf-8")) - 1
        summaries.append(
            ExportSummary(
                folder=export.name,
                model=models.iloc[0] if not models.empty else "Unknown Model",
                levels=sorted(levels),
                images=max(media_rows, 0),
                species=get_species_display_list(deployments),
            )
        )
    return summaries


def effective_observation_level(requested: str, summaries: List[ExportSummary]) -> str:
    """Mirror the pipeline rule: media is only used if every export has media-level observations."""
    if requested == "media" and all("media" in s.levels for s in summaries):
        return "media"
    return "event"


def describe_set(summaries: List[ExportSummary], requested_level: str) -> str:
    """HTML preview of the exports found in the selected test set."""
    rows = "".join(
        f"<tr><td>{html.escape(s.model)}</td><td>{s.images:,}</td><td>{', '.join(s.levels)}</td>"
        f"<td title='{html.escape(s.folder)}'>{html.escape(s.folder)}</td></tr>"
        for s in summaries
    )
    table = (
        "<table style='border-collapse:collapse' cellpadding='4'><tr><th align='left'>Model</th>"
        "<th align='left'>Images</th><th align='left'>Observation levels</th><th align='left'>Export folder</th></tr>"
        f"{rows}</table>"
    )
    used = effective_observation_level(requested_level, summaries)
    if requested_level == "media" and used == "event":
        missing = [s.model for s in summaries if "media" not in s.levels]
        warning = (
            "<p><span style='color:#b45309'>&#9888; Media-based observations are not available in every export "
            f"({html.escape(', '.join(missing))} lack them), so <b>event-based</b> observations will be used for all exports.</span></p>"
        )
        return table + warning
    return table


def make_config(
    test_set: str,
    source_location: str,
    collation_process: str,
    thresholds_text: str,
    observation_level: str,
    template: str,
    exclude_sections: List[str],
    export_errors: bool,
    lookup_taxonomy: bool,
    clear_existing: bool,
    log_level: str = "INFO",
    input_path: str = "Input_Data",
    output_path: str = "Output_Reports",
) -> BenchmarkConfig:
    """Turn form values into a BenchmarkConfig (raises ValueError with a friendly message)."""
    if not test_set or test_set == NO_SETS:
        raise ValueError("Choose a test set folder first.")
    return BenchmarkConfig(
        test_set_name=test_set,
        data_source_location=source_location.strip(),
        data_collation_process=collation_process.strip(),
        thresholds=parse_thresholds(thresholds_text),
        observation_level=observation_level,
        template=template,
        exclude_sections=list(exclude_sections),
        export_errors=export_errors,
        lookup_taxonomy=lookup_taxonomy,
        clear_existing_outputs=clear_existing,
        log_level=log_level,
        input_path=input_path,
        output_path=output_path,
    )


def build_form(project_root: Optional[Path] = None):
    """Build and display the benchmark form. Returns the container widget."""
    import ipywidgets as w
    from IPython.display import display

    from .notebook_display import display_report

    root = Path(project_root) if project_root else PROJECT_ROOT
    input_path, templates_path = root / "Input_Data", root / "templates"
    layout = w.Layout(width="650px")
    style = {"description_width": "190px"}

    def field(widget_cls, label, **kwargs):
        kwargs.setdefault("layout", layout)
        return widget_cls(description=label, style=style, **kwargs)

    sets = list_test_sets(input_path)
    test_set = field(w.Dropdown, "Test set folder", options=sets or [NO_SETS],
                     tooltip="Folder inside Input_Data containing one or more Camtrap DP exports")
    refresh = w.Button(description="Refresh", icon="refresh", layout=w.Layout(width="100px"))
    preview = w.HTML()
    source = field(w.Text, "Data source location", placeholder="e.g. Tarcutta Reserve, NSW")
    collation = field(w.Textarea, "Data collation process", placeholder="How the test images were selected",
                      layout=w.Layout(width="650px", height="80px"))
    thresholds = field(w.Dropdown, "Confidence thresholds", options=THRESHOLD_PRESETS, value=DEFAULT_THRESHOLD_PRESET)
    level = field(w.RadioButtons, "Observations to use", options=[("Event-based (recommended)", "event"), ("Media-based", "media")],
                  value="event")
    level_help = w.HTML("<i>Media-based is only used if every export contains it; otherwise event-based is used for all.</i>")

    templates = {name: desc for name, desc in list_templates(templates_path).items() if name != "model_comparison"}
    template = field(w.Dropdown, "Report template", options=list(templates), value="full" if "full" in templates else list(templates)[0])
    template_help = w.HTML()
    export_errors = field(w.Checkbox, "Export misclassified images csv", value=False, indent=False)
    taxonomy = field(w.Checkbox, "Look up taxonomy (needs internet and galah)", value=True, indent=False)
    clear = field(w.Checkbox, "Clear existing outputs for this set first", value=True, indent=False)
    log_level = field(w.Dropdown, "Log level", options=["INFO", "DEBUG", "WARNING", "ERROR"], value="INFO")
    advanced = w.Accordion(children=[w.VBox([export_errors, taxonomy, clear, log_level])], selected_index=None)
    advanced.set_title(0, "Advanced")

    run = w.Button(description="Run evaluation", button_style="success", icon="play", layout=w.Layout(width="200px"))
    status = w.HTML()
    output = w.Output()

    def update_preview(*_):
        if test_set.value == NO_SETS:
            preview.value = f"<span style='color:#b91c1c'>No test sets found in <code>{html.escape(str(input_path))}</code>. " \
                            "Add a folder of Camtrap DP exports there, then click Refresh.</span>"
            return
        try:
            preview.value = describe_set(summarise_set(input_path / test_set.value), level.value)
        except Exception as exc:
            preview.value = f"<span style='color:#b91c1c'>Could not read this test set: {html.escape(str(exc))}</span>"

    def update_template(*_):
        template_help.value = f"<i>{html.escape(templates.get(template.value, ''))}</i>"

    def on_refresh(_):
        current = test_set.value
        options = list_test_sets(input_path)
        test_set.options = options or [NO_SETS]
        if current in options:
            test_set.value = current
        update_preview()

    def on_run(_):
        output.clear_output()
        status.value = ""
        try:
            config = make_config(
                test_set.value, source.value, collation.value, thresholds.value, level.value, template.value,
                [], export_errors.value, taxonomy.value, clear.value, log_level.value,
                input_path=str(input_path), output_path=str(root / "Output_Reports"),
            )
        except ValueError as exc:
            status.value = f"<span style='color:#b91c1c'>&#10060; {html.escape(str(exc))}</span>"
            return
        run.disabled = True
        status.value = "<i>Running... this can take a few minutes for large test sets.</i>"
        with output:
            try:
                results = run_benchmark(config)
            except Exception as exc:
                status.value = f"<span style='color:#b91c1c'>&#10060; The run failed: {html.escape(str(exc))}. " \
                               "See the log file in the logs folder for details.</span>"
            else:
                status.value = (
                    f"<span style='color:#15803d'>&#9989; Done. {len(results.runs)} evaluation report(s) saved to "
                    f"<code>{html.escape(str(config.set_output_folder))}</code></span>"
                )
                for r in results.runs:
                    print(f"{r.model_name}: {r.report_path}")
                if results.comparison_report_path:
                    print(f"Comparison report: {results.comparison_report_path}")
                display_report(results.report_path)
            finally:
                run.disabled = False

    test_set.observe(update_preview, "value")
    level.observe(update_preview, "value")
    template.observe(update_template, "value")
    refresh.on_click(on_refresh)
    run.on_click(on_run)
    update_template()
    update_preview()

    form = w.VBox([
        w.HTML("<h3>1. Select test set input folder</h3>"), w.HBox([test_set, refresh]), preview,
        w.HTML("<h3>2. Define the report details</h3>"), source, collation,
        w.HTML("<h3>3. Configure analysis settings</h3>"), thresholds, level, level_help,
        w.HTML("<h3>4. Select the report format</h3>"), template, template_help,
        advanced,
        w.HTML("<br>"), run, status, output,
    ])
    display(form)
    return form
