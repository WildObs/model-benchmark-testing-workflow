"""Point-and-click form (ipywidgets) for running the benchmark workflow from a notebook."""

import html
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import markdown
import pandas as pd

from .config import PROJECT_ROOT, BenchmarkConfig
from .data_loader import available_observation_levels, get_species_display_list
from .pipeline import discover_export_folders, run_benchmark
from .templating import list_templates

def load_help(path: Path) -> dict:
    """Read '## key' sections from a Markdown file into {key: html}. Missing file -> no help."""
    path = Path(path)
    if not path.is_file():
        return {}
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    sections = {}
    for chunk in re.split(r"^##[ \t]+", text, flags=re.MULTILINE)[1:]:
        key, _, body = chunk.partition("\n")
        if body.strip():
            sections[key.strip()] = markdown.markdown(body.strip(), extensions=["fenced_code", "sane_lists"])
    return sections


def heading_with_help(title: str, help_html: Optional[str]) -> str:
    """Section heading plus a collapsible help block when help text exists."""
    heading = f"<h3>{title}</h3>"
    if not help_html:
        return heading
    return (
        f"{heading}<details style='margin:0 0 8px 0'><summary style='cursor:pointer;color:#2563eb'>"
        f"&#9432; Help</summary><div style='max-width:650px;padding:6px 12px;background:#f3f4f6;"
        f"border-left:3px solid #2563eb;color:#111827'>{help_html}</div></details>"
    )


def set_options(sets: List[str]) -> list:
    """Dropdown options with a leading blank entry so nothing is pre-selected."""
    return [("", None)] + [(f"Input_Data/{s}", s) for s in sets]

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
    species_subset: Optional[List[str]] = None,
    allow_higher_taxon_matching: bool = False,
    export_subset: Optional[List[str]] = None,
) -> BenchmarkConfig:
    """Turn form values into a BenchmarkConfig (raises ValueError with a friendly message).

    species_subset is None/empty for all species, otherwise the species to report on.
    """
    if not test_set:
        raise ValueError("Choose a test set folder first.")
    return BenchmarkConfig(
        test_set_name=test_set,
        data_source_location=source_location.strip(),
        data_collation_process=collation_process.strip(),
        thresholds=parse_thresholds(thresholds_text),
        observation_level=observation_level,
        template=template,
        exclude_sections=list(exclude_sections),
        export_subset=list(export_subset or []),
        species_subset=list(species_subset or []),
        allow_higher_taxon_matching=allow_higher_taxon_matching,
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
    style = {"description_width": "260px"}

    def field(widget_cls, label, **kwargs):
        kwargs.setdefault("layout", layout)
        return widget_cls(description=label, style=style, **kwargs)

    sets = list_test_sets(input_path)
    test_set = field(w.Dropdown, "Test set folder", options=set_options(sets), value=None,
                     tooltip="Folder inside Input_Data containing one or more Camtrap DP exports")
    refresh = w.Button(description="Refresh", icon="refresh", layout=w.Layout(width="100px"))
    preview = w.HTML()
    model_mode = field(w.RadioButtons, "Model subset",
                       options=[("Include all available input files", "all"),
                                ("Select a subset of the available input files", "subset")], value="all")
    model_select = field(w.SelectMultiple, "Input files to include", options=[], rows=6,
                         layout=w.Layout(width="650px", display="none"))
    source = field(w.Text, "Data source location", placeholder="e.g. Tarcutta Reserve, NSW")
    collation = field(w.Textarea, "Data collation process", placeholder="How the test images were selected",
                      layout=w.Layout(width="650px", height="80px"))
    thresholds = field(w.Dropdown, "Confidence thresholds to include", options=THRESHOLD_PRESETS, value=DEFAULT_THRESHOLD_PRESET)
    taxon_matching = field(w.RadioButtons, "Taxonomy matching preferences",
                           options=[("Restrict to exact matches of taxonomy name", "exact"),
                                    ("Allow higher ranking input taxonomies to match to lower ranking taxonomies predicted by model", "higher")],
                           value="exact")
    taxon_help = w.HTML("<i>Higher-ranking matching uses ALA galah (needs internet). It affects recall, precision and F1 only; "
                        "confusion matrices always show the exact predictions.</i>")
    level = field(w.RadioButtons, "Observation Levels to include in the analysis", options=[("Event-based (recommended)", "event"), ("Media-based", "media")],
                  value="event")
    level_help = w.HTML("<i>Media-based is only used if every export contains it; otherwise event-based is used for all.</i>")

    templates = {name: desc for name, desc in list_templates(templates_path).items() if name != "model_comparison"}
    template = field(w.Dropdown, "Report template", options=list(templates), value="full" if "full" in templates else list(templates)[0])
    template_help = w.HTML()
    species_mode = field(w.RadioButtons, "Species subset",
                         options=[("Include results for all species tested", "all"),
                                  ("Include results for a subset of the species tested", "subset")], value="all")
    species_select = field(w.SelectMultiple, "Species to include", options=[], rows=8,
                           layout=w.Layout(width="650px", display="none"))
    species_help = w.HTML("<i>Hold Ctrl (or Cmd) to select more than one species.</i>",
                          layout=w.Layout(display="none"))
    export_errors = field(w.Checkbox, "Export misclassified images csv", value=False, indent=False)
    taxonomy = field(w.Checkbox, "Look up taxonomy (needs internet and galah)", value=True, indent=False)
    clear = field(w.Checkbox, "Clear existing outputs for this set first", value=True, indent=False)
    log_level = field(w.Dropdown, "Log level", options=["INFO", "DEBUG", "WARNING", "ERROR"], value="INFO")
    advanced = w.Accordion(children=[w.VBox([export_errors, taxonomy, clear, log_level])], selected_index=None)
    advanced.set_title(0, "Advanced")

    run = w.Button(description="Run evaluation", button_style="success", icon="play", layout=w.Layout(width="200px"))
    status = w.HTML()
    open_folder = w.Button(description="Open reports folder", icon="folder-open", layout=w.Layout(width="200px", display="none"))
    open_folder.folder = None
    output = w.Output()
    log_box = w.Accordion(children=[output], selected_index=None, layout=w.Layout(display="none"))
    log_box.set_title(0, "Log messages")
    reports_box = w.VBox()

    def build_report_accordion(results):
        """One collapsed section per report; each report is rendered the first time it is expanded."""
        items = []
        if results.comparison_report_path:
            items.append(("Model comparison report", results.comparison_report_path))
        items += [(f"{r.model_name}", r.report_path) for r in results.runs]
        outputs = [w.Output() for _ in items]
        accordion = w.Accordion(children=outputs, selected_index=None)
        for i, (title, _) in enumerate(items):
            accordion.set_title(i, title)
        rendered = set()

        def on_select(change):
            i = change["new"]
            if i is None or i in rendered:
                return
            rendered.add(i)
            with outputs[i]:
                display_report(items[i][1])

        accordion.observe(on_select, "selected_index")
        return w.VBox([w.HTML("<h3>Reports (click to expand)</h3>"), accordion])

    def update_species_options():
        """Species list is built from the selected test set's data (union across exports)."""
        previous = set(species_select.value)
        names = []
        if test_set.value is not None:
            try:
                names = sorted({sp for s in summarise_set(input_path / test_set.value) for sp in s.species})
            except Exception:
                names = []
        species_select.options = names
        species_select.value = tuple(n for n in names if n in previous)

    def update_model_options():
        """Input files (export folders) are listed from the selected test set, labelled by model."""
        previous = set(model_select.value)
        options = []
        if test_set.value is not None:
            try:
                options = [(f"{s.model} ({s.folder})", s.folder) for s in summarise_set(input_path / test_set.value)]
            except Exception:
                options = []
        model_select.options = options
        model_select.value = tuple(v for _, v in options if v in previous)

    def update_model_visibility(*_):
        model_select.layout.display = "block" if model_mode.value == "subset" else "none"

    def update_species_visibility(*_):
        display = "block" if species_mode.value == "subset" else "none"
        species_select.layout.display = display
        species_help.layout.display = display

    def update_preview(*_):
        if not list_test_sets(input_path):
            preview.value = f"<span style='color:#b91c1c'>No test sets found in <code>{html.escape(str(input_path))}</code>. " \
                            "Add a folder of Camtrap DP exports there, then click Refresh.</span>"
            return
        if test_set.value is None:
            preview.value = ""
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
        test_set.options = set_options(options)
        test_set.value = current if current in options else None
        update_preview()

    def on_open_folder(button):
        try:
            if hasattr(os, "startfile"):
                os.startfile(button.folder)
            else:
                subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(button.folder)], check=False)
        except Exception as exc:
            status.value += f"<br><span style='color:#b91c1c'>Could not open the folder: {html.escape(str(exc))}</span>"

    def on_run(_):
        output.clear_output()
        reports_box.children = []
        log_box.layout.display = "none"
        status.value = ""
        open_folder.layout.display = "none"
        try:
            files = list(model_select.value) if model_mode.value == "subset" else []
            if model_mode.value == "subset" and not files:
                raise ValueError("Select at least one input file, or choose to include all available input files.")
            subset = list(species_select.value) if species_mode.value == "subset" else []
            if species_mode.value == "subset" and not subset:
                raise ValueError("Select at least one species, or choose to include all species tested.")
            config = make_config(
                test_set.value, source.value, collation.value, thresholds.value, level.value, template.value,
                [], export_errors.value, taxonomy.value, clear.value, log_level.value,
                input_path=str(input_path), output_path=str(root / "Output_Reports"),
                species_subset=subset,
                export_subset=files,
                allow_higher_taxon_matching=taxon_matching.value == "higher",
            )
        except ValueError as exc:
            status.value = f"<span style='color:#b91c1c'>&#10060; {html.escape(str(exc))}</span>"
            return
        run.disabled = True
        status.value = "<i>Running... this can take a few minutes for large test sets.</i>"
        log_box.layout.display = "block"
        log_box.selected_index = 0
        with output:
            try:
                results = run_benchmark(config)
            except Exception as exc:
                status.value = f"<span style='color:#b91c1c'>&#10060; The run failed: {html.escape(str(exc))}. " \
                               "See the log file in the logs folder for details.</span>"
            else:
                folder = Path(config.set_output_folder).resolve()
                status.value = (
                    f"<span style='color:#15803d'>&#9989; Done. {len(results.runs)} evaluation report(s) saved to "
                    f"<a href='{folder.as_uri()}' target='_blank'>{html.escape(str(folder))}</a></span>"
                )
                open_folder.folder = folder
                open_folder.layout.display = "inline-flex"
                for r in results.runs:
                    print(f"{r.model_name}: {r.report_path}")
                if results.comparison_report_path:
                    print(f"Comparison report: {results.comparison_report_path}")
                reports_box.children = [build_report_accordion(results)]
            finally:
                run.disabled = False
                log_box.selected_index = None

    test_set.observe(update_preview, "value")
    test_set.observe(lambda _: update_species_options(), "value")
    test_set.observe(lambda _: update_model_options(), "value")
    model_mode.observe(update_model_visibility, "value")
    species_mode.observe(update_species_visibility, "value")
    level.observe(update_preview, "value")
    template.observe(update_template, "value")
    refresh.on_click(on_refresh)
    run.on_click(on_run)
    open_folder.on_click(on_open_folder)
    update_template()
    update_preview()

    help_text = load_help(templates_path / "tooltips" / "form_help.md")

    def heading(key, title):
        return w.HTML(heading_with_help(title, help_text.get(key)))

    form = w.VBox([
        heading("section_1", "1. Select the Test Set input folder (containing WildObs camtrapdp data exports)"), w.HBox([test_set, refresh]), preview, model_mode, model_select,
        heading("section_2", "2. Define the descriptive details for the report"), source, collation,
        heading("section_3", "3. Configure the analysis settings"), thresholds, taxon_matching, taxon_help, level, level_help,
        heading("section_4", "4. Select the report format"), template, template_help,
                species_mode, species_select, species_help,
                advanced,
        w.HTML("<br>"), run, status, open_folder, reports_box, log_box,
    ])
    display(form)
    return form
