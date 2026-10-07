# Report templates

Reports are built from Markdown templates in this folder. Pick one with `template="full"` in the notebook (or `--template full` on the command line).

| Template | Purpose |
|:--|:--|
| `full` | Everything: test details, purpose, limitations, metric explanations, and all result tables |
| `minimal` | Test details and result tables only, with no interpretive text |

Run the notebook's template cell or `python scripts/run_benchmark.py --list-templates` to see the templates and their sections.

## Making your own template

1. Copy `full.md` or `minimal.md` to a new name, e.g. `templates/my_report.md`.
2. Edit it. Use `template="my_report"` to select it, or pass a full path to a `.md` file stored anywhere.

Templates are standard Markdown (including tables). Raw HTML is also allowed.

### Placeholders

Write `{Placeholder_Name}` where generated content should appear.

| Placeholder | Content |
|:--|:--|
| `{Report_Timestamp}` | Date and time of the run |
| `{Model_Name}` | Model tested (`classifiedBy` in the export) |
| `{Data_Source_Location}` | Source location of test images |
| `{Data_Collation_Process}` | Description of how the test data was collated |
| `{Confidence_Thresholds}` | Thresholds evaluated |
| `{Species_List}` | Bulleted list of species/labels evaluated |
| `{Species_Count}` | Number of species/labels evaluated |
| `{Image_Count}` | Number of images evaluated |
| `{Results_Table}` | Recall, precision and F1 by species and threshold |
| `{Best_Threshold_Table}` | Threshold with the best F1 for each species |
| `{Prediction_Distribution_Tables}` | Per-species prediction distribution (with headings) |
| `{Full_Confusion_Matrices}` | Confusion matrix for each threshold (with headings) |

Unknown placeholders containing an underscore are left as-is and a warning is written to the log.

### Optional sections

Wrap content in comments to make it switchable:

```markdown
<!-- BEGIN:Limitations -->
## Limitations
...
<!-- END:Limitations -->
```

Users can then drop it without editing the template, e.g. `exclude_sections=["Limitations"]`. Sections cannot be nested.

### Other features

- `<!-- description: One line summary -->` anywhere in the file is shown when listing templates.
- Formulas written as `\[ ... \]` are rendered with MathJax.
- Styling comes from `report.css`. To style one template differently, add a CSS file with the same name (e.g. `my_report.css`).
- Adding a new placeholder means adding an entry in `src/report_components.py` and listing it above.
