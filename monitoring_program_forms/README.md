# Monitoring program forms

Structured questionnaires describing the goals and priorities of a monitoring program. They are read by the `interpret-evaluation-reports` skill (see `.github/skills/interpret-evaluation-reports/SKILL.md`) to tailor model recommendations.

- `_BLANK_TEMPLATE.md` - the master blank form. Do not fill it in; keep it unchanged.
- `EXAMPLE_adaptive_management_reserve.md` - an illustrative completed form.

## Usage

1. Copy `_BLANK_TEMPLATE.md` to a new file in this folder, e.g. `my_reserve_2026.md`.
2. Answer each question under its `Answer:` label and set the evaluation reports folder path. Keep the `##` headings unchanged.
3. Ask the agent to "interpret the evaluation reports using `monitoring_program_forms\my_reserve_2026.md`".

Use scientific names where possible, as these are matched against species labels in the reports.
