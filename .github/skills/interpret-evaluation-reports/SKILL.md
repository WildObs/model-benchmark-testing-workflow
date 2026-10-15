---
name: interpret-evaluation-reports
description: Interpret a set of WildObs computer vision model evaluation reports against a completed monitoring program form and recommend which model/confidence thresholds/validation workflow best suit the program's priorities. Flags uncertainty, test-data gaps (missing priority species, small samples, taxonomic-level mismatches) and suggests improvements to the test. Use when asked to interpret, compare, or make recommendations from model evaluation reports for a monitoring program.
---

# Interpret evaluation reports for a monitoring program

## Inputs

1. A completed form from `monitoring_program_forms\` (copied from `_BLANK_TEMPLATE.md`). If the user has not named one, list the completed forms (exclude `_BLANK_TEMPLATE.md` and `EXAMPLE_*`) and ask which to use. If the form has unanswered required questions (1, 2, 3, 3A/3B, 4, 5; 2A is optional), ask the user rather than guessing.
2. The evaluation report folder given in the form. It contains HTML reports from this workflow (`WildObs_CV_Model_Evaluation_Report_*.html`, one per model, optionally a comparison report) and possibly `Misclassified_Images_*.csv`. Use `src\report_components.py` and `src\metrics.py` (if present) to understand how tables are built.

## Procedure

1. **Parse the form.** Extract: program type (research / adaptive management / both), primary and secondary goals, whether a subset or all species are needed, priority species (3A), quick-alert value (3B), dominant species (4), rare-but-critical species (5). Normalise names to scientific names and note common-name or taxonomy errors in the form (e.g. deer are Cervidae, not Corvidae) — state the assumption made and ask if material.
2. **Extract report data** per model: test details (species/labels evaluated, data source, confidence thresholds, collation process), the Results table (recall, precision, F1 per species per threshold), optimal-threshold table, prediction distribution and full confusion matrices. Parse HTML tables with pandas (`pandas.read_html`) or BeautifulSoup. Do not invent numbers: every figure quoted must come from the reports.
3. **Derive sample sizes.** Row sums of the confusion matrices give images per true label (and predictions per label by column sums). Record n for every species; these drive uncertainty.
4. **Classify species** into tiers from the form:
   - *Priority* (3A, and 3B subset): recall and precision both matter; weight recall highest when the target is rarely detected or a missed detection has high cost (invasive early-detection, threatened species).
   - *Dominant* (4): drives review workload and overall metrics; high precision/recall here reduces manual effort. Low precision on a rare priority species also swamps reviewers.
   - *Rare/critical* (5): recall is paramount; accept lower precision if reviewers can verify alerts.
   - *Other*: context only, unless the form says all species must be identified (Q3), in which case treat all as required and use 3B for tiered priority.
5. **Match labels** between the form, test data and model output taxonomy. Handle:
   - Test data at a coarse level (e.g. `Macropus`) while the model predicts species (e.g. *Macropus giganteus*): the confusion matrix will show true-label `Macropus` split across species columns. Treat predictions within the same genus/family as correct for program purposes when species-level ID is not needed, state this explicitly, and note that reported recall/precision for that label under-represents the model. Never silently re-score; give an adjusted estimate only when computable from the matrix and label it as such.
   - Model predicts coarser than the test (aggregated labels): note that species-level recall cannot be assessed.
   - Priority species absent from the test data or never predicted by the model: report as "not evaluated" / "not supported" rather than as zero performance.
6. **Assess each model against the program**, per tier, at its optimal threshold and across thresholds (recall–precision trade-off). Consider the program type: adaptive management favours timely, sufficient-accuracy alerts and manageable review load; research favours higher accuracy, calibrated uncertainty and documented validation.
7. **Identify uncertainty** and attach to each conclusion a confidence level (High/Medium/Low) with reasons. Flag at least:
   - Small sample sizes (suggest: n < 30 low reliability; n < 100 moderate; state the rule used) — recall of 1.0 on n = 5 is not evidence.
   - Species in the form missing from the test, or present only as a coarse label.
   - Test data not representative of the location (different region, season, camera type) or class imbalance far from field proportions; check `Data_Collation_Process` and data source.
   - Metrics at a single threshold only; thresholds chosen on the same data used to report performance (optimistic).
   - Differences between models smaller than sampling error (do not rank models on differences within noise).
   - Reports that disagree with each other or were run on different datasets (not comparable).
8. **Recommend:**
   - Preferred model(s) and confidence threshold(s) per use (e.g. a low threshold for priority-species alerts, higher for bulk classification), or a combination of models.
   - A validation workflow: what to manually review (all priority-species detections, low-confidence predictions, rare-critical alerts), and how much, given the precision/recall observed.
   - Whether the model can be used for the stated goals now, with conditions, or not yet.
9. **Recommend test improvements** to enable a more accurate recommendation: add missing priority/rare species (including look-alikes that could be confused, e.g. dogs/foxes/cats, deer species), increase n to a stated target for priority classes, label test data at the same taxonomic level as needed, include hard-case images (night, partial, small animals), use site-representative proportions, include blank/empty and "other" images to measure false alarms, and re-run with a threshold range covering the priority species' optimum. Where a species was recorded by the form's rare/critical list, suggest sourcing images from other locations or the model provider's data if local samples are too few.

## Output

Write a Markdown report to `Output_Reports\Recommendations_<form file name without extension>_<YYYYMMDD>.md` (create the folder if needed) and give the user a short summary in chat. Structure:

1. **Summary and headline recommendation** (3–6 bullets, with overall confidence).
2. **Program priorities as understood** (restate from the form, including assumptions/corrections).
3. **Model comparison against priorities** (table: model × priority tier with recall/precision/F1 at recommended threshold, n, and confidence).
4. **Species-by-species notes** (priority, dominant, rare/critical), including interpretive notes for taxonomic-level mismatches.
5. **Areas of uncertainty and limitations** (itemised, tied to the data).
6. **Recommended thresholds and validation workflow.**
7. **Recommended improvements to the evaluation test** (prioritised, with target sample sizes).
8. **Appendix:** reports read, how sample sizes were derived, and any adjustments made to the reported figures.

## Rules

- Quote only numbers present in or directly derivable from the reports; show the derivation of anything derived.
- Be explicit about what the data cannot tell the user. Prefer "not evaluated" over inference.
- Do not modify the evaluation reports or the form.
