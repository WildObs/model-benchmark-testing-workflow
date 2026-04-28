import pandas as pd
import os
import re

# -------- USER INPUT --------

#camtrap_folder = r"model-benchmarking-wildobs-national-20260401220208"
#camtrap_folder = r"model-benchmarking-awc-135-20260401220412"
# camtrap_folder = r"model-benchmarking-speciesnet-v4-20260401215931"
camtrap_folder = r"model-benchmarking-wildobs-national-20260317045910"

output_report_folder = r"model_Benchmarking_Report_Exports"
confidence_threshold = None  # e.g. 0.8 or None to disable
export_errors = True
data_source_location = "Bon Bon Reserve, SA"
data_collation_process = """
Collated 1000 images of each target species.
Images were manually validated as having at least one detection of the target species and no other species present in the image.
"""

# -----------------------------
# Load data
# -----------------------------
observations = pd.read_csv(os.path.join(camtrap_folder, "observations.csv"))
media = pd.read_csv(os.path.join(camtrap_folder, "media.csv"))
deployments = pd.read_csv(os.path.join(camtrap_folder, "deployments.csv"))

# -----------------------------
# Extract sequenceID
# -----------------------------
def extract_sequence(comment):
    if pd.isna(comment):
        return None
    match = re.search(r"sequenceID:([^\s]+)", str(comment))
    return match.group(1) if match else None


media["sequenceID"] = media["mediaComments"].apply(extract_sequence)

# -----------------------------
# Keep required columns
# -----------------------------
observations = observations[[
    "eventID",
    "deploymentID",
    "scientificName",
    "classificationProbability",
    "classifiedBy"
]].rename(columns={
    "classificationProbability": "confidence"
})

media = media[["mediaID", "mediaComments", "sequenceID"]]

deployments = deployments[["deploymentID", "deploymentTags"]]
# Extract model name
model_name_series = observations["classifiedBy"].dropna().astype(str).str.strip()
model_name_series = model_name_series[model_name_series != ""]
model_name = model_name_series.iloc[0] if not model_name_series.empty else "Unknown Model"

# Extract species list
species_list = (
    deployments["deploymentTags"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
)

# Convert to sorted list
species_list = sorted(species_list)

def format_species_name(name):
    parts = str(name).split()
    if not parts:
        return name
    return " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])

species_list_display = [format_species_name(s) for s in species_list]

# -----------------------------
# Merge tables
# -----------------------------
print("Columns in media collection:", media.columns)
print("Columns in observations collection:", observations.columns)
merged = pd.merge(media, observations, left_on="sequenceID", right_on="eventID", how="left")
print("Columns in merged collection (using media and observations):", merged.columns)
merged = pd.merge(merged, deployments, on="deploymentID", how="left")
print("Columns in final merge collection:", merged.columns)

# -----------------------------
# Normalize species names
# -----------------------------
merged["true_species"] = (
    merged["deploymentTags"]
    .astype(str)
    .str.strip()
    .str.lower()
)

merged["pred_species"] = (
    merged["scientificName"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# -----------------------------
# Apply confidence threshold (optional)
# -----------------------------
if confidence_threshold is not None:
    merged = merged[merged["confidence"] >= confidence_threshold]

# -----------------------------
# Remove rows with missing truth
# -----------------------------
merged = merged[merged["true_species"].notna()]

# -----------------------------
# Evaluation metrics
# -----------------------------

results = []

# Only include species that exist in ground truth
species_list = sorted(merged["true_species"].dropna().unique())

for species in species_list:

    tp = ((merged["true_species"] == species) & (merged["pred_species"] == species)).sum()
    fn = ((merged["true_species"] == species) & (merged["pred_species"] != species)).sum()
    fp = ((merged["true_species"] != species) & (merged["pred_species"] == species)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    results.append({
        "species": species,
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4)
    })

results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)

def format_species_name(name):
    if pd.isna(name):
        return name
    
    parts = str(name).strip().split()
    
    if not parts:
        return name
    
    return " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])

results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
results_df["species_display"] = results_df["species"].apply(format_species_name)

# Reorder columns (optional)
results_df = results_df[
    ["species_display", "true_positives", "false_negatives", "false_positives", "recall", "precision", "f1_score"]
].rename(columns={"species_display": "species"})


# -----------------------------
# Confusion matrix
# -----------------------------
conf_matrix = pd.crosstab(
    merged["true_species"],
    merged["pred_species"]
)
print(conf_matrix)

single_row_matrices = {}

species_list = sorted(merged["true_species"].dropna().unique())

for species in species_list:

    # Get only rows where this is the true species
    subset = merged[merged["true_species"] == species]

    # Count predictions
    pred_counts = subset["pred_species"].value_counts()

    # Convert to DataFrame (single row)
    sub_matrix = pd.DataFrame([pred_counts])
    sub_matrix.index = [species]

    # Ensure all predicted species columns are present (optional)
    sub_matrix = sub_matrix.fillna(0)

    # Sort columns by frequency (descending)
    sub_matrix = sub_matrix.loc[:, sub_matrix.sum().sort_values(ascending=False).index]

    single_row_matrices[species] = sub_matrix

formatted_single_row = {}

for species, matrix in single_row_matrices.items():

    # Set row label
    matrix.index = ["count"]

    # Format species names
    matrix.columns = [format_species_name(col) for col in matrix.columns]

    # Reset index so "count" becomes a column
    matrix_reset = matrix.reset_index()

    # Rename columns
    matrix_reset.rename(columns={"index": "pred_species"}, inplace=True)

    formatted_single_row[format_species_name(species)] = matrix_reset


conf_matrix.index = conf_matrix.index.map(format_species_name)
conf_matrix.columns = conf_matrix.columns.map(format_species_name)


# -----------------------------
# Misclassified images export
# -----------------------------
if export_errors:
    errors = merged[merged["true_species"] != merged["pred_species"]]
    errors.to_csv("misclassified_images.csv", index=False)

"""
# -----------------------------
# Output (Jupyter-friendly)
# -----------------------------
from IPython.display import display

print("\n=== MODEL EVALUATION RESULTS ===")
display(results_df)

print("\n=== CONFUSION MATRIX ===")
display(conf_matrix)
"""
print("\nTotal observations analysed:", len(merged))

# Generating a nicely formatted report that can be shared with others as a .html file

html_file = output_report_folder + "/" + f"Model_Benchmark_Test_Report_{model_name}.html"

style = """
<style>
body { font-family: Arial; margin: 40px; }
h1 { color: #2c3e50; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #2c3e50; margin-top: 20px; }
p { max-width: 900px; line-height: 1.5; }
table { border-collapse: collapse; width: 80%; margin-top: 10px; }
th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
th { background-color: #f2f2f2; }
</style>
"""

# MathJax script
mathjax = """
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

with open(html_file, "w", encoding="utf-8") as f:

    f.write("<html><head>")
    f.write(style)
    f.write(mathjax)
    f.write("</head><body>")

    f.write("<h1>Model Evaluation Report</h1>")
    
    f.write(f"<h2>Test details</h2>")
    f.write(f"<p><b>Computer Vision (CV) model tested: </b>{model_name}<p>")
    f.write(f"<p><b>Source location of test images: </b>{data_source_location}</p>")
    #f.write(f"<p><b>Target species tested: </b>{species_list_display}</p>")
    f.write("<p><b>Species (or labels) evaluated:</b></p>")
    f.write("<ul>")
    for species in species_list_display:
        f.write(f"<li><i>{species}</i></li>")
    f.write("</ul>")
    f.write(f"<p><b>Data collation process used: </b>{data_collation_process}</p>")

    # -----------------------------
    # Explanation Section
    # -----------------------------

    f.write("<h2>Purpose and intended use of this report</h2>")
    f.write("""
        <ul>
            <li>Inform the potential suitability of a model for a given location and set of species relevant to the monitoring program.</li>
            <li>Inform the design of a validation workflow (manual human review) to ensure a sufficient level of accuracy can be reached to meet the goals of the monitoring program.</li>
            <li>Inform potential gaps in model performance and priorities for provision of additional training images to improve performance.</li>
        </ul>
    """)
    f.write("<h2>Disclaimer / limitations</h2>")
    f.write("""
        <ul>
            <li>The accuracy of the results presented in this report is primairly limited by the quality of the test data used.</li>
            <li>Please follow the recommended guidelines for collating a representative test dataset.</li>
            <li>Species name mismatches may be a cause of low values for recall or precision. 
                <ul>
                    <li>Models sometimes aggregate multiple species into one label.</li>
                    <li>Conversely, the input test data used may also contain aggregated labels that are not matched by a model that is splitting by individual species</li>
                    <li>If there is a species name mismatch, inspection of the confusion matrix will often reveal what name the model is using for that species.</li>
                </ul>
            <li>This version of the report is not considering model confidence due to a software bug that is being worked on. This will be fixed in a future version.</li>
            <li>The results presented in this report do not necessarily prove definitively that one model is better than another, but may provide guidence on selecting the most suitable model for your location and the species important to your monitoring program.</li>
        </ul>
    """)
    f.write("<p></p>")
    f.write("<h2>Understanding the Results</h2>")
    f.write("<p>Recall, precision, and F1 score are standard metrics used to evaluate the performance of machine learning models. Each metric has a precise definition and corresponding mathematical formula.</p>")

    f.write("<h3>Recall (Sensitivity)</h3>")
    f.write(r"<p>\[ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}} \]</p>")
    f.write("<p>Of all the images that actually contain the target species, how many did the model correctly identify?</p>")
    f.write("<p><b>Example: </b>If there are 100 images of Feral Cat within a dataset, and the model correctly detects 90 of them, recall = 0.9 (90%).</p>")
    f.write("<p><b>Interpretation:</b> Quantifying recall helps us to understand how many true detections of a given species might be missed by the model.</p>")

    f.write("<h3>Precision</h3>")
    f.write(r"<p>\[ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}} \]</p>")
    f.write("<p>Of all the images the model predicts as the target species, how many are actually correct?</p>")
    f.write("<p><b>Example: </b>If the model makes 100 predictions of Feral Cat within a dataset, and only 90 of them are actually detections of Feral Cat, precision = 0.9 (90%).</p>")
    f.write("<p><b>Interpretation:</b> High precision means predictions are reliable. Low precision means more false detections.</p>")

    f.write("<h3>F1 Score</h3>")
    f.write(r"<p>\[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}} \]</p>")
    f.write("<p>A balanced measure that combines both recall and precision.</p>")
    f.write("<p><b>Interpretation:</b> High F1 score indicates a good balance between detecting animals and avoiding false detections.</p>")
    f.write("<h3>Definition of Terms</h3>")

    f.write("""
    <table>
        <tr>
            <th>Term</th>
            <th>Definition</th>
            <th>Shorthand definition</th>
        </tr>
        <tr>
            <td><b>True Positive (TP)</b></td>
            <td>The model correctly predicts the presence of a species.</td>
            <td>Correct detection</td>
        </tr>
        <tr>
            <td><b>False Positive (FP)</b></td>
            <td>The model predicts a species, but that species is not actually present.</td>
            <td>False alarm</td>
        </tr>
        <tr>
            <td><b>False Negative (FN)</b></td>
            <td>The model fails to detect a species that is actually present.</td>
            <td>Missed detection</td>
        </tr>
    </table>
    """)
    
    f.write("<h3>How to interpret results</h3>")
    f.write("""
    <table>
        <tr><th>Scenario</th><th>Interpretation</th></tr>
        <tr><td>High Recall, Low Precision</td><td>Finds most animals but includes many false positives</td></tr>
        <tr><td>Low Recall, High Precision</td><td>Accurate predictions but misses many animals</td></tr>
        <tr><td>High Recall, High Precision</td><td>Ideal performance</td></tr>
        <tr><td>Low Recall, Low Precision</td><td>Poor performance</td></tr>
    </table>
    """)
    f.write("""
    <h3>What is a Confusion Matrix?</h3>
    <ul>
        <li>A confusion matrix summarises the model’s performance by comparing actual vs predicted labels.</li>
        <li>In simple terms the confusion matrix shows what the model got right and wrong, broken down by type of error.</li>
        <li>Inspecting the confusion matrix helps us to understand the direction of error in cases where an image was incorrectly classified.</li>
        <li>Be aware that not all models define or name species the same way. In some cases a model may also generalise a species detections to a broader group e.g. Genus or Family. If recall is returning a very low number for a given species and model, inspection of confusion matrix may help to reveal inconsistencies in naming or categorisation.</li>
    </ul>
    """)

    # -----------------------------
    # Results Tables
    # -----------------------------
    f.write("<h2>Recall, Precision, and F1 Results</h2>")
    f.write(results_df.to_html(index=False))

    f.write("<h2>Prediction Distribution by Species</h2>")
    f.write("<p>The outputs below represent simplified confusion matricies highliting the distribution of predictions relevant to each target (test) species. Refer to the full confusion matrix for every combination of True Positive, False Positive, and False Negative.</p>")

    for species, matrix in formatted_single_row.items():
        f.write(f"<h3>True species (or label): <i>{species}</i></h3>")
        f.write(matrix.to_html(index=False))

    f.write("<h2>Full Confusion Matrix</h2>")
    f.write(conf_matrix.to_html())

    f.write("</body></html>")

print(f"Report saved to: {html_file}")

