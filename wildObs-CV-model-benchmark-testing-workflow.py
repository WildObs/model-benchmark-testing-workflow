# %% [markdown]
# # Model evaluation workflow designed for WildObs Image Management Platform
# 
# ## Description
# - This script can be used for benchmarking an ai species recognition model with a local dataset to get an independent assessment of Recall, Precision and F1 Score for a given location
# - The purpose is to evaluate the most suitable or best performing model for a given location and inform an appropriate validation workflow to ensure accuracy requirements are met
# 
# ## Setup instructions:
# 1) Prepare the testing dataset. Start by organising camera trap images into folders by species on your local computer. The quality of the testing dataset will determine the accuracy of the report generated. Here are a few tips:
# - Use a representive number of images of each species e.g. at least 1000 if possible
# - Avoid using images that have been used as part of the model training dataset as these will create a biased result
# - If possible, select a random subset of testing images from a larger pool aiming to get a wide range of images over space and time
# - For the purpose of calculating Recall, include species that are relevant to your monitoring program
# - For the the purpose of calculating Precision, also include images of other species that are commonly detected on the cameras at that location even if they are not relevant to your monitoring. Also include some blank images. Below is an example breakdown:
# 
# | Species         | # images | Reason for inclusion                     |
# |:----------------|:---------|:-----------------------------------------|
# | Feral Cat       | 1000     | Target species in monitoring program     |
# | Red Fox         | 1000     | Target species in monitoring program     |
# | European Rabbit | 1000     | Target species in monitoring program     |
# | Kangaroo        | 1000     | Non-target species but abundant at this location. Impact on Precision. |
# | Emu             | 1000     | Non-target species but abundant at this location. Impact on Precision. |
# | Blank           | 1000     | Impact on Precision. |
# 
# 2) Establish new Project/s in the WildObs WIMP for benchmarking purposes:
# -  You will need a separate Project for each model you are testing. 
# -  Name the project based on the model that will be tested e.g. "Model benchmark testing: WildObs National".
# -  Set the Sequence cutoff to 0 seconds. This aims to prevent the software from creating sequences so that each image is assessed independently.
# -  Define Tags in the project based on the scientific names of the species you are testing. Tags need to match with the species names used in the WIMP.
# -  Configure the project to use the model you want to test
# 3) Create Deployments:
# - You will need to create a Deployment for each of the species you are testing.
# - Upload the relevant images into each deployment.
# - Use the tags created earlier to assign to the deployment so you know which species it is supposed to be. This will be used by the script to match the species to the model predictions
# - Repeat for each Project, uploading the same set of images to each
# 4) Run the uploaded images through the AI species recognition model
# 5) Once model processing is complete for all deployments, export the project data in Camtrap DP format
# 6) Download and extract (unzip) the exported data to a folder on your local computer
# 7) Use the folder path as input to this script

# %%
import pandas as pd
import os
import re
from IPython.display import display, HTML
from pathlib import Path
from datetime import datetime

# -------- USER INPUT --------

output_path = "./Output_Reports"
input_path = "./Input_Data"

#input_camtrap_folder_name = "speciesnet-v4-20260428054705"
#input_camtrap_folder_name = "awc-135-20260428055153"
input_camtrap_folder_name = "wildobs-national-20260428055250"


data_source_location = "Bon Bon Reserve, SA"
data_collation_process = """
Collated 1000 images of each target species.
Images were manually validated as having at least one detection of the target species and no other species present in the image.
"""
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
#thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]
#thresholds = [0.0., 0.5]
#thresholds = [0.0]

export_errors = True


# ----------------------------

# Get current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create input and output folders
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------

observations = pd.read_csv(
    os.path.join(input_path, input_camtrap_folder_name, "observations.csv"),
    dtype=str,
    encoding="utf-8"
    )
media = pd.read_csv(
    os.path.join(input_path, input_camtrap_folder_name, "media.csv"),
    dtype=str,
    encoding="utf-8"
)
deployments = pd.read_csv(
    os.path.join(input_path, input_camtrap_folder_name, "deployments.csv"),
    dtype=str,
    encoding="utf-8"
)

observations["classificationProbability"] = (
    observations["classificationProbability"]
    .fillna("")
    .astype(str)
    .str.strip()
)

observations["classificationProbability"] = (
    pd.to_numeric(
        observations["classificationProbability"],
        errors="coerce"
    )
    .fillna(0.0)
    .astype(float)
)

# -----------------------------
# Keep required columns
# -----------------------------
observations = observations[[
    "eventID",
    "deploymentID",
    "observationType",
    "scientificName",
    "classificationProbability",
    "classifiedBy"
]].rename(columns={
    "classificationProbability": "confidence"
})

media = media[["mediaID", "mediaComments"]]

deployments = deployments[["deploymentID", "deploymentTags"]]

# Extract model name
model_name_series = observations["classifiedBy"].dropna().astype(str).str.strip()
model_name_series = model_name_series[model_name_series != ""]
model_name = model_name_series.iloc[0] if not model_name_series.empty else "Unknown Model"

# Fill missing scientificName values for blank observations

observations.loc[
    observations["observationType"]
    .str.lower()
    .eq("blank"),
    "scientificName"
] = (
    observations.loc[
        observations["observationType"]
        .str.lower()
        .eq("blank"),
        "scientificName"
    ]
    .fillna("Blank")
)

# Fill species name with "Unclassified" if observationType is unclassifed
observations.loc[
    observations["observationType"]
    .str.lower()
    .eq("unclassified"),
    "scientificName"
] = (
    observations.loc[
        observations["observationType"]
        .str.lower()
        .eq("unclassified"),
        "scientificName"
    ]
    .fillna("Unclassified")
)


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
# Extract sequenceID
# -----------------------------
def extract_sequence(comment):
    if pd.isna(comment):
        return None
    match = re.search(r"sequenceID:([^\s]+)", str(comment))
    return match.group(1) if match else None


media["sequenceID"] = media["mediaComments"].apply(extract_sequence)


# -----------------------------
# Merge tables
# -----------------------------
merged = pd.merge(media, observations, left_on="sequenceID", right_on="eventID", how="left")
merged = pd.merge(merged, deployments, on="deploymentID", how="left")


# -----------------------------
# Normalize species names
# -----------------------------


merged["true_species"] = (
    merged["deploymentTags"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

merged.loc[
    merged["true_species"] == "",
    "true_species"
] = "blank"


merged["pred_species"] = (
    merged["scientificName"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

merged.loc[
    merged["pred_species"] == "",
    "pred_species"
] = "blank"



# -----------------------------
# Remove rows with missing truth
# -----------------------------
merged = merged[merged["true_species"].notna()]


# -----------------------------
# Evaluation metrics
# -----------------------------

results = []


species_list = sorted(merged["true_species"].dropna().unique())

for species in species_list:

    for threshold in thresholds:

        # Apply threshold
        filtered = merged.copy()

        filtered["passes_threshold"] = (
            filtered["confidence"] >= threshold
        )

        # TP
        tp = (
            (filtered["true_species"] == species) &
            (filtered["pred_species"] == species) &
            (filtered["passes_threshold"])
        ).sum()

        # FN
        fn = (
            (filtered["true_species"] == species) &
            (
                (filtered["pred_species"] != species) |
                (~filtered["passes_threshold"])
            )
        ).sum()

        # FP
        fp = (
            (filtered["true_species"] != species) &
            (filtered["pred_species"] == species) &
            (filtered["passes_threshold"])
        ).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        results.append({
            "species": format_species_name(species),
            "model_confidence": f">={threshold:.1f}",
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4)
        })

results_df = pd.DataFrame(results)

def format_species_name(name):
    if pd.isna(name):
        return name
    
    parts = str(name).strip().split()
    
    if not parts:
        return name
    
    return " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])

#results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
results_df["species_display"] = results_df["species"].apply(format_species_name)

# Reorder columns (optional)
results_df = results_df[
    ["species_display","model_confidence", "true_positives", "false_negatives", "false_positives", "recall", "precision", "f1_score"]
].rename(columns={"species_display": "species"})

# sort by species and model confidence
results_df = results_df.sort_values(
    by=["species", "model_confidence"]
)

# Extract numeric threshold value for sorting
results_df["threshold_value"] = (
    results_df["model_confidence"]
    .str.replace(">=", "", regex=False)
    .astype(float)
)

# calculate best threshold per species
best_thresholds = (
    results_df
    .sort_values(
        by=["species", "f1_score", "threshold_value"],
        ascending=[True, False, True]
    )
    .groupby("species")
    .first()
    .reset_index()
)

best_thresholds = best_thresholds[
    [
        "species",
        "model_confidence",
        "recall",
        "precision",
        "f1_score"
    ]
]

# remove threshold_value column from results_df for display purposes
results_df = results_df.drop(columns=["threshold_value"])

# -----------------------------
# Confusion matrix
# -----------------------------
conf_matrix = pd.crosstab(
    merged["true_species"],
    merged["pred_species"]
)



full_confusion_matrices = {}


for threshold in thresholds:

    # Apply confidence threshold
    subset = merged[
        merged["confidence"] >= threshold
    ].copy()

    # Replace blanks
    subset["pred_species"] = (
        subset["pred_species"]
        .fillna("Blank")
    )

    subset["true_species"] = (
        subset["true_species"]
        .fillna("Blank")
    )

    # Create confusion matrix
    conf_matrix = pd.crosstab(
        subset["true_species"],
        subset["pred_species"]
    )

    # Sort columns by total prediction count
    conf_matrix = conf_matrix.loc[
        :,
        conf_matrix.sum().sort_values(ascending=False).index
    ]

    # Sort rows by total occurrence
    conf_matrix = conf_matrix.loc[
        conf_matrix.sum(axis=1)
        .sort_values(ascending=False)
        .index
    ]

    full_confusion_matrices[
        f">={threshold:.1f}"
    ] = conf_matrix

formatted_full_confusion_matrices = {}

for threshold, matrix in full_confusion_matrices.items():

    matrix = matrix.copy()

    # Format row names
    matrix.index = [
        format_species_name(x)
        if x != "Blank" else "Blank"
        for x in matrix.index
    ]

    # Format column names
    matrix.columns = [
        format_species_name(x)
        if x != "Blank" else "Blank"
        for x in matrix.columns
    ]


    # Set index name before reset
    matrix.index.name = "predicted_species→\ntrue_species↓"

    # Reset index for HTML display
    matrix_reset = matrix.reset_index()

    formatted_full_confusion_matrices[
        threshold
    ] = matrix_reset


single_row_matrices = {}

species_list = sorted(merged["true_species"].dropna().unique())

for species in species_list:

    rows = []

    for threshold in thresholds:

        # Get only rows where this is the true species
        subset = merged[
            merged["true_species"] == species
        ].copy()

        # Apply confidence threshold
        subset = subset[
            subset["confidence"] >= threshold
        ]

        # Count predictions
        pred_counts = (
            subset["pred_species"]
            .fillna("NaN")
            .value_counts()
        )

        # Convert counts to dictionary
        row = pred_counts.to_dict()

        # Add threshold label
        row["confidence_threshold"] = f">={threshold:.1f}"

        # Add row label
        #row["pred_species"] = "count"

        rows.append(row)

    # Convert all rows to DataFrame
    sub_matrix = pd.DataFrame(rows)

    # Fill blanks
    sub_matrix = sub_matrix.fillna(0)

    # Convert numeric columns to integers
    numeric_cols = sub_matrix.columns.difference(
        ["confidence_threshold"]
    )

    sub_matrix[numeric_cols] = (
        sub_matrix[numeric_cols]
        .astype(int)
    )


    # Reorder columns
    fixed_cols = ["confidence_threshold"]

    other_cols = [
        c for c in sub_matrix.columns
        if c not in fixed_cols
    ]

    

    # Sort prediction columns by total occurrence
    other_cols = (
        sub_matrix[other_cols]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    sub_matrix = sub_matrix[
        fixed_cols + other_cols
    ]

    # define formatted name for confidence threshold column
    confidence_threshold_formatted_name = "predicted_species→\nconfidence_threshold↓"
    sub_matrix = sub_matrix.rename(columns={"confidence_threshold": confidence_threshold_formatted_name})

    single_row_matrices[species] = sub_matrix

    

formatted_single_row = {}

for species, matrix in single_row_matrices.items():

    # Format species names in prediction columns
    renamed_cols = {}

    for col in matrix.columns:

        if col not in [confidence_threshold_formatted_name]:

            renamed_cols[col] = format_species_name(col)

    matrix = matrix.rename(columns=renamed_cols)

    formatted_single_row[
        format_species_name(species)
    ] = matrix


conf_matrix.index = conf_matrix.index.map(format_species_name)
conf_matrix.columns = conf_matrix.columns.map(format_species_name)



# -----------------------------
# Misclassified images export
# -----------------------------

misclassified_images_report = output_path + "/" + f"Misclassified_Images_{model_name}.csv"

if export_errors:
    errors = merged[merged["true_species"] != merged["pred_species"]]
    errors.to_csv(misclassified_images_report, index=False)


# %%
# generate a nicely formatted report that can be shared with others as a .html file

html_file = output_path + "/" + f"Model_Benchmark_Test_Report_{model_name}.html"

style = """
<style>

body {
    font-family: Arial;
    margin: 0;
    padding: 0;
}

.report-container {
    max-width: 1400px;
    margin: 20px auto;
    padding: 20px;
}

h1 {
    color: #2c3e50;
}

h2 {
    color: #34495e;
    margin-top: 30px;
}

h3 {
    color: #2c3e50;
    margin-top: 20px;
}

p,
ul,
ol {
    max-width: 900px;
    line-height: 1.6;
}

ul,
ol {
    padding-left: 25px;
    margin-bottom: 16px;
}

.styled-table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
}

.styled-table th,
.styled-table td {
    border: 1px solid #cccccc;
    padding: 6px 10px;
    text-align: center;
}

.styled-table th {
    background-color: #f2f2f2;
    position: sticky;
    top: 0;
    z-index: 2;
    font-weight: bold;
}

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
    f.write('<div class="report-container">')


    # -----------------------------
    # Summary information
    # -----------------------------

    f.write("<h1>Model Evaluation Report</h1>")
    
    f.write(f"<h2>Test details</h2>")
    f.write(f"<p><b>Timestamp of report: </b>{timestamp}</p>")
    f.write(f"<p><b>Computer Vision (CV) model tested: </b>{model_name}</p>")
    f.write(f"<p><b>Source location of test images: </b>{data_source_location}</p>")
    f.write("<p><b>Species (or labels) evaluated:</b></p>")
    f.write("<ul>")
    for species in species_list_display:
        f.write(f"<li><i>{species}</i></li>")
    f.write("</ul>")
    f.write(f"<p><b>Confidence thresholds used: </b>{thresholds}</p>")
    f.write(f"<p><b>Data collation process used: </b>{data_collation_process}</p>")

    # -----------------------------
    # Explanation Section
    # -----------------------------

    f.write("<h2>Purpose and intended use of this report</h2>")
    f.write("""
        <ol>
            <li>Inform the potential suitability of a model for a given location and set of species relevant to the monitoring program.</li>
            <li>Inform the design of a validation workflow (manual human review) to ensure a sufficient level of accuracy can be reached to meet the goals of the monitoring program.</li>
            <li>Inform potential gaps in model performance and priorities for provision of additional training images to improve performance.</li>
        </ol>
    """)
    
    f.write("<h2>Limitations</h2>")

    f.write("""
    <p>
    The accuracy of the results presented in this report is primarily limited by the quality of the test data used. If the input dataset is not representative of real-world conditions, model performance metrics may be misleading.
    It is important to follow recommended guidelines when constructing a test dataset to ensure it is balanced, representative, and consistent across species and environments.
    </p>

    <p>
    Species name mismatches can also significantly affect recall and precision scores. Some models may aggregate multiple taxonomic species into a single prediction label. Conversely, the test dataset may contain fine-grained species labels that do not directly match the model’s output taxonomy.
    Where mismatches occur, inspection of the confusion matrix will often help identify the label conventions used by the model and clarify how predictions are being grouped.
    </p>

    <p>
    Finally, the results presented in this report should not be interpreted as definitive proof that one model is superior to another. Instead, they are intended to support informed selection of the most appropriate model for a given location, application, and set of target species.
    </p>
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
    <table class="styled-table">
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
    <table class="styled-table">
        <tr><th>Scenario</th><th>Interpretation</th></tr>
        <tr><td>High Recall, Low Precision</td><td>Finds most animals but includes many false positives</td></tr>
        <tr><td>Low Recall, High Precision</td><td>Accurate predictions but misses many animals</td></tr>
        <tr><td>High Recall, High Precision</td><td>Ideal performance</td></tr>
        <tr><td>Low Recall, Low Precision</td><td>Poor performance</td></tr>
    </table>
    """)

    f.write("""
    <h3>What is a Confusion Matrix?</h3>

    <p>
    A confusion matrix summarises model performance by comparing the true species labels against the species predicted by the model.
    In simple terms, the confusion matrix shows what the model classified correctly and where errors occurred, broken down by type of misclassification.
    </p>

    <p>
    Inspecting the confusion matrix can help identify the direction of error in cases where images were classified incorrectly. This is often useful for understanding which species are being confused with one another.
    It is also important to recognise that not all models define or name species in the same way. Some models may group multiple species into broader taxonomic categories such as Genus or Family, while others may use highly specific species labels.
    If recall or precision values are unexpectedly low for a particular species, inspection of the confusion matrix may help reveal inconsistencies in naming conventions, taxonomy, or label aggregation used by the model.
    </p>
    """)

    # -----------------------------
    # Results Tables
    # -----------------------------
    f.write("<h2>Recall, Precision, and F1 Results</h2>")
    f.write(results_df.to_html(classes="styled-table",index=False))

    f.write("<h2>Optimal Confidence Threshold by Species</h2>")

    f.write("""
    <p>
    The table below shows the confidence threshold that produced the highest F1 score for each species. This can help identify the optimal balance between recall and precision for operational use.
    </p>
    """)

    f.write(
        best_thresholds.to_html(
            index=False,
            classes="styled-table",
            border=0
        )
    )

    
    f.write("<h2>Prediction Distribution by Species</h2>")

    f.write("""
    <p>
    The outputs below represent simplified confusion matrices highlighting
    the distribution of predictions relevant to each target (test) species
    across multiple confidence thresholds.
    Refer to the full confusion matrix for every combination of
    True Positive, False Positive, and False Negative.
    </p>
    """)

    for species, matrix in formatted_single_row.items():

        f.write(
            f"<h3>True species (or label): "
            f"<i>{species}</i></h3>"
     )

        f.write('<div style="overflow-x:auto;">')
        f.write(
            matrix.to_html(
                index=False,
                classes="styled-table",
                border=0
            )
        )
        f.write('</div>')

    f.write("<br>")


    f.write("<h2>Full Confusion Matrices</h2>")

    f.write("""
    <p>
    The outputs below represent the complete confusion matrix
    for each confidence threshold.
    Predictions below the specified threshold are excluded.
    </p>
    """)

    for threshold, matrix in formatted_full_confusion_matrices.items():

        f.write(
            f"<h3>Confusion Matrix - Confidence Threshold: "
            f"{threshold}</h3>"
        )

        f.write('<div style="overflow-x:auto;">')
        f.write(
            matrix.to_html(
                index=False,
                classes="styled-table",
                border=0
            )
        )
        f.write('</div>')

        f.write("<br>")

    f.write("</div>")
    f.write("</body></html>")

file_path = Path(html_file).resolve()
print(f"Report saved to: {file_path}")

# %%
# Optional: display HTML report directly within this notebook

with open(html_file, "r", encoding="utf-8") as f:
    html_content = f.read()

# Inject MathJax
mathjax = r"""
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['\\[', '\\]']]
  }
};
</script>
<script id="MathJax-script" async
src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
"""

display(HTML(mathjax + html_content))


