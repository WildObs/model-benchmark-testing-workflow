# Model evaluation workflow designed for WildObs Image Management Platform https://wildobs.org.au/

## Description
- The GUI notebook can be used for benchmarking an AI species recognition model with a local dataset to get an independent assessment of Recall, Precision and F1 Score for a given location
- The purpose is to evaluate the most suitable or best performing model for a given location and inform an appropriate validation workflow to ensure accuracy requirements are met

## Setup instructions:
1) Prepare the testing dataset. Start by organising camera trap images into folders by species on your local computer. The quality of the testing dataset will determine the accuracy of the report generated. Here are a few tips:
- Use a representive number of images of each species e.g. at least 100 if possible
- If you have a very large number of images of some species, extract a random sample e.g. 1000 rather than processing every available image. This saves you time and reduces unnecessary system usage.
- Avoid using images that have been used as part of the model training dataset as these will create a biased result
- For the purpose of calculating Recall, include species that are relevant to your monitoring program
- For the the purpose of calculating Precision, also include images of other species that are commonly detected on the cameras at that location even if they are not relevant to your monitoring. Also include some blank images. Below is an example breakdown:

| Species                  | # images | Reason for inclusion                     |
|:-------------------------|:---------|:-----------------------------------------|
| Felis catus              | 500      | Target species in monitoring program     |
| Vulpes vulpes            | 500      | Target species in monitoring program     |
| Oryctolagus cuniculus    | 1000     | Target species in monitoring program     |
| Osphranter rufus         | 1000     | Non-target species but abundant at this location. Impact on Precision. |
| Macropus giganteus       | 1000     | Non-target species but abundant at this location. Impact on Precision. |
| Notamacropus rufogriseus | 1000     | Non-target species but abundant at this location. Impact on Precision. |
| Wallabia bicolor         | 1000     | Non-target species but abundant at this location. Impact on Precision. |
| Dromaius novaehollandiae | 1000     | Non-target species but abundant at this location. Impact on Precision. |
| Blank                    | 1000     | Impact on Precision. |

2) Establish new Project/s in the WildObs WIMP for benchmarking purposes:
-  You will need a separate Project for each model you are testing. 
-  Name the project based on the model that will be tested e.g. "Model benchmark testing: WildObs National".
-  Set the Sequence cutoff to 0 seconds. This aims to prevent the software from creating sequences so that each image is assessed independently.
-  Define Tags in the project based on the scientific names of the species you are testing. Tags need to match with the species names used in the WIMP.
-  Configure the project to use the model you want to test
3) Create Deployments:
- You will need to create a Deployment for each of the species you are testing.
- Upload the relevant images into each deployment.
- Use the tags created earlier to assign to the deployment so you know which species it is supposed to be. The GUI workflow uses this tag to match the species to the model predictions.
- Repeat for each Project, uploading the same set of images to each
4) Run the uploaded images through the AI species recognition model
5) Once model processing is complete for all deployments, export the project data in Camtrap DP format
6) Download and extract (unzip) the exported data to a folder on your local computer
7) Organise exports into a testing set folder inside `Input_Data/` (e.g. `Input_Data/Edgbaston_2026_10/`). A set contains one or more Camtrap DP export folders for the same set of images processed through different models.
8) Open `WildObs-CV-Model-Evaluation-GUI.ipynb` from the repository root, select the test set folder in section 1, and run the evaluation.

## Prerequisites

- Python 3.x
- Required packages listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/WildObs/model-benchmark-testing-workflow.git
cd model-benchmark-testing-workflow
```

## Project structure

```
model-benchmark-testing-workflow/
├── WildObs-CV-Model-Evaluation-GUI.ipynb                  # primary user interface
├── scripts/
│   ├── WildObs-CV-model-benchmark-testing-workflow.ipynb   # development notebook
│   └── run_benchmark.py                                    # development command-line entry point
├── src/                  # functional code (config, data loading, metrics, templating, reports, logging)
├── templates/            # report templates (full, minimal, or your own) and stylesheet
├── tests/
├── Input_Data/           # test sets containing Camtrap DP exports (not tracked)
│   ├── Wychitella_2026_09/
│   │   ├── CamtrapDP_Export_Model_1/
│   │   └── CamtrapDP_Export_Model_2/
│   └── Edgbaston_2026_10/
│       ├── CamtrapDP_Export_Model_1/
│       └── CamtrapDP_Export_Model_2/
├── Output_Reports/       # matching set subfolders containing generated reports (not tracked)
│   ├── Wychitella_2026_09/
│   │   ├── Evaluation_Report_Model_1.html
│   │   ├── Evaluation_Report_Model_2.html
│   │   └── Comparison_Report.html
│   └── Edgbaston_2026_10/
│       ├── Evaluation_Report_Model_1.html
│       ├── Evaluation_Report_Model_2.html
│       └── Comparison_Report.html
├── logs/                 # run logs (not tracked)
└── requirements.txt
```

## Data Preparation
1. Log in to the WildObs image management platform
2. Export your dataset in Camtrap-DP format for each model being tested
3. Extract the exports into a set subfolder inside `Input_Data/` (e.g. `Input_Data/Edgbaston_2026_10/`). Each export folder must contain `observations.csv`, `media.csv` and `deployments.csv`.

## Using the GUI notebook

Open `WildObs-CV-Model-Evaluation-GUI.ipynb` from the repository root and run its setup cell. Select a test set in section 1 (the form previews the models, image counts and observation levels it finds), enter the report details, choose confidence thresholds, observation levels and a report template, then click **Run evaluation**. The individual model reports and, when applicable, the comparison report are available in expandable sections beneath the form and are saved to `Output_Reports/<test set>`. Taxonomy lookup, clearing previous outputs, misclassified-image export and log level are in the collapsed **Advanced** section.

Install the notebook widget dependency with `pip install ipywidgets` if it is not already installed. Install all project dependencies with `pip install -r requirements.txt`.

A single run generates:
- An individual model evaluation report for each Camtrap DP export in the set
- An overall model comparison report for the set (generated automatically when the set contains 2 or more models)
- An optional CSV of misclassified images for each model

All reports for the set are stored in the matching subfolder in `Output_Reports/<test_set_name>/`.

## Choosing the report format
Reports are generated from Markdown templates in `templates/`:

- `full`: test details, purpose, limitations, explanations of the metrics and all results
- `minimal`: test details and results only

Choose the report template in section 4 of the GUI notebook. To create your own template, copy an existing one and see [templates/README.md](templates/README.md).

Taxonomy lookup is enabled by default in the GUI notebook's **Advanced** section. It replaces the species list in the report with a taxonomy table (number of images, matched name, taxonomic rank, kingdom to genus, common name) looked up from the Atlas of Living Australia using the [galah](https://galah.ala.org.au/Python/) package. This needs an internet connection. Labels with no valid match (such as `blank`) are kept with empty lookup values, and a failed lookup never stops the report.

### Event-based or media-based observations

Newer WildObs Camtrap DP exports can optionally include bounding boxes, in which case `observations.csv` contains both `event` and `media` rows in its `observationLevel` column (legacy exports and exports without bounding boxes contain `event` rows only). The workflow analyses one level at a time, never both:

- **Event-based (recommended):** event-based observations, which every export supports.
- **Media-based:** media-based observations, joined to images directly by `mediaID`. This is only used if **every** export in the test set contains media-level observations. If any export does not (for example a set mixing old and new exports), the whole set reverts to event-based and a warning is logged, so the reports stay directly comparable. Images with no media-level observation (blank or unclassified events have no bounding box) are scored as `blank` predictions.

## Outputs
Every run has a timestamp. The GUI notebook's **Advanced** section has **Clear existing outputs for this set first** enabled by default; untick it to keep outputs from previous runs.

Outputs are organized into matching set subfolders:

- `Output_Reports/<test_set_name>/WildObs_CV_Model_Evaluation_Report_<data_source_location>_<Model_Name>_<TimeStamp>.html`: the HTML evaluation report for each model in the set.
- `Output_Reports/<test_set_name>/WildObs_CV_Model_Evaluation_Comparison_Report_<TimeStamp>.html`: the multi-model comparison report for the set (generated when the set contains 2 or more models).
- `Output_Reports/<test_set_name>/Misclassified_Images_<model>_<timestamp>.csv`: records of incorrectly classified images, useful for error analysis (optional, disabled by default; enable **Export misclassified images CSV** in the GUI notebook's **Advanced** section).
- `logs/benchmark_<timestamp>.log`: detailed log of the run, including warnings about data issues such as untagged deployments or images without a matching observation.

## Example Workflow
1. Export datasets from WildObs (Camtrap-DP format) for each model evaluated
2. Extract them into a testing set folder in `Input_Data/` (e.g. `Input_Data/Wychitella_2026_09/`)
3. Install dependencies and open `WildObs-CV-Model-Evaluation-GUI.ipynb` from the repository root
4. Run the setup cell, complete the form, and click **Run evaluation**
5. Review the individual evaluation reports, the set comparison report, and any misclassified images CSV. Expand **Log messages** if you need to inspect the run log.

## Troubleshooting

#### Missing packages
Run:

```bash
pip install -r requirements.txt
```

#### File not found errors
Ensure your exported dataset is extracted in `Input_Data/` and select the matching `Input_Data/<test set>` option in section 1 of the GUI notebook.

#### Unexpected results
Open the latest file in `logs/`. To see more detail while running, choose `DEBUG` from **Log level** in the GUI notebook's **Advanced** section.

#### No outputs generated
Export the data again from the WildObs platform and extract it into `Input_Data/` to make sure that you are using the most recent classification results acquired from WildObs models.

## Development
Run the tests with:

```bash
python -m pytest tests
```

## License

Apache License 2.0
