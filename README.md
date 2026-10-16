# Model evaluation workflow designed for WildObs Image Management Platform https://wildobs.org.au/

## Description
- This script can be used for benchmarking an ai species recognition model with a local dataset to get an independent assessment of Recall, Precision and F1 Score for a given location
- The purpose is to evaluate the most suitable or best performing model for a given location and inform an appropriate validation workflow to ensure accuracy requirements are met

## Setup instructions:
1) Prepare the testing dataset. Start by organising camera trap images into folders by species on your local computer. The quality of the testing dataset will determine the accuracy of the report generated. Here are a few tips:
- Use a representive number of images of each species e.g. at least 1000 if possible
- Avoid using images that have been used as part of the model training dataset as these will create a biased result
- If possible, select a random subset of testing images from a larger pool aiming to get a wide range of images over space and time
- For the purpose of calculating Recall, include species that are relevant to your monitoring program
- For the the purpose of calculating Precision, also include images of other species that are commonly detected on the cameras at that location even if they are not relevant to your monitoring. Also include some blank images. Below is an example breakdown:

| Species         | # images | Reason for inclusion                     |
|:----------------|:---------|:-----------------------------------------|
| Feral Cat       | 1000     | Target species in monitoring program     |
| Red Fox         | 1000     | Target species in monitoring program     |
| European Rabbit | 1000     | Target species in monitoring program     |
| Kangaroo        | 1000     | Non-target species but abundant at this location. Impact on Precision. |
| Emu             | 1000     | Non-target species but abundant at this location. Impact on Precision. |
| Blank           | 1000     | Impact on Precision. |

2) Establish new Project/s in the WildObs WIMP for benchmarking purposes:
-  You will need a separate Project for each model you are testing. 
-  Name the project based on the model that will be tested e.g. "Model benchmark testing: WildObs National".
-  Set the Sequence cutoff to 0 seconds. This aims to prevent the software from creating sequences so that each image is assessed independently.
-  Define Tags in the project based on the scientific names of the species you are testing. Tags need to match with the species names used in the WIMP.
-  Configure the project to use the model you want to test
3) Create Deployments:
- You will need to create a Deployment for each of the species you are testing.
- Upload the relevant images into each deployment.
- Use the tags created earlier to assign to the deployment so you know which species it is supposed to be. This will be used by the script to match the species to the model predictions
- Repeat for each Project, uploading the same set of images to each
4) Run the uploaded images through the AI species recognition model
5) Once model processing is complete for all deployments, export the project data in Camtrap DP format
6) Download and extract (unzip) the exported data to a folder on your local computer
7) Organise exports into a testing set folder inside `Input_Data/` (e.g. `Input_Data/Edgbaston_2026_10/`). A set contains one or more Camtrap DP export folders for the same set of images processed through different models.
8) Use the test set folder name as input to the workflow

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
├── scripts/
│   ├── WildObs-CV-model-benchmark-testing-workflow.ipynb   # notebook: batch test set evaluation & comparison
│   ├── WildObs-CV-model-benchmark-testing-form.ipynb       # notebook: same workflow driven by a point-and-click form
│   ├── WildObs-CV-model-comparison.ipynb                   # notebook: compare model reports
│   └── run_benchmark.py                                    # command line version
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

## Usage

### Notebook (form version, no coding)
Open `scripts/WildObs-CV-model-benchmark-testing-form.ipynb`, run the two cells, and use the form to pick a test set (the form previews the models, image counts and observation levels it finds), enter the report details, choose a threshold preset, observation level and template, and click **Run evaluation** (taxonomy lookup, clearing old outputs, misclassified-image export and log level are under the collapsed **Advanced** section). The report is displayed under the form and saved to `Output_Reports/<test set>`. It needs `ipywidgets` (`pip install ipywidgets`). The form logic lives in `src/ui.py`.

### Notebook
Open `scripts/WildObs-CV-model-benchmark-testing-workflow.ipynb`, edit the configuration cell (`test_set_name`, thresholds, template, etc.) and run the cells.

A single run generates:
- An individual model evaluation report for each Camtrap DP export in the set
- An overall model comparison report for the set (generated automatically when the set contains 2 or more models)
- An optional CSV of misclassified images for each model

All reports for the set are stored in the matching subfolder in `Output_Reports/<test_set_name>/`.

To independently compare multiple evaluation reports from an existing folder, open `scripts/WildObs-CV-model-comparison.ipynb`, set the test set name or folder, and run the notebook.

### Command line

```bash
python scripts/run_benchmark.py Edgbaston_2026_10 --template minimal
python scripts/run_benchmark.py Wychitella_2026_09 --clear-existing
python scripts/run_benchmark.py --list-templates
python scripts/run_benchmark.py --help
```

## Choosing the report format
Reports are generated from Markdown templates in `templates/`:

- `full`: test details, purpose, limitations, explanations of the metrics and all results
- `minimal`: test details and results only

Individual sections of any template can be switched off with `exclude_sections` (notebook) or `--exclude-sections` (command line), for example `Limitations` or `Full_Confusion_Matrices`. To create your own template, copy an existing one and see [templates/README.md](templates/README.md).

Set `lookup_taxonomy=True` (notebook) or pass `--lookup-taxonomy` (command line) to replace the species list in the report with a taxonomy table (number of images, matched name, taxonomic rank, kingdom to genus, common name) looked up from the Atlas of Living Australia using the [galah](https://galah.ala.org.au/Python/) package. This needs an internet connection and `pip install galah`. Labels with no valid match (such as `blank`) are kept with empty lookup values, and a failed lookup never stops the report.

### Event-based or media-based observations

Newer WildObs Camtrap DP exports can optionally include bounding boxes, in which case `observations.csv` contains both `event` and `media` rows in its `observationLevel` column (legacy exports and exports without bounding boxes contain `event` rows only). The workflow analyses one level at a time, never both:

- `observation_level="event"` (default; `--observation-level event`): event-based observations, which every export supports.
- `observation_level="media"` (`--observation-level media`): media-based observations, joined to images directly by `mediaID`. This is only used if **every** export in the test set contains media-level observations. If any export does not (for example a set mixing old and new exports), the whole set reverts to `event` and a warning is logged, so the reports stay directly comparable. Images with no media-level observation (blank or unclassified events have no bounding box) are scored as `blank` predictions.

## Outputs
Every run has a timestamp. To prevent reports from accumulating across multiple runs, set `clear_existing_outputs=True` in the notebook configuration or pass `--clear-existing` on the command line to clear existing reports and outputs in the set folder before generating new ones.

Outputs are organized into matching set subfolders:

- `Output_Reports/<test_set_name>/WildObs_CV_Model_Evaluation_Report_<data_source_location>_<Model_Name>_<TimeStamp>.html`: the HTML evaluation report for each model in the set.
- `Output_Reports/<test_set_name>/WildObs_CV_Model_Evaluation_Comparison_Report_<TimeStamp>.html`: the multi-model comparison report for the set (generated when the set contains 2 or more models).
- `Output_Reports/<test_set_name>/Misclassified_Images_<model>_<timestamp>.csv`: records of incorrectly classified images, useful for error analysis (optional, disabled by default; enable with `export_errors=True` or `--export-errors`).
- `logs/benchmark_<timestamp>.log`: detailed log of the run, including warnings about data issues such as untagged deployments or images without a matching observation.

## Example Workflow
1. Export datasets from WildObs (Camtrap-DP format) for each model evaluated
2. Extract them into a testing set folder in `Input_Data/` (e.g. `Input_Data/Wychitella_2026_09/`)
3. Install dependencies
4. Choose a template and run the notebook or command line script specifying the set name
5. Review the individual evaluation reports, the set comparison report, and the misclassified images CSV. Check the log if anything looks unexpected.

## Troubleshooting

#### Missing packages
Run:

```bash
pip install -r requirements.txt
```

#### File not found errors
Ensure your exported dataset is extracted in `Input_Data/` and that the folder name matches the configuration. The error message lists the folders that were found.

#### Unexpected results
Open the latest file in `logs/`. Set `log_level` to `DEBUG` for more detail on screen.

#### No outputs generated
Export the data again from the WildObs platform and extract it into `Input_Data/` to make sure that you are using the most recent classification results acquired from WildObs models.

## Development
Run the tests with:

```bash
python -m pytest tests
```
## Contributing

Contributions are welcome:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

## License

Apache License 2.0
