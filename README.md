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
7) Use the folder path as input to this script

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
│   ├── WildObs-CV-model-benchmark-testing-workflow.ipynb   # notebook: set preferences and run
│   └── run_benchmark.py                                    # command line version
├── src/                  # functional code (config, data loading, metrics, templating, reports, logging)
├── templates/            # report templates (full, minimal, or your own) and stylesheet
├── tests/
├── Input_Data/           # extracted Camtrap DP exports (not tracked)
├── Output_Reports/       # generated reports (not tracked)
├── logs/                 # run logs (not tracked)
└── requirements.txt
```

## Data Preparation
1. Log in to the WildObs image management platform
2. Export your dataset in Camtrap-DP format
3. Extract the export into a new folder inside `Input_Data/`. The folder must contain `observations.csv`, `media.csv` and `deployments.csv`.

## Usage

### Notebook
Open `scripts/WildObs-CV-model-benchmark-testing-workflow.ipynb`, edit the configuration cell (input folder, thresholds, template, etc.) and run the cells.

### Command line

```bash
python scripts/run_benchmark.py <input_folder_name> --template minimal
python scripts/run_benchmark.py --list-templates
python scripts/run_benchmark.py --help
```

## Choosing the report format
Reports are generated from Markdown templates in `templates/`:

- `full`: test details, purpose, limitations, explanations of the metrics and all results
- `minimal`: test details and results only

Individual sections of any template can be switched off with `exclude_sections` (notebook) or `--exclude-sections` (command line), for example `Limitations` or `Full_Confusion_Matrices`. To create your own template, copy an existing one and see [templates/README.md](templates/README.md).

Set `lookup_taxonomy=True` (notebook) or pass `--lookup-taxonomy` (command line) to replace the species list in the report with a taxonomy table (matched name, taxonomic rank, kingdom to genus, common name) looked up from the Atlas of Living Australia using the [galah](https://galah.ala.org.au/Python/) package. This needs an internet connection and `pip install galah`. Labels with no valid match (such as `blank`) are kept with empty lookup values, and a failed lookup never stops the report.

## Outputs
Every run has a timestamp so earlier results are never overwritten.

- `Output_Reports/Model_Benchmark_Test_Report_<model>_<template>_<timestamp>.html`: the HTML report
- `Output_Reports/Misclassified_Images_<model>_<timestamp>.csv`: records of incorrectly classified images, useful for error analysis (optional)
- `logs/benchmark_<timestamp>.log`: detailed log of the run, including warnings about data issues such as untagged deployments or images without a matching observation

## Example Workflow
1. Export dataset from WildObs (Camtrap-DP format)
2. Extract it into `Input_Data/`
3. Install dependencies
4. Choose a template and run the notebook or script
5. Review the report and the misclassified images CSV. Check the log if anything looks unexpected.

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
