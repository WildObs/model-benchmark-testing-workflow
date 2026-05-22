# Model evaluation workflow: alternative version designed for eVorta (in development)

## Description
- This script can be used for benchmarking an ai species recognition model with a local dataset to get an independent assessment of Recall, Precision and F1 Score for a given location
- The purpose is to evaluate the most suitable or best performing model for a given location and inform an appropriate validation workflow to ensure accuracy requirements are met

## Setup instructions:
1) Prepare the testing dataset. Start by organising camera trap images into folders by species on your local computer. Match the names of the folders to the label classes used by eVorta. These are generally the common names of the species e.g. cat, fox, emu. The quality of the testing dataset will determine the accuracy of the report generated. Here are a few tips:
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

2) Before processing the images on eVorta, make sure your images have unique file names. The methodology of this workflow relies on matching images based on their names. An optional pre-processing step is available to generate unique file names.
3) Upload the images to eVorta and process through the model/s that you wish to test. 
4) Once processing is complete, export the eVorta model predictions to csv. Before exporting, use the filter options to set the confidence threshold to "0". This will ensure all predictions with a confidence of 0 or higher will be included in the export. Check other filters to make sure all relevant records are being included in the export.
5) Download and extract (unzip) the exported data to a folder on your local computer
6) Use the file path as input to this script

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

## Data Preparation
1. Log in to the WildObs image management platform
2. Export your dataset in Camtrap-DP format
3. Place the exported files in the root directory of this repository

Example structure:

```bash
model-benchmark-testing-workflow/
│
├── wildObs-CV-model-benchmark-testing-workflow.py
├── requirements.txt
├── README.md
├── data/                  # (optional) your exported dataset
│   ├── observations.csv
│   ├── media.csv
│   └── deployments.csv
│   └── ...
```

Note: Adjust paths in the script if your data is stored in a different location.

## Usage

Run the workflow from the command line:

```bash
python wildObs-CV-model-benchmark-testing-workflow.py
```

OR

Run the workflow from the Python notebook WildObs-CV-model-benchmark-testing-workflow.ipynb

## Outputs
#### Misclassified Images

misclassified_images.csv

- Contains records of incorrectly classified images
- Useful for error analysis and model comparison

#### Model Benchmark Reports

model_Benchmarking_Report_Exports/

- Directory containing HTML reports for each model
- Each report includes performance metrics and visual summaries

## Example Workflow
1. Export dataset from WildObs (Camtrap-DP format)
2. Place data in project folder
3. Install dependencies
4. Run the script
5. Review outputs:
- misclassified_images.csv
- HTML reports in model_Benchmarking_Report_Exports/

## Troubleshooting

#### Missing packages
Run:

```bash
pip install -r requirements.txt
```

#### File not found errors
Ensure your exported dataset is in the correct directory

#### No outputs generated
Export the data again from the WildObs platform and place the extracted data in the /data folder to make sure that you are using the most recent classification results acquired from WildObs models.

## Contributing

Contributions are welcome:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

## License

Apache License 2.0
