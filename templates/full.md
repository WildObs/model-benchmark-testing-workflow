<!-- description: Complete report including purpose, limitations, metric explanations and all results. -->

# Model Evaluation Report

<!-- BEGIN:Test_Details -->
## Test details

**Timestamp of report:** {Report_Timestamp}

**Computer Vision (CV) model tested:** {Model_Name}

**Source location of test images:** {Data_Source_Location}

**Species (or labels) evaluated:**

{Species_List}

**Confidence thresholds used:** {Confidence_Thresholds}

**Data collation process used:** {Data_Collation_Process}
<!-- END:Test_Details -->

<!-- BEGIN:Purpose -->
## Purpose and intended use of this report

1. Inform the potential suitability of a model for a given location and set of species relevant to the monitoring program.
2. Inform the design of a validation workflow (manual human review) to ensure a sufficient level of accuracy can be reached to meet the goals of the monitoring program.
3. Inform potential gaps in model performance and priorities for provision of additional training images to improve performance.
<!-- END:Purpose -->

<!-- BEGIN:Limitations -->
## Limitations

The accuracy of the results presented in this report is primarily limited by the quality of the test data used. If the input dataset is not representative of real-world conditions, model performance metrics may be misleading. It is important to follow recommended guidelines when constructing a test dataset to ensure it is balanced, representative, and consistent across species and environments.

Species name mismatches can also significantly affect recall and precision scores. Some models may aggregate multiple taxonomic species into a single prediction label. Conversely, the test dataset may contain fine-grained species labels that do not directly match the model’s output taxonomy. Where mismatches occur, inspection of the confusion matrix will often help identify the label conventions used by the model and clarify how predictions are being grouped.

Finally, the results presented in this report should not be interpreted as definitive proof that one model is superior to another. Instead, they are intended to support informed selection of the most appropriate model for a given location, application, and set of target species.
<!-- END:Limitations -->

<!-- BEGIN:Metric_Definitions -->
## Understanding the Results

Recall, precision, and F1 score are standard metrics used to evaluate the performance of machine learning models. Each metric has a precise definition and corresponding mathematical formula.

### Recall (Sensitivity)

\[ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}} \]

Of all the images that actually contain the target species, how many did the model correctly identify?

**Example:** If there are 100 images of Feral Cat within a dataset, and the model correctly detects 90 of them, recall = 0.9 (90%).

**Interpretation:** Quantifying recall helps us to understand how many true detections of a given species might be missed by the model.

### Precision

\[ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}} \]

Of all the images the model predicts as the target species, how many are actually correct?

**Example:** If the model makes 100 predictions of Feral Cat within a dataset, and only 90 of them are actually detections of Feral Cat, precision = 0.9 (90%).

**Interpretation:** High precision means predictions are reliable. Low precision means more false detections.

### F1 Score

\[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}} \]

A balanced measure that combines both recall and precision.

**Interpretation:** High F1 score indicates a good balance between detecting animals and avoiding false detections.
<!-- END:Metric_Definitions -->

<!-- BEGIN:Terms_And_Interpretation -->
### Definition of Terms

| Term | Definition | Shorthand definition |
|:--|:--|:--|
| **True Positive (TP)** | The model correctly predicts the presence of a species. | Correct detection |
| **False Positive (FP)** | The model predicts a species, but that species is not actually present. | False alarm |
| **False Negative (FN)** | The model fails to detect a species that is actually present. | Missed detection |

### How to interpret results

| Scenario | Interpretation |
|:--|:--|
| High Recall, Low Precision | Finds most animals but includes many false positives |
| Low Recall, High Precision | Accurate predictions but misses many animals |
| High Recall, High Precision | Ideal performance |
| Low Recall, Low Precision | Poor performance |
<!-- END:Terms_And_Interpretation -->

<!-- BEGIN:Confusion_Matrix_Explanation -->
### What is a Confusion Matrix?

A confusion matrix summarises model performance by comparing the true species labels against the species predicted by the model. In simple terms, the confusion matrix shows what the model classified correctly and where errors occurred, broken down by type of misclassification.

Inspecting the confusion matrix can help identify the direction of error in cases where images were classified incorrectly. This is often useful for understanding which species are being confused with one another. It is also important to recognise that not all models define or name species in the same way. Some models may group multiple species into broader taxonomic categories such as Genus or Family, while others may use highly specific species labels. If recall or precision values are unexpectedly low for a particular species, inspection of the confusion matrix may help reveal inconsistencies in naming conventions, taxonomy, or label aggregation used by the model.
<!-- END:Confusion_Matrix_Explanation -->

<!-- BEGIN:Results -->
## Recall, Precision, and F1 Results

{Results_Table}
<!-- END:Results -->

<!-- BEGIN:Optimal_Thresholds -->
## Optimal Confidence Threshold by Species

The table below shows the confidence threshold that produced the highest F1 score for each species. This can help identify the optimal balance between recall and precision for operational use.

{Best_Threshold_Table}
<!-- END:Optimal_Thresholds -->

<!-- BEGIN:Prediction_Distribution -->
## Prediction Distribution by Species

The outputs below represent simplified confusion matrices highlighting the distribution of predictions relevant to each target (test) species across multiple confidence thresholds. Refer to the full confusion matrix for every combination of True Positive, False Positive, and False Negative.

{Prediction_Distribution_Tables}
<!-- END:Prediction_Distribution -->

<!-- BEGIN:Full_Confusion_Matrices -->
## Full Confusion Matrices

The outputs below represent the complete confusion matrix for each confidence threshold. Predictions below the specified threshold are excluded.

{Full_Confusion_Matrices}
<!-- END:Full_Confusion_Matrices -->
