<!-- description: Per-species comparison of F1 scores at each model's optimal confidence threshold. -->

# AI Computer Vision Model Evaluation Comparison Report

## Introduction
This report aggregates multiple individual model evaluation reports and generates a comparison by species.

## Test details

**Computer Vision (CV) models tested:** {Model_Names}

**Timestamps of input evaluation reports:** {Report_Timestamp}

**Source location of test images:** {Data_Source_Location}

## Model performance summary

Average recall, precision, and F1 score across all evaluated species at each model's optimal confidence threshold.

{Model_Performance_Summary_Table}

## F1 score by species and model

F1 scores are taken from the **Optimal Confidence Threshold by Species** table in each evaluation report. Missing species scores are shown as N/A. A best-performing model is shown only when one model has the highest F1 score; ties are shown as N/A.

{Comparison_Summary_Table}

## Animal Recall

Animal recall measures how well each model detects the presence of animals, regardless of the species assigned. It is calculated over images of animals (images whose true label is a species). Images with no observation are counted as Blank.

- **Detector Model Animal Recall:** the proportion of animal images that were *not* marked as 'Blank' (at least one animal was detected above the minimum confidence threshold).
- **Classifier Model Animal Recall:** of the animal images that were detected, the proportion that were *not* marked as 'Unclassified' (the species could be predicted above the minimum confidence threshold).
- **Overall Animal Recall:** the proportion of all animal images that were neither marked as 'Blank' nor 'Unclassified'.

{Animal_Recall_Table}