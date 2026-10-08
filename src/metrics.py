"""Performance metrics: per-species recall/precision/F1 and confusion matrices."""

import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .utils import format_species_name, threshold_label

logger = logging.getLogger(__name__)

PRED_AXIS_LABEL = "predicted_species→\ntrue_species↓"
SINGLE_ROW_AXIS_LABEL = "predicted_species→\nconfidence_threshold↓"


@dataclass
class BenchmarkMetrics:
    results: pd.DataFrame
    best_thresholds: pd.DataFrame
    full_confusion_matrices: Dict[str, pd.DataFrame]
    single_row_matrices: Dict[str, pd.DataFrame]


def compute_metrics(merged: pd.DataFrame, thresholds: List[float]) -> BenchmarkMetrics:
    species_list = sorted(merged["true_species"].dropna().unique())
    logger.info("Computing metrics for %d species x %d thresholds", len(species_list), len(thresholds))

    results = compute_results_table(merged, species_list, thresholds)
    best = compute_best_thresholds(results)
    results = results.drop(columns="threshold_value")
    full = compute_full_confusion_matrices(merged, thresholds)
    single = compute_single_row_matrices(merged, species_list, thresholds)
    return BenchmarkMetrics(results, best, full, single)


def compute_results_table(merged: pd.DataFrame, species_list: List[str], thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for species in species_list:
        is_true = merged["true_species"] == species
        is_pred = merged["pred_species"] == species
        for threshold in thresholds:
            passes = merged["confidence"] >= threshold

            tp = int((is_true & is_pred & passes).sum())
            fn = int((is_true & (~is_pred | ~passes)).sum())
            fp = int((~is_true & is_pred & passes).sum())

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if tp + fn == 0:
                logger.warning("No test images for '%s'; metrics will be zero", species)
            logger.debug(
                "%s %s: TP=%d FN=%d FP=%d recall=%.4f precision=%.4f f1=%.4f",
                species, threshold_label(threshold), tp, fn, fp, recall, precision, f1,
            )
            rows.append({
                "species": format_species_name(species),
                "model_confidence": threshold_label(threshold),
                "threshold_value": threshold,
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1_score": round(f1, 4),
            })
    return pd.DataFrame(rows).sort_values(by=["species", "model_confidence"]).reset_index(drop=True)


def compute_best_thresholds(results: pd.DataFrame) -> pd.DataFrame:
    """Threshold with the highest F1 per species (lowest threshold wins ties)."""
    best = (
        results.sort_values(by=["species", "f1_score", "threshold_value"], ascending=[True, False, True])
        .groupby("species")
        .first()
        .reset_index()
    )
    return best[["species", "model_confidence", "recall", "precision", "f1_score"]]


def compute_full_confusion_matrices(merged: pd.DataFrame, thresholds: List[float]) -> Dict[str, pd.DataFrame]:
    matrices = {}
    for threshold in thresholds:
        subset = merged[merged["confidence"] >= threshold]
        matrix = pd.crosstab(subset["true_species"], subset["pred_species"])
        matrix = matrix.loc[:, matrix.sum().sort_values(ascending=False).index]
        matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]

        matrix.index = [format_species_name(x) for x in matrix.index]
        matrix.columns = [format_species_name(x) for x in matrix.columns]
        matrix.index.name = PRED_AXIS_LABEL
        matrices[threshold_label(threshold)] = matrix.reset_index()
    return matrices


def compute_single_row_matrices(
    merged: pd.DataFrame, species_list: List[str], thresholds: List[float]
) -> Dict[str, pd.DataFrame]:
    """For each true species, the distribution of predictions across thresholds."""
    matrices = {}
    for species in species_list:
        truth = merged[merged["true_species"] == species]
        rows = []
        for threshold in thresholds:
            counts = truth[truth["confidence"] >= threshold]["pred_species"].fillna("NaN").value_counts()
            row = counts.to_dict()
            row["confidence_threshold"] = threshold_label(threshold)
            rows.append(row)

        matrix = pd.DataFrame(rows).fillna(0)
        prediction_cols = [c for c in matrix.columns if c != "confidence_threshold"]
        matrix[prediction_cols] = matrix[prediction_cols].astype(int)

        ordered = matrix[prediction_cols].sum().sort_values(ascending=False).index.tolist()
        matrix = matrix[["confidence_threshold"] + ordered]
        matrix = matrix.rename(
            columns={c: format_species_name(c) for c in ordered}
            | {"confidence_threshold": SINGLE_ROW_AXIS_LABEL}
        )
        matrices[format_species_name(species)] = matrix
    return matrices
