"""Performance metrics: per-species recall/precision/F1 and confusion matrices."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .utils import format_species_name, threshold_label

logger = logging.getLogger(__name__)

PRED_AXIS_LABEL = "predicted_species→\ntrue_species↓"
SINGLE_ROW_AXIS_LABEL = "predicted_species→\nconfidence_threshold↓"


# True labels (deployment tags) that are not animals, so they are excluded from animal recall
NON_ANIMAL_LABELS = {"blank", "human", "homo sapiens", "person", "people", "vehicle"}


@dataclass
class AnimalRecall:
    """How well a model detects the presence of animals, ignoring the species it assigned.

    Detector recall: animal images not marked Blank / all animal images.
    Classifier recall: animal images the detector passed on (not Blank) that were not
    marked Unclassified / animal images not marked Blank.
    Overall recall: animal images marked as neither Blank nor Unclassified / all animal images.
    """

    animal_images: int
    blank: int
    unclassified: int

    @property
    def detected(self) -> int:
        return self.animal_images - self.blank

    @property
    def classified(self) -> int:
        return self.detected - self.unclassified

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 4) if denominator > 0 else None

    @property
    def detector_recall(self) -> Optional[float]:
        return self._ratio(self.detected, self.animal_images)

    @property
    def classifier_recall(self) -> Optional[float]:
        return self._ratio(self.classified, self.detected)

    @property
    def overall_recall(self) -> Optional[float]:
        return self._ratio(self.classified, self.animal_images)


@dataclass
class BenchmarkMetrics:
    results: pd.DataFrame
    best_thresholds: pd.DataFrame
    full_confusion_matrices: Dict[str, pd.DataFrame]
    single_row_matrices: Dict[str, pd.DataFrame]
    animal_recall: Optional[AnimalRecall] = None


def compute_animal_recall(merged: pd.DataFrame) -> AnimalRecall:
    """Count animal images (true label is a species) that the model marked Blank or Unclassified.

    The merged table can hold several rows per image (one per observation), so each image is
    collapsed to a single outcome first: classified if any observation names a species, else
    Unclassified if any observation is Unclassified, else Blank. Images with no observation
    are predicted 'blank' by the merge, so they count as Blank.
    """
    animals = merged[~merged["true_species"].isin(NON_ANIMAL_LABELS)]
    pred = animals["pred_species"]
    outcome = pd.Series("blank", index=animals.index)
    outcome[pred == "unclassified"] = "unclassified"
    outcome[~pred.isin(["blank", "unclassified"])] = "classified"
    rank = outcome.map({"blank": 0, "unclassified": 1, "classified": 2})
    best = rank.groupby(animals["mediaID"].values).max()
    blank = int((best == 0).sum())
    unclassified = int((best == 1).sum())
    result = AnimalRecall(len(best), blank, unclassified)
    logger.info(
        "Animal recall: %d animal images, %d Blank, %d Unclassified (detector=%s classifier=%s overall=%s)",
        result.animal_images, blank, unclassified,
        result.detector_recall, result.classifier_recall, result.overall_recall,
    )
    return result


def compute_metrics(
    merged: pd.DataFrame, thresholds: List[float], match_map: Optional[Dict[str, set]] = None
) -> BenchmarkMetrics:
    """match_map ({true label: extra predicted labels counted as correct}) only affects recall,
    precision and F1; the confusion matrices always show the exact predictions."""
    species_list = sorted(merged["true_species"].dropna().unique())
    logger.info("Computing metrics for %d species x %d thresholds", len(species_list), len(thresholds))

    results = compute_results_table(merged, species_list, thresholds, match_map)
    best = compute_best_thresholds(results)
    results = results.drop(columns="threshold_value")
    full = compute_full_confusion_matrices(merged, thresholds)
    single = compute_single_row_matrices(merged, species_list, thresholds)
    return BenchmarkMetrics(results, best, full, single, compute_animal_recall(merged))


def filter_species(merged: pd.DataFrame, species: List[str]) -> pd.DataFrame:
    """Rows whose true species is in the given list (matched case-insensitively)."""
    wanted = {s.strip().lower() for s in species}
    return merged[merged["true_species"].isin(wanted)]


def restrict_metrics_to_species(metrics: BenchmarkMetrics, merged: pd.DataFrame, species: List[str]) -> BenchmarkMetrics:
    """Cut metrics computed on the full test set down to the chosen species.

    Per-species figures (including precision) keep the full test set as context; only the
    reported rows change. Animal recall is recomputed over the selected species' images.
    """
    wanted = {format_species_name(s.strip().lower()) for s in species}
    subset = filter_species(merged, species)
    full = {}
    for label, matrix in metrics.full_confusion_matrices.items():
        rows = matrix[matrix[PRED_AXIS_LABEL].isin(wanted)].reset_index(drop=True)
        counts = rows.drop(columns=PRED_AXIS_LABEL)
        # drop predictions that never occur for the selected species
        full[label] = rows[[PRED_AXIS_LABEL] + list(counts.columns[(counts != 0).any()])]
    return BenchmarkMetrics(
        results=metrics.results[metrics.results["species"].isin(wanted)].reset_index(drop=True),
        best_thresholds=metrics.best_thresholds[metrics.best_thresholds["species"].isin(wanted)].reset_index(drop=True),
        full_confusion_matrices=full,
        single_row_matrices={k: v for k, v in metrics.single_row_matrices.items() if k in wanted},
        animal_recall=compute_animal_recall(subset) if len(subset) else None,
    )


def compute_results_table(
    merged: pd.DataFrame,
    species_list: List[str],
    thresholds: List[float],
    match_map: Optional[Dict[str, set]] = None,
) -> pd.DataFrame:
    rows = []
    match_map = match_map or {}
    for species in species_list:
        is_true = merged["true_species"] == species
        accepted = match_map.get(species, set())
        is_pred = merged["pred_species"].isin({species} | accepted)
        # images truly of an accepted lower taxon are not false positives for the higher taxon
        not_covered = ~merged["true_species"].isin(accepted)
        for threshold in thresholds:
            passes = merged["confidence"] >= threshold

            tp = int((is_true & is_pred & passes).sum())
            fn = int((is_true & (~is_pred | ~passes)).sum())
            fp = int((~is_true & not_covered & is_pred & passes).sum())

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
