"""Optional taxonomic lookup of the evaluated species using the ALA galah package."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# Output column -> galah column. Columns galah does not return are left blank.
TAXONOMY_COLUMNS = {
    "Matched scientific name (ALA)": "scientificName",
    "Taxonomic Rank": "rank",
    "Kingdom": "kingdom",
    "Phylum": "phylum",
    "Order": "order",
    "Family": "family",
    "Genus": "genus",
    "Common name": "vernacularName",
}


def _lookup_one(label: str, search_taxa) -> dict:
    """Best match for one label, or an empty dict if nothing valid was found."""
    try:
        result = search_taxa(taxa=[label])
    except Exception as exc:  # network, API or parsing problems must never stop the report
        logger.warning("Taxonomy lookup failed for '%s': %s", label, exc)
        return {}
    if result is None or result.empty or "scientificName" not in result.columns:
        logger.info("No taxonomic match for '%s'", label)
        return {}
    row = result.iloc[0]
    if "issues" in result.columns and str(row["issues"]).lower() not in ("noissue", "nan", ""):
        logger.info("Taxonomic match for '%s' has issue '%s'", label, row["issues"])
    return row.to_dict()


def build_taxonomy_table(species_labels: List[str], image_counts=None, search_taxa=None) -> pd.DataFrame:
    """One row per label. Unmatched labels keep their name and have blank lookup values."""
    image_counts = image_counts or {}
    if search_taxa is None:
        try:
            import galah
        except ImportError:
            logger.warning("The 'galah' package is not installed (pip install galah); skipping taxonomy lookup")
            return _blank_table(species_labels, image_counts)
        search_taxa = galah.search_taxa

    rows = []
    for label in species_labels:
        match = _lookup_one(label, search_taxa)
        row = {"Species (or label)": label, "Number of images": image_counts.get(label, 0)}
        for column, source in TAXONOMY_COLUMNS.items():
            value = match.get(source, "")
            row[column] = "" if pd.isna(value) else str(value)
        rows.append(row)
    table = pd.DataFrame(rows)
    matched = int((table["Matched scientific name (ALA)"] != "").sum())
    logger.info("Taxonomy lookup matched %d of %d labels", matched, len(table))
    return table


RANK_ORDER = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def _lineage(match: dict) -> dict:
    """{rank: lowercase name} for a galah match, including the matched taxon at its own rank."""
    lineage = {}
    for rank in RANK_ORDER:
        value = match.get(rank)
        if value is not None and not pd.isna(value) and str(value).strip():
            lineage[rank] = str(value).strip().lower()
    rank = str(match.get("rank", "")).strip().lower()
    name = match.get("scientificName")
    if rank in RANK_ORDER and name is not None and not pd.isna(name):
        lineage[rank] = str(name).strip().lower()
    return lineage


def build_match_map(true_labels, predicted_labels, search_taxa=None) -> dict:
    """Which predicted labels count as correct for each higher-rank true label.

    A true label with an ALA rank above species (e.g. order 'Lagomorpha') accepts any predicted
    label whose ALA classification falls inside it (e.g. 'Oryctolagus cuniculus'). Labels are
    lowercase, as in the merged table. Returns {true_label: {accepted predicted labels}},
    containing only true labels that accept something beyond themselves.
    """
    if search_taxa is None:
        try:
            import galah
        except ImportError:
            logger.warning("The 'galah' package is not installed (pip install galah); using exact taxonomy matches only")
            return {}
        search_taxa = galah.search_taxa

    higher = {}
    for label in true_labels:
        match = _lookup_one(label, search_taxa)
        rank = str(match.get("rank", "")).strip().lower()
        lineage = _lineage(match)
        if rank in RANK_ORDER and rank != "species" and rank in lineage:
            higher[label] = (rank, lineage[rank])
    if not higher:
        logger.info("No higher-ranking taxonomy labels found in the test data; exact matching applies")
        return {}

    predicted = {}
    for label in predicted_labels:
        if label in ("blank", "unclassified"):
            continue
        match = _lookup_one(label, search_taxa)
        predicted[label] = (str(match.get("rank", "")).strip().lower(), _lineage(match))

    match_map = {}
    for label, (rank, name) in higher.items():
        accepted = {
            pred for pred, (pred_rank, lineage) in predicted.items()
            if pred != label and lineage.get(rank) == name
            and pred_rank in RANK_ORDER and RANK_ORDER.index(pred_rank) > RANK_ORDER.index(rank)
        }
        if accepted:
            match_map[label] = accepted
            logger.info("'%s' (%s) will also accept predictions: %s", label, rank, sorted(accepted))
    return match_map


def _blank_table(species_labels: List[str], image_counts: dict) -> pd.DataFrame:
    table = pd.DataFrame({"Species (or label)": species_labels, "Number of images": [image_counts.get(s, 0) for s in species_labels]})
    for column in TAXONOMY_COLUMNS:
        table[column] = ""
    return table
