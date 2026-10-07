"""Optional taxonomic lookup of the evaluated species using the ALA galah package."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# Output column -> galah column. Columns galah does not return are left blank.
TAXONOMY_COLUMNS = {
    "Matched scientific name": "scientificName",
    "Authority": "scientificNameAuthorship",
    "Rank": "rank",
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


def build_taxonomy_table(species_labels: List[str], search_taxa=None) -> pd.DataFrame:
    """One row per label. Unmatched labels keep their name and have blank lookup values."""
    if search_taxa is None:
        try:
            import galah
        except ImportError:
            logger.warning("The 'galah' package is not installed (pip install galah); skipping taxonomy lookup")
            return _blank_table(species_labels)
        search_taxa = galah.search_taxa

    rows = []
    for label in species_labels:
        match = _lookup_one(label, search_taxa)
        row = {"Species (or label)": label}
        for column, source in TAXONOMY_COLUMNS.items():
            value = match.get(source, "")
            row[column] = "" if pd.isna(value) else str(value)
        rows.append(row)
    table = pd.DataFrame(rows)
    matched = int((table["Matched scientific name"] != "").sum())
    logger.info("Taxonomy lookup matched %d of %d labels", matched, len(table))
    return table


def _blank_table(species_labels: List[str]) -> pd.DataFrame:
    table = pd.DataFrame({"Species (or label)": species_labels})
    for column in TAXONOMY_COLUMNS:
        table[column] = ""
    return table
