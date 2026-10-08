"""Load a Camtrap DP export and merge it into one table of truth vs prediction."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .utils import format_species_name

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "observations.csv": [
        "eventID", "deploymentID", "observationType", "scientificName",
        "classificationProbability", "classifiedBy",
    ],
    "media.csv": ["mediaID", "deploymentID", "mediaComments"],
    "deployments.csv": ["deploymentID", "deploymentTags"],
}


def _read_csv(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    missing = [c for c in REQUIRED_COLUMNS[filename] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    logger.debug("Read %s: %d rows", path, len(df))
    return df


def load_camtrap_data(folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read observations, media and deployments from a Camtrap DP folder."""
    folder = Path(folder)
    if not folder.is_dir():
        siblings = sorted(p.name for p in folder.parent.iterdir() if p.is_dir()) if folder.parent.is_dir() else []
        raise FileNotFoundError(
            f"Camtrap DP folder not found: {folder}. Folders available in {folder.parent}: {siblings}"
        )
    logger.info("Loading Camtrap DP data from %s", folder)
    observations = _read_csv(folder, "observations.csv")
    media = _read_csv(folder, "media.csv")
    deployments = _read_csv(folder, "deployments.csv")
    logger.info(
        "Loaded %d observations, %d media files, %d deployments",
        len(observations), len(media), len(deployments),
    )
    return observations, media, deployments


def extract_sequence(comment) -> Optional[str]:
    """Pull the sequenceID out of a WIMP mediaComments string."""
    if pd.isna(comment):
        return None
    match = re.search(r"sequenceID:([^\s]+)", str(comment))
    return match.group(1) if match else None


def prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.copy()
    observations["classificationProbability"] = (
        pd.to_numeric(
            observations["classificationProbability"].fillna("").astype(str).str.strip(),
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )
    observations = observations[[
        "eventID", "deploymentID", "observationType",
        "scientificName", "classificationProbability", "classifiedBy",
    ]].rename(columns={"classificationProbability": "confidence"})

    # Blank/unclassified observations carry no species name, so label them explicitly
    for observation_type, label in (("blank", "Blank"), ("unclassified", "Unclassified")):
        mask = observations["observationType"].str.lower().eq(observation_type)
        observations.loc[mask, "scientificName"] = observations.loc[mask, "scientificName"].fillna(label)
    return observations


def get_model_name(observations: pd.DataFrame) -> str:
    names = observations["classifiedBy"].dropna().astype(str).str.strip()
    names = names[names != ""]
    if names.empty:
        logger.warning("No classifiedBy value found; model name set to 'Unknown Model'")
        return "Unknown Model"
    if names.nunique() > 1:
        logger.warning("Multiple models found in classifiedBy %s; using '%s'", sorted(names.unique()), names.iloc[0])
    return names.iloc[0]


def build_merged_table(
    observations: pd.DataFrame, media: pd.DataFrame, deployments: pd.DataFrame
) -> pd.DataFrame:
    """Join media -> observations (via sequenceID) -> deployments, adding true/pred species."""
    media = media[["mediaID", "mediaComments"]].copy()
    media["sequenceID"] = media["mediaComments"].apply(extract_sequence)
    no_sequence = int(media["sequenceID"].isna().sum())
    if no_sequence:
        logger.warning("%d media files have no sequenceID in mediaComments; they will be treated as Blank predictions", no_sequence)

    deployments = deployments[["deploymentID", "deploymentTags"]]

    merged = pd.merge(media, observations, left_on="sequenceID", right_on="eventID", how="left")
    merged = pd.merge(merged, deployments, on="deploymentID", how="left")

    merged["true_species"] = _normalise(merged["deploymentTags"])
    merged["pred_species"] = _normalise(merged["scientificName"])

    untagged = int((merged["deploymentTags"].isna() | merged["deploymentTags"].str.strip().eq("")).sum())
    if untagged:
        logger.warning("%d images belong to deployments with no tag; their true species is set to 'blank'", untagged)
    unmatched = int(merged["eventID"].isna().sum())
    if unmatched:
        logger.warning("%d images have no matching observation; their prediction is set to 'blank'", unmatched)
    return merged


def _normalise(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip().str.lower()
    return cleaned.where(cleaned != "", "blank")


def get_species_display_list(deployments: pd.DataFrame) -> List[str]:
    tags = deployments["deploymentTags"].dropna().astype(str).str.strip().str.lower().unique()
    return [format_species_name(s) for s in sorted(tags)]


def count_images_per_species(media: pd.DataFrame, deployments: pd.DataFrame) -> Dict[str, int]:
    """Number of images (media rows) per species, using each image's deployment tag."""
    tags = deployments[["deploymentID", "deploymentTags"]].copy()
    tags["species"] = tags["deploymentTags"].fillna("").astype(str).str.strip().str.lower()
    images = media[["mediaID", "deploymentID"]].drop_duplicates("mediaID").merge(
        tags[["deploymentID", "species"]], on="deploymentID", how="left"
    )
    counts = images.loc[images["species"].fillna("") != "", "species"].value_counts()
    return {format_species_name(name): int(n) for name, n in counts.items()}


def load_benchmark_table(folder: Path):
    """Load and merge everything needed for metrics.

    Returns (merged, model_name, species_display_list, image_counts).
    """
    observations, media, deployments = load_camtrap_data(folder)
    observations = prepare_observations(observations)
    model_name = get_model_name(observations)
    species_display = get_species_display_list(deployments)
    image_counts = count_images_per_species(media, deployments)
    merged = build_merged_table(observations, media, deployments)
    logger.info(
        "Model: %s | %d images | %d species/labels in deployment tags",
        model_name, len(merged), len(species_display),
    )
    return merged, model_name, species_display, image_counts
