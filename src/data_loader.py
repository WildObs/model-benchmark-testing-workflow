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


OBSERVATION_LEVELS = ("event", "media")


def available_observation_levels(observations: pd.DataFrame) -> set:
    """Observation levels present in an observations table (legacy exports are event-only)."""
    if "observationLevel" not in observations.columns:
        return {"event"}
    levels = observations["observationLevel"].dropna().astype(str).str.strip().str.lower()
    return set(levels[levels != ""]) or {"event"}


def read_observation_levels(folder: Path) -> set:
    """Read just the observationLevel column of an export's observations.csv."""
    path = Path(folder) / "observations.csv"
    header = pd.read_csv(path, nrows=0).columns
    if "observationLevel" not in header:
        return {"event"}
    return available_observation_levels(pd.read_csv(path, usecols=["observationLevel"], dtype=str))


def resolve_observation_level(requested: str, export_folders: List[Path]) -> str:
    """Return the observation level to use for every export in a set.

    Media-level analysis is only used when requested AND every export contains media-level
    observations; otherwise the set reverts to event-level so reports stay comparable.
    """
    requested = requested.lower()
    if requested == "event":
        return "event"
    without_media = [
        Path(f).name for f in export_folders if "media" not in read_observation_levels(f)
    ]
    if without_media:
        logger.warning(
            "observation_level='media' requested but %d of %d export(s) have no media-level observations "
            "(%s); reverting to 'event' for ALL exports so the reports are comparable",
            len(without_media), len(export_folders), ", ".join(without_media),
        )
        return "event"
    return "media"


def select_observation_level(observations: pd.DataFrame, level: str) -> pd.DataFrame:
    """Keep only event-level or only media-level rows (never both)."""
    if "observationLevel" not in observations.columns:
        if level == "media":
            raise ValueError("observations.csv has no observationLevel column, so media-level observations are unavailable")
        return observations
    levels = observations["observationLevel"].fillna("").astype(str).str.strip().str.lower()
    # Rows with no level are treated as event-level (legacy exports)
    if level == "event":
        selected = observations[levels.isin(["event", ""])]
    else:
        selected = observations[levels == "media"]
    logger.info("Using %s-level observations: %d of %d rows", level, len(selected), len(observations))
    return selected


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
    if "mediaID" not in observations.columns:
        observations["mediaID"] = pd.NA
    observations = observations[[
        "eventID", "mediaID", "deploymentID", "observationType",
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
    observations: pd.DataFrame, media: pd.DataFrame, deployments: pd.DataFrame, observation_level: str = "event"
) -> pd.DataFrame:
    """Join media -> observations -> deployments, adding true/pred species.

    Event-level observations are joined via the sequenceID in mediaComments; media-level
    observations are joined directly on mediaID.
    """
    media = media[["mediaID", "deploymentID", "mediaComments"]].copy()
    media["sequenceID"] = media["mediaComments"].apply(extract_sequence)
    no_sequence = int(media["sequenceID"].isna().sum())
    if observation_level == "event" and no_sequence:
        logger.warning("%d media files have no sequenceID in mediaComments; they will be treated as Blank predictions", no_sequence)

    deployments = deployments[["deploymentID", "deploymentTags"]]

    if observation_level == "media":
        # deploymentID comes from media.csv, so images with no media-level observation keep their true label
        merged = pd.merge(media, observations.drop(columns=["deploymentID"]), on="mediaID", how="left")
    else:
        merged = pd.merge(
            media.drop(columns=["deploymentID"]),
            observations.drop(columns=["mediaID"]),
            left_on="sequenceID", right_on="eventID", how="left",
        )
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


def load_benchmark_table(folder: Path, observation_level: str = "event"):
    """Load and merge everything needed for metrics.

    observation_level is "event" or "media"; only that level's observations are used.
    Returns (merged, model_name, species_display_list, image_counts).
    """
    if observation_level not in OBSERVATION_LEVELS:
        raise ValueError(f"observation_level must be one of {OBSERVATION_LEVELS}, got {observation_level!r}")
    observations, media, deployments = load_camtrap_data(folder)
    observations = select_observation_level(observations, observation_level)
    observations = prepare_observations(observations)
    model_name = get_model_name(observations)
    species_display = get_species_display_list(deployments)
    image_counts = count_images_per_species(media, deployments)
    merged = build_merged_table(observations, media, deployments, observation_level)
    logger.info(
        "Model: %s | %d images | %d species/labels in deployment tags",
        model_name, len(merged), len(species_display),
    )
    return merged, model_name, species_display, image_counts
