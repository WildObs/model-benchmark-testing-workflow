"""Build the placeholder values used by report templates."""

from typing import Dict

import pandas as pd

from .config import BenchmarkConfig
from .metrics import BenchmarkMetrics
from .templating import Placeholder
from .utils import threshold_label

TABLE_CLASS = "styled-table"


def _table(df: pd.DataFrame, border: int = 0) -> str:
    return df.to_html(index=False, classes=TABLE_CLASS, border=border)


def _scrollable(df: pd.DataFrame) -> str:
    return f'<div style="overflow-x:auto;">{_table(df)}</div>'


def build_placeholders(
    config: BenchmarkConfig,
    model_name: str,
    species_display: list,
    metrics: BenchmarkMetrics,
    image_count: int,
    timestamp: str,
) -> Dict[str, Placeholder]:
    """All placeholders available to templates. Keep in sync with templates/README.md."""
    species_md = "\n".join(f"- *{s}*" for s in species_display)

    prediction_distribution = "\n".join(
        f"<h3>True species (or label): <i>{species}</i></h3>{_scrollable(matrix)}"
        for species, matrix in metrics.single_row_matrices.items()
    )
    full_matrices = "\n".join(
        f"<h3>Confusion Matrix - Confidence Threshold: {threshold}</h3>{_scrollable(matrix)}<br>"
        for threshold, matrix in metrics.full_confusion_matrices.items()
    )

    return {
        "Report_Timestamp": Placeholder(timestamp),
        "Model_Name": Placeholder(model_name),
        "Data_Source_Location": Placeholder(config.data_source_location.strip() or "Not specified"),
        "Data_Collation_Process": Placeholder(config.data_collation_process.strip() or "Not specified"),
        "Confidence_Thresholds": Placeholder(", ".join(threshold_label(t)[2:] for t in config.thresholds)),
        "Species_List": Placeholder(species_md, "markdown"),
        "Species_Count": Placeholder(str(len(species_display))),
        "Image_Count": Placeholder(str(image_count)),
        "Results_Table": Placeholder(_table(metrics.results, border=1), "html"),
        "Best_Threshold_Table": Placeholder(_table(metrics.best_thresholds), "html"),
        "Prediction_Distribution_Tables": Placeholder(prediction_distribution, "html"),
        "Full_Confusion_Matrices": Placeholder(full_matrices, "html"),
    }
