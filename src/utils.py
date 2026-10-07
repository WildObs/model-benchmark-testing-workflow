"""Small shared helpers."""

import pandas as pd


def format_species_name(name):
    """'felis catus' -> 'Felis catus' (genus capitalised, remainder lower case)."""
    if pd.isna(name):
        return name
    parts = str(name).strip().split()
    if not parts:
        return name
    return " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])


def threshold_label(threshold: float) -> str:
    """Label such as '>=0.5'; uses two decimals only when needed (e.g. '>=0.05')."""
    decimals = 1 if abs(threshold * 10 - round(threshold * 10)) < 1e-9 else 2
    return f">={threshold:.{decimals}f}"
