from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import STANDARD_COLUMNS


def empty_standard_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the project standard schema."""
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame contains standard columns in a stable order."""
    result = df.copy()
    for column in STANDARD_COLUMNS:
        if column not in result.columns:
            result[column] = None

    extra_columns = [column for column in result.columns if column not in STANDARD_COLUMNS]
    return result[STANDARD_COLUMNS + extra_columns]


def is_missing(value: Any) -> bool:
    """Return True when a value is None, NaN, or an empty string."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def none_if_missing(value: Any) -> Any:
    """Convert empty pandas values to None while preserving useful values."""
    return None if is_missing(value) else value


def row_to_evidence_text(row: pd.Series) -> str | None:
    """Serialize a source row into compact evidence text."""
    parts = []
    for key, value in row.items():
        if is_missing(value):
            continue
        parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else None


def safe_file_name(file_name: str) -> str:
    """Make an uploaded file name safe for local storage."""
    name = Path(file_name).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe_name or "uploaded_file"
