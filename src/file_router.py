from __future__ import annotations

from pathlib import Path


def detect_file_type(file_name: str) -> str:
    """Detect a supported file type by file extension."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".xlsx":
        return "xlsx"
    if suffix == ".pdf":
        return "pdf"
    return "unknown"
