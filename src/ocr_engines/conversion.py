"""Conversion utilities for OCR engine results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.ocr_engines.base import OcrPageResult

if TYPE_CHECKING:
    pass


def ocr_page_results_to_dataframe(
    results: list[OcrPageResult],
    source_file: str,
) -> pd.DataFrame:
    """Convert OcrPageResult list to DataFrame compatible with pipeline.

    Args:
        results: List of OcrPageResult from OCR engine
        source_file: Source PDF filename

    Returns:
        DataFrame with columns: source_file, source_type, page, row_id, evidence_text,
                               extraction_method, extraction_level, section_name
    """
    rows = []

    for idx, result in enumerate(results):
        row = {
            "source_file": source_file,
            "source_type": "pdf",
            "page": result.page,
            "row_id": idx + 1,
            "evidence_text": result.text,
            "extraction_method": f"{result.engine}_ocr",
            "extraction_level": "raw_ocr",
            "section_name": f"ocr_page_text",
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_file",
                "source_type",
                "page",
                "row_id",
                "evidence_text",
                "extraction_method",
                "extraction_level",
                "section_name",
            ]
        )

    return pd.DataFrame(rows)
