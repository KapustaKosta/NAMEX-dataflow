"""Tesseract OCR engine implementation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.extract_ocr import (
    extract_ocr_pages as extract_ocr_pages_direct,
    is_tesseract_available,
    TESSERACT_INSTALL_MESSAGE,
)
from src.ocr_engines.base import OcrEngine, OcrPageResult, OcrSettings

if TYPE_CHECKING:
    import pandas as pd


class TesseractEngine(OcrEngine):
    """Tesseract OCR engine (local, open-source)."""

    name = "tesseract"

    def check_availability(self) -> tuple[bool, str]:
        """Check if Tesseract is installed and available."""
        if is_tesseract_available():
            return True, "Tesseract is available"
        return False, TESSERACT_INSTALL_MESSAGE

    def recognize_pdf(
        self,
        pdf_path: str,
        settings: OcrSettings,
    ) -> list[OcrPageResult]:
        """Recognize text from PDF using Tesseract.

        Args:
            pdf_path: Path to PDF file
            settings: OCR configuration

        Returns:
            List of OcrPageResult, one per page

        Raises:
            RuntimeError: If Tesseract unavailable or OCR fails
            ValueError: If PDF path invalid
        """
        available, msg = self.check_availability()
        if not available:
            raise RuntimeError(f"Tesseract OCR unavailable: {msg}")

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise ValueError(f"PDF file not found: {pdf_path}")

        pages = settings.pages if settings.pages else []

        ocr_df = extract_ocr_pages_direct(
            str(pdf_path_obj),
            lang=settings.lang,
            dpi=settings.dpi,
            pages=pages,
        )

        results: list[OcrPageResult] = []
        if not ocr_df.empty:
            for _, row in ocr_df.iterrows():
                page = int(row.get("page", 0))
                text = str(row.get("evidence_text", ""))

                result = OcrPageResult(
                    page=page,
                    text=text,
                    engine=self.name,
                    language=settings.lang,
                    confidence=0.95,
                    raw_data={
                        "source_file": row.get("source_file"),
                        "extraction_method": row.get("extraction_method"),
                        "extraction_level": row.get("extraction_level"),
                    },
                )
                results.append(result)

        return results
