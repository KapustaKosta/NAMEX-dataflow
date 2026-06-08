"""Base classes and interfaces for OCR engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OcrPageResult:
    """Result of OCR processing for a single page."""

    page: int
    text: str
    words: list[dict] | None = None
    lines: list[dict] | None = None
    engine: str = "unknown"
    confidence: float | None = None
    language: str = "ru+en"
    processing_time_ms: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.page, int) or self.page < 1:
            raise ValueError(f"Invalid page number: {self.page}")
        if not isinstance(self.text, str):
            raise ValueError(f"Text must be string, got {type(self.text)}")


@dataclass
class OcrSettings:
    """Configuration for OCR processing."""

    lang: str = "rus+eng"
    dpi: int = 300
    pages: list[int] | None = None
    use_gpu: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dpi < 100:
            raise ValueError(f"DPI must be >= 100, got {self.dpi}")
        if self.pages is None:
            self.pages = []
        if not isinstance(self.pages, list):
            raise ValueError(f"Pages must be list, got {type(self.pages)}")


class OcrEngine:
    """Base interface for OCR engines."""

    name: str = "unknown"

    def recognize_pdf(
        self,
        pdf_path: str,
        settings: OcrSettings,
    ) -> list[OcrPageResult]:
        """Recognize text from PDF.

        Args:
            pdf_path: Path to PDF file
            settings: OCR configuration

        Returns:
            List of page results, one per page

        Raises:
            ValueError: If PDF path invalid or settings invalid
            RuntimeError: If OCR processing fails
        """
        raise NotImplementedError("Subclasses must implement recognize_pdf")

    def check_availability(self) -> tuple[bool, str]:
        """Check if OCR engine is available.

        Returns:
            (is_available, message) - True if ready, False if missing dependencies/config
        """
        raise NotImplementedError("Subclasses must implement check_availability")
