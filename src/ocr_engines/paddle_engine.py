"""PaddleOCR engine implementation (local, GPU-capable)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.ocr_engines.base import OcrEngine, OcrPageResult, OcrSettings

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


def _get_paddle_ocr(lang: str, use_gpu: bool) -> PaddleOCR:
    """Lazy import and instantiate PaddleOCR."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError(
            "PaddleOCR not installed. Install with:\n"
            "  pip install paddleocr paddlepaddle\n"
            "For GPU support:\n"
            "  pip install paddlepaddle-gpu"
        )

    lang_code = _normalize_paddle_lang(lang)
    return PaddleOCR(
        use_angle_cls=True,
        use_gpu=use_gpu,
        lang=lang_code,
    )


def _normalize_paddle_lang(lang: str) -> str:
    """Convert language code to PaddleOCR format.

    Args:
        lang: Language code like 'rus+eng', 'rus', 'eng'

    Returns:
        PaddleOCR format like 'ch', 'en', 'ru'
    """
    paddle_langs = {
        "rus": "ru",
        "ru": "ru",
        "russian": "ru",
        "eng": "en",
        "en": "en",
        "english": "en",
        "ch": "ch",
        "chi": "ch",
        "chinese": "ch",
    }

    if "+" in lang:
        parts = [p.strip().lower() for p in lang.split("+")]
        result = []
        for part in parts:
            mapped = paddle_langs.get(part, part.lower()[:2])
            if mapped:
                result.append(mapped)
        return ",".join(result) if result else "en"

    lang_lower = lang.strip().lower()
    return paddle_langs.get(lang_lower, lang_lower[:2])


class PaddleOCREngine(OcrEngine):
    """PaddleOCR engine (local, GPU-capable)."""

    name = "paddleocr"

    def __init__(self) -> None:
        self._ocr_instance = None
        self._last_config = None

    def check_availability(self) -> tuple[bool, str]:
        """Check if PaddleOCR is installed."""
        try:
            import paddleocr  # noqa: F401
            import paddlepaddle  # noqa: F401

            return True, "PaddleOCR is available"
        except ImportError as e:
            return (
                False,
                f"PaddleOCR not installed: {e}\n"
                "Install with: pip install paddleocr paddlepaddle",
            )

    def recognize_pdf(
        self,
        pdf_path: str,
        settings: OcrSettings,
    ) -> list[OcrPageResult]:
        """Recognize text from PDF using PaddleOCR.

        Args:
            pdf_path: Path to PDF file
            settings: OCR configuration

        Returns:
            List of OcrPageResult, one per page

        Raises:
            RuntimeError: If PaddleOCR unavailable or OCR fails
            ValueError: If PDF path invalid
        """
        available, msg = self.check_availability()
        if not available:
            raise RuntimeError(f"PaddleOCR unavailable: {msg}")

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise ValueError(f"PDF file not found: {pdf_path}")

        try:
            import fitz
        except ImportError:
            raise RuntimeError(
                "fitz (PyMuPDF) required for PDF processing: pip install pymupdf"
            )

        paddle_ocr = _get_paddle_ocr(settings.lang, settings.use_gpu)

        results: list[OcrPageResult] = []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                doc = fitz.open(str(pdf_path_obj))
                pages = settings.pages if settings.pages else list(range(1, doc.page_count + 1))

                for page_num in pages:
                    if page_num < 1 or page_num > doc.page_count:
                        continue

                    page = doc[page_num - 1]
                    pix = page.get_pixmap(matrix=fitz.Matrix(settings.dpi / 72, settings.dpi / 72))
                    image_path = Path(tmpdir) / f"page_{page_num}.png"
                    pix.save(str(image_path))

                    ocr_result = paddle_ocr.ocr(str(image_path), cls=True)

                    text_lines = []
                    if ocr_result:
                        for line in ocr_result[0] if ocr_result[0] else []:
                            if len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                                confidence = float(line[1][1]) if isinstance(line[1], tuple) else 0.9
                                text_lines.append((text, confidence))

                    full_text = "\n".join([text for text, _ in text_lines])
                    avg_confidence = (
                        sum(conf for _, conf in text_lines) / len(text_lines)
                        if text_lines
                        else 0.0
                    )

                    result = OcrPageResult(
                        page=page_num,
                        text=full_text,
                        engine=self.name,
                        language=settings.lang,
                        confidence=avg_confidence,
                        raw_data={
                            "source_file": str(pdf_path_obj),
                            "extraction_method": "paddleocr",
                            "extraction_level": "raw_ocr",
                            "dpi": settings.dpi,
                        },
                    )
                    results.append(result)

                doc.close()

        except Exception as e:
            raise RuntimeError(f"PaddleOCR processing failed: {e}")

        return results
