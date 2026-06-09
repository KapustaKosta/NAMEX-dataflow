"""PaddleOCR engine implementation (local, GPU-capable)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.ocr_engines.base import OcrEngine, OcrPageResult, OcrSettings

def _normalize_paddle_lang(lang: str) -> str:
    """Convert language code to PaddleOCR format."""
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
        for part in parts:
            mapped = paddle_langs.get(part, part.lower()[:2])
            if mapped:
                return mapped
        return "en"

    lang_lower = lang.strip().lower()
    return paddle_langs.get(lang_lower, lang_lower[:2])


class PaddleOCREngine(OcrEngine):
    """PaddleOCR engine (local, GPU-capable)."""

    name = "paddleocr"

    def __init__(self) -> None:
        self._ocr_instance = None
        self._last_config = None

    def check_availability(self) -> tuple[bool, str]:
        """Check if PaddleOCR is installed using lightweight spec check."""
        import importlib.util
        
        has_paddle = importlib.util.find_spec("paddleocr") is not None
        has_paddlepaddle = importlib.util.find_spec("paddle") is not None or importlib.util.find_spec("paddlepaddle") is not None
        
        if has_paddle and has_paddlepaddle:
            return True, "PaddleOCR is available"
        
        missing = []
        if not has_paddle: missing.append("paddleocr")
        if not has_paddlepaddle: missing.append("paddlepaddle")
        
        return (
            False,
            f"PaddleOCR dependencies missing: {', '.join(missing)}\n"
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

        lang_code = _normalize_paddle_lang(settings.lang)
        results: list[OcrPageResult] = []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                doc = fitz.open(str(pdf_path_obj))
                pages = settings.pages if settings.pages else list(range(1, doc.page_count + 1))
                
                images_to_process = []

                for page_num in pages:
                    if page_num < 1 or page_num > doc.page_count:
                        continue

                    page = doc[page_num - 1]
                    pix = page.get_pixmap(matrix=fitz.Matrix(settings.dpi / 72, settings.dpi / 72))
                    image_path = Path(tmpdir) / f"page_{page_num}.png"
                    pix.save(str(image_path))
                    
                    images_to_process.append({
                        "page": page_num,
                        "path": str(image_path)
                    })

                doc.close()
                
                if not images_to_process:
                    return []
                
                # Execute Paddle worker
                worker_script = Path(__file__).parent / "paddle_worker.py"
                request_data = {
                    "lang": lang_code,
                    "use_gpu": settings.use_gpu,
                    "images": images_to_process
                }
                
                process = subprocess.run(
                    [sys.executable, str(worker_script)],
                    input=json.dumps(request_data),
                    text=True,
                    capture_output=True,
                )
                
                if process.returncode != 0:
                    try:
                        error_msg = json.loads(process.stdout).get("error", process.stderr)
                    except json.JSONDecodeError:
                        error_msg = process.stderr or process.stdout
                    raise RuntimeError(f"PaddleOCR worker failed: {error_msg}")
                
                try:
                    response_data = json.loads(process.stdout)
                except json.JSONDecodeError:
                    raise RuntimeError(f"Failed to parse PaddleOCR worker output: {process.stdout}")
                
                if not response_data.get("success"):
                    raise RuntimeError(f"PaddleOCR worker reported error: {response_data.get('error')}")
                
                for page_result in response_data.get("results", []):
                    page_num = page_result["page"]
                    text_lines = page_result["lines"]
                    
                    full_text = "\n".join([line["text"] for line in text_lines])
                    avg_confidence = (
                        sum(line["confidence"] for line in text_lines) / len(text_lines)
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
                            "extraction_method": "paddleocr_ocr",
                            "extraction_level": "raw_ocr",
                            "dpi": settings.dpi,
                        },
                    )
                    results.append(result)

        except Exception as e:
            if isinstance(e, RuntimeError) and "PaddleOCR" in str(e):
                raise
            raise RuntimeError(f"PaddleOCR processing failed: {e}")

        return results
