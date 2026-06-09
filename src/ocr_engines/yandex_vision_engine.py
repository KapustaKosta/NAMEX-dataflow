"""Yandex Vision OCR engine implementation (cloud-based)."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.ocr_engines.base import OcrEngine, OcrPageResult, OcrSettings

if TYPE_CHECKING:
    import requests


def _get_yandex_credentials() -> dict[str, str]:
    """Get Yandex Vision API credentials from environment or Streamlit secrets."""
    creds = {
        "iam_token": os.getenv("YANDEX_IAM_TOKEN"),
        "folder_id": os.getenv("YANDEX_FOLDER_ID"),
        "api_key": os.getenv("YANDEX_API_KEY"),
    }

    try:
        import streamlit as st

        if not creds["iam_token"]:
            creds["iam_token"] = st.secrets.get("YANDEX_IAM_TOKEN")
        if not creds["folder_id"]:
            creds["folder_id"] = st.secrets.get("YANDEX_FOLDER_ID")
        if not creds["api_key"]:
            creds["api_key"] = st.secrets.get("YANDEX_API_KEY")
    except (ImportError, AttributeError, KeyError):
        pass

    return {k: v for k, v in creds.items() if v}


class YandexVisionEngine(OcrEngine):
    """Yandex Vision OCR engine (cloud-based)."""

    name = "yandex_vision"
    endpoint = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    def check_availability(self) -> tuple[bool, str]:
        """Check if Yandex Vision is configured and dependencies are met."""
        import importlib.util
        if importlib.util.find_spec("requests") is None:
            return False, "Library 'requests' not installed: pip install requests"

        creds = _get_yandex_credentials()

        if creds.get("api_key"):
            return True, "Yandex Vision API key configured"

        if creds.get("iam_token") and creds.get("folder_id"):
            return True, "Yandex Vision IAM token and folder ID configured"

        return (
            False,
            "Yandex Vision credentials missing. Provide either YANDEX_API_KEY "
            "or both YANDEX_IAM_TOKEN and YANDEX_FOLDER_ID.",
        )

    def recognize_pdf(
        self,
        pdf_path: str,
        settings: OcrSettings,
    ) -> list[OcrPageResult]:
        """Recognize text from PDF using Yandex Vision API.

        Args:
            pdf_path: Path to PDF file
            settings: OCR configuration

        Returns:
            List of OcrPageResult, one per page

        Raises:
            RuntimeError: If API unavailable or request fails
            ValueError: If PDF path invalid
        """
        available, msg = self.check_availability()
        if not available:
            raise RuntimeError(f"Yandex Vision unavailable: {msg}")

        try:
            import requests
        except ImportError:
            raise RuntimeError("requests library required: pip install requests")

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise ValueError(f"PDF file not found: {pdf_path}")

        try:
            import fitz
        except ImportError:
            raise RuntimeError(
                "fitz (PyMuPDF) required for PDF processing: pip install pymupdf"
            )

        creds = _get_yandex_credentials()
        api_key = creds.get("api_key")
        iam_token = creds.get("iam_token")
        folder_id = creds.get("folder_id")

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

                    with open(str(image_path), "rb") as f:
                        image_data = f.read()

                    image_base64 = base64.b64encode(image_data).decode("utf-8")

                    request_body = {
                        "mimeType": "image/png",
                        "languageCodes": self._parse_languages(settings.lang),
                        "model": "page",
                        "content": image_base64,
                    }

                    headers = {"Content-Type": "application/json"}
                    if api_key:
                        headers["Authorization"] = f"Api-Key {api_key}"
                    else:
                        headers["Authorization"] = f"Bearer {iam_token}"
                        headers["x-folder-id"] = folder_id

                    try:
                        response = requests.post(
                            self.endpoint,
                            json=request_body,
                            headers=headers,
                            timeout=30,
                        )
                        response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        raise RuntimeError(f"Yandex Vision API request failed: {e}")

                    api_result = response.json()

                    # Support both nested {"result": {"textAnnotation": ...}} and flat {"textAnnotation": ...}
                    result_data = api_result.get("result", api_result)
                    text_annotation = result_data.get("textAnnotation", {})
                    
                    full_text = text_annotation.get("fullText", "")
                    
                    if not full_text:
                        full_text_lines = []
                        for block in text_annotation.get("blocks", []):
                            for line in block.get("lines", []):
                                line_text = line.get("text", "")
                                if line_text:
                                    full_text_lines.append(line_text)
                                else:
                                    line_words = []
                                    for word in line.get("words", []):
                                        text = word.get("text", "")
                                        if text:
                                            line_words.append(text)
                                    if line_words:
                                        full_text_lines.append(" ".join(line_words))
                        full_text = "\n".join(full_text_lines)

                    if not full_text.strip():
                        debug_info = {
                            "status": response.status_code,
                            "top_level_keys": list(api_result.keys()),
                            "has_textAnnotation": "textAnnotation" in result_data,
                            "has_fullText": "fullText" in text_annotation,
                            "blocks_count": len(text_annotation.get("blocks", [])),
                        }
                        raise RuntimeError(f"Yandex Vision returned no text. Response shape: {debug_info}")

                    # Extract confidence
                    text_lines = []
                    for block in text_annotation.get("blocks", []):
                        for line in block.get("lines", []):
                            for word in line.get("words", []):
                                text = word.get("text", "")
                                confidence = word.get("confidence", 0.0)
                                if text:
                                    text_lines.append((text, confidence))
                                    
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
                            "extraction_method": "yandex_vision_ocr",
                            "extraction_level": "raw_ocr",
                            "dpi": settings.dpi,
                        },
                    )
                    results.append(result)

                doc.close()

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Yandex Vision processing failed: {e}")

        return results

    @staticmethod
    def _parse_languages(lang: str) -> list[str]:
        """Parse language string to Yandex format.

        Args:
            lang: Language code like 'rus+eng', 'rus', 'eng'

        Returns:
            List of language codes for Yandex: ['ru', 'en']
        """
        yandex_langs = {
            "rus": "ru",
            "ru": "ru",
            "russian": "ru",
            "eng": "en",
            "en": "en",
            "english": "en",
        }

        if "+" in lang:
            parts = [p.strip().lower() for p in lang.split("+")]
            result = []
            for part in parts:
                mapped = yandex_langs.get(part, part.lower()[:2])
                if mapped:
                    result.append(mapped)
            return result if result else ["en"]

        lang_lower = lang.strip().lower()
        mapped = yandex_langs.get(lang_lower, lang_lower[:2])
        return [mapped] if mapped else ["en"]
