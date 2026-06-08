from __future__ import annotations

import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import STANDARD_COLUMNS
from .utils import empty_standard_dataframe, ensure_standard_columns


TESSERACT_INSTALL_MESSAGE = (
    "Не найден tesseract.exe. Установите Tesseract OCR или задайте переменную окружения TESSERACT_CMD."
)

WINDOWS_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


class OCRUnavailableError(RuntimeError):
    """Raised when OCR dependencies or the Tesseract binary are unavailable."""


class OCRLanguageError(RuntimeError):
    """Raised when Tesseract is available but the requested OCR language is not."""


class OCRPageRenderError(RuntimeError):
    """Raised when a selected PDF page cannot be rendered for OCR."""


def _load_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRUnavailableError("Для OCR требуется библиотека pytesseract.") from exc
    return pytesseract


def get_tesseract_cmd() -> str | None:
    """Resolve the Tesseract executable from env, PATH, or common Windows installs."""
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        env_cmd = env_cmd.strip().strip('"')
        if Path(env_cmd).exists():
            return env_cmd

    for candidate in WINDOWS_TESSERACT_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd

    return None


def configure_tesseract() -> str | None:
    """Point pytesseract at the resolved executable, when available."""
    cmd = get_tesseract_cmd()
    if cmd is None:
        return None

    pytesseract = _load_pytesseract()

    pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd


def is_tesseract_available() -> bool:
    """Return True when pytesseract can reach a resolved Tesseract executable."""
    try:
        if configure_tesseract() is None:
            return False
        pytesseract = _load_pytesseract()
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def get_available_tesseract_languages() -> list[str]:
    """Return language codes reported by the configured Tesseract executable."""
    try:
        cmd = configure_tesseract()
    except Exception:
        return []
    if cmd is None:
        return []

    try:
        result = subprocess.run(
            [cmd, "--list-langs"],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        languages = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("List of available languages")
        ]
        if languages:
            return list(dict.fromkeys(languages))
    except Exception:
        pass

    try:
        pytesseract = _load_pytesseract()
        languages = pytesseract.get_languages(config="")
    except Exception:
        return []

    return list(dict.fromkeys(str(language) for language in languages))


def is_language_available(lang: str, available_languages: list[str]) -> bool:
    parts = [part.strip() for part in lang.split("+") if part.strip()]
    if not parts:
        return False
    return all(part in available_languages for part in parts)


def _format_language_error(lang: str, available_languages: list[str]) -> str:
    languages = ", ".join(available_languages) if available_languages else "не удалось определить"
    return (
        "Tesseract найден, но возникла ошибка языка OCR. "
        f"Доступные языки: {languages}. Выбранный язык: {lang}"
    )


def _is_tesseract_language_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "failed loading language",
            "error opening data file",
            "could not initialize tesseract",
            "tessdata",
        )
    )


def _validate_pages(pages: Iterable[int]) -> list[int]:
    page_numbers = []
    for page in pages:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number > 0 and page_number not in page_numbers:
            page_numbers.append(page_number)
    return page_numbers


def get_pdf_page_count(file_path: str) -> int:
    """Get total number of pages in a PDF file."""
    try:
        import fitz
    except ImportError as exc:
        raise OCRUnavailableError("Для определения количества страниц требуется библиотека pymupdf.") from exc
    
    try:
        with fitz.open(file_path) as pdf:
            return len(pdf)
    except Exception as exc:
        raise RuntimeError(f"Ошибка при определении количества страниц PDF: {exc}") from exc


def _base_ocr_record(source_file: str, page: int, row_id: int) -> dict[str, object]:
    record = {column: None for column in STANDARD_COLUMNS}
    record["source_file"] = source_file
    record["source_type"] = "pdf"
    record["page"] = page
    record["row_id"] = row_id
    record["section_name"] = "ocr_page_text"
    record["bbox"] = None
    record["extraction_method"] = "tesseract_ocr"
    record["extraction_level"] = "raw_ocr"
    record["text_layer_quality"] = "ocr"
    record["text_layer_warning"] = None
    return record


def extract_ocr_pages(file_path: str, pages: list[int], lang: str = "rus+eng") -> pd.DataFrame:
    """Run OCR for selected 1-based PDF pages and return raw OCR text rows."""
    page_numbers = _validate_pages(pages)
    if not page_numbers:
        return empty_standard_dataframe()

    cmd = configure_tesseract()
    if cmd is None:
        raise OCRUnavailableError(TESSERACT_INSTALL_MESSAGE)

    pytesseract = _load_pytesseract()
    available_languages = get_available_tesseract_languages()
    if available_languages and not is_language_available(lang, available_languages):
        raise OCRLanguageError(_format_language_error(lang, available_languages))

    try:
        import fitz
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailableError("Для OCR требуются библиотеки pymupdf и Pillow.") from exc

    source_file = Path(file_path).name
    records = []

    with fitz.open(file_path) as pdf:
        page_count = len(pdf)
        for row_id, page_number in enumerate(page_numbers, start=1):
            if page_number > page_count:
                continue

            try:
                page = pdf.load_page(page_number - 1)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(BytesIO(pixmap.tobytes("png")))
            except Exception as exc:
                raise OCRPageRenderError(f"Ошибка рендера страницы PDF: {exc}") from exc

            try:
                text = pytesseract.image_to_string(image, lang=lang).strip()
            except Exception as exc:
                if _is_tesseract_language_error(exc):
                    raise OCRLanguageError(_format_language_error(lang, available_languages)) from exc
                raise

            record = _base_ocr_record(source_file, page_number, row_id)
            record["evidence_text"] = text or None
            record["confidence"] = 0.6 if text else 0.2
            record["validation_status"] = "raw_extracted"
            record["review_status"] = "needs_profile_setup" if text else "manual_required"
            records.append(record)

    if not records:
        return empty_standard_dataframe()

    return ensure_standard_columns(pd.DataFrame(records))


def extract_ocr(file_path: str) -> pd.DataFrame:
    """Backward-compatible OCR entry point.

    OCR is intentionally page-scoped for generic PDF fallback. Use
    ``extract_ocr_pages(file_path, pages=[...])`` to avoid OCR-processing an
    entire document by accident.
    """
    return empty_standard_dataframe()
