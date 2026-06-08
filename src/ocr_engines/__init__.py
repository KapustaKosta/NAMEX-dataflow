"""OCR engine registry and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ocr_engines.base import OcrEngine

if TYPE_CHECKING:
    pass


def get_ocr_engine(engine_name: str) -> OcrEngine:
    """Get OCR engine by name.

    Args:
        engine_name: Engine name ('tesseract', 'paddleocr', 'yandex_vision')

    Returns:
        OcrEngine instance

    Raises:
        ValueError: If engine name unknown
    """
    engine_name_lower = engine_name.lower().strip()

    if engine_name_lower == "tesseract":
        from src.ocr_engines.tesseract_engine import TesseractEngine

        return TesseractEngine()
    elif engine_name_lower == "paddleocr":
        from src.ocr_engines.paddle_engine import PaddleOCREngine

        return PaddleOCREngine()
    elif engine_name_lower in ("yandex", "yandex_vision"):
        from src.ocr_engines.yandex_vision_engine import YandexVisionEngine

        return YandexVisionEngine()
    else:
        raise ValueError(f"Unknown OCR engine: {engine_name}")


def get_available_engines() -> dict[str, tuple[str, tuple[bool, str]]]:
    """Get all available OCR engines and their availability status.

    Returns:
        Dict mapping engine name to (display_name, (is_available, message))
    """
    engines = {
        "tesseract": "Tesseract",
        "paddleocr": "PaddleOCR",
        "yandex_vision": "Yandex Vision OCR",
    }

    result = {}
    for engine_name, display_name in engines.items():
        try:
            engine = get_ocr_engine(engine_name)
            availability = engine.check_availability()
            result[engine_name] = (display_name, availability)
        except Exception as e:
            result[engine_name] = (display_name, (False, str(e)))

    return result


__all__ = [
    "OcrEngine",
    "get_ocr_engine",
    "get_available_engines",
]
