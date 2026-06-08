from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.extract_ocr import configure_tesseract, get_available_tesseract_languages


def main() -> int:
    cmd = configure_tesseract()
    print("cmd:", cmd)
    print("languages:", get_available_tesseract_languages())

    if cmd is None:
        return 1

    import pytesseract

    print("version:", pytesseract.get_tesseract_version())

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return 0

    image = Image.new("RGB", (240, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 24), "OCR TEST 123", fill="black")
    text = pytesseract.image_to_string(image, lang="eng").strip()
    print("sample_ocr:", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
