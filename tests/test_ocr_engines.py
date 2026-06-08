"""Tests for OCR engine interface and implementations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.ocr_engines import get_ocr_engine, get_available_engines
from src.ocr_engines.base import OcrPageResult, OcrSettings
from src.ocr_engines.tesseract_engine import TesseractEngine
from src.ocr_engines.conversion import ocr_page_results_to_dataframe


class TestOcrSettings(unittest.TestCase):
    """Test OcrSettings dataclass."""

    def test_ocr_settings_defaults(self) -> None:
        """Test default OcrSettings values."""
        settings = OcrSettings()
        self.assertEqual(settings.lang, "rus+eng")
        self.assertEqual(settings.dpi, 300)
        self.assertEqual(settings.pages, [])
        self.assertFalse(settings.use_gpu)

    def test_ocr_settings_custom_values(self) -> None:
        """Test custom OcrSettings values."""
        settings = OcrSettings(
            lang="eng",
            dpi=150,
            pages=[1, 2, 3],
            use_gpu=True,
        )
        self.assertEqual(settings.lang, "eng")
        self.assertEqual(settings.dpi, 150)
        self.assertEqual(settings.pages, [1, 2, 3])
        self.assertTrue(settings.use_gpu)

    def test_ocr_settings_invalid_dpi(self) -> None:
        """Test OcrSettings rejects invalid DPI."""
        with self.assertRaises(ValueError):
            OcrSettings(dpi=50)


class TestOcrPageResult(unittest.TestCase):
    """Test OcrPageResult dataclass."""

    def test_ocr_page_result_creation(self) -> None:
        """Test creating OcrPageResult."""
        result = OcrPageResult(
            page=1,
            text="Sample OCR text",
            engine="tesseract",
        )
        self.assertEqual(result.page, 1)
        self.assertEqual(result.text, "Sample OCR text")
        self.assertEqual(result.engine, "tesseract")

    def test_ocr_page_result_invalid_page(self) -> None:
        """Test OcrPageResult rejects invalid page number."""
        with self.assertRaises(ValueError):
            OcrPageResult(page=0, text="text")
        with self.assertRaises(ValueError):
            OcrPageResult(page=-1, text="text")

    def test_ocr_page_result_invalid_text(self) -> None:
        """Test OcrPageResult rejects non-string text."""
        with self.assertRaises(ValueError):
            OcrPageResult(page=1, text=123)


class TestTesseractEngine(unittest.TestCase):
    """Test Tesseract OCR engine."""

    def test_tesseract_engine_name(self) -> None:
        """Test Tesseract engine has correct name."""
        engine = TesseractEngine()
        self.assertEqual(engine.name, "tesseract")

    def test_tesseract_check_availability(self) -> None:
        """Test checking Tesseract availability."""
        engine = TesseractEngine()
        available, msg = engine.check_availability()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(msg, str)


class TestOcrEngineFactory(unittest.TestCase):
    """Test OCR engine factory functions."""

    def test_get_ocr_engine_tesseract(self) -> None:
        """Test getting Tesseract engine."""
        engine = get_ocr_engine("tesseract")
        self.assertIsInstance(engine, TesseractEngine)
        self.assertEqual(engine.name, "tesseract")

    def test_get_ocr_engine_case_insensitive(self) -> None:
        """Test getting engine is case insensitive."""
        engine1 = get_ocr_engine("TESSERACT")
        engine2 = get_ocr_engine("Tesseract")
        engine3 = get_ocr_engine("tesseract")
        self.assertEqual(engine1.name, engine2.name)
        self.assertEqual(engine2.name, engine3.name)

    def test_get_ocr_engine_invalid(self) -> None:
        """Test getting invalid engine raises error."""
        with self.assertRaises(ValueError):
            get_ocr_engine("invalid_engine")

    def test_get_available_engines(self) -> None:
        """Test getting available engines."""
        engines = get_available_engines()
        self.assertIsInstance(engines, dict)
        self.assertIn("tesseract", engines)
        
        for engine_name, (display_name, (is_available, message)) in engines.items():
            self.assertIsInstance(display_name, str)
            self.assertIsInstance(is_available, bool)
            self.assertIsInstance(message, str)


class TestOcrConversion(unittest.TestCase):
    """Test OCR result conversion utilities."""

    def test_ocr_page_results_to_dataframe_single_page(self) -> None:
        """Test converting single page result to dataframe."""
        results = [
            OcrPageResult(
                page=1,
                text="Sample text",
                engine="tesseract",
            )
        ]
        
        df = ocr_page_results_to_dataframe(results, "test.pdf")
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["source_file"], "test.pdf")
        self.assertEqual(df.iloc[0]["page"], 1)
        self.assertEqual(df.iloc[0]["evidence_text"], "Sample text")
        self.assertEqual(df.iloc[0]["extraction_method"], "tesseract_ocr")
        self.assertEqual(df.iloc[0]["extraction_level"], "raw_ocr")

    def test_ocr_page_results_to_dataframe_multiple_pages(self) -> None:
        """Test converting multiple page results to dataframe."""
        results = [
            OcrPageResult(page=1, text="Page 1 text", engine="tesseract"),
            OcrPageResult(page=2, text="Page 2 text", engine="tesseract"),
            OcrPageResult(page=3, text="Page 3 text", engine="tesseract"),
        ]
        
        df = ocr_page_results_to_dataframe(results, "multipage.pdf")
        
        self.assertEqual(len(df), 3)
        self.assertEqual(df["page"].tolist(), [1, 2, 3])
        self.assertEqual(df["evidence_text"].tolist(), ["Page 1 text", "Page 2 text", "Page 3 text"])

    def test_ocr_page_results_to_dataframe_empty(self) -> None:
        """Test converting empty results to dataframe."""
        df = ocr_page_results_to_dataframe([], "test.pdf")
        
        self.assertEqual(len(df), 0)
        self.assertIn("source_file", df.columns)
        self.assertIn("page", df.columns)
        self.assertIn("evidence_text", df.columns)


if __name__ == "__main__":
    unittest.main()
