
import unittest
import pandas as pd
from pathlib import Path
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to sys.path
sys.path.append(str(Path.cwd()))

from src.ocr_engines import get_available_engines, get_ocr_engine
from src.ocr_table_candidates import extract_ocr_table_candidates

class TestOcrEngineRegistry(unittest.TestCase):
    def test_engine_factory_returns_correct_types(self):
        tesseract = get_ocr_engine("tesseract")
        self.assertEqual(tesseract.name, "tesseract")
        
        yandex = get_ocr_engine("yandex_vision")
        self.assertEqual(yandex.name, "yandex_vision")

class TestOcrCandidatePropagation(unittest.TestCase):
    def test_extraction_method_propagation(self):
        ocr_df = pd.DataFrame([
            {
                "source_file": "test.pdf",
                "page": 1,
                "evidence_text": (
                    "Таблица импорта\n"
                    "Товар 1 | 100\n"
                    "Товар 2 | 200\n"
                    "Товар 3 | 300\n"
                    "Товар 4 | 400"
                ),
                "extraction_method": "yandex_vision_ocr",
                "extraction_level": "raw_ocr",
            }
        ])
        
        candidates = extract_ocr_table_candidates(ocr_df)
        self.assertFalse(candidates.empty)
        # It should have yandex_vision in extraction_method
        method = candidates.iloc[0]["extraction_method"]
        self.assertIn("yandex_vision", method)
        self.assertIn("candidate", method)

    def test_fallback_propagation(self):
        # Long text to trigger fallback
        long_text = "Table Header\n" + "\n".join([f"Item {i} | {i*100}" for i in range(10)])
        ocr_df = pd.DataFrame([
            {
                "source_file": "test.pdf",
                "page": 2,
                "evidence_text": long_text,
                "extraction_method": "paddleocr_ocr",
                "extraction_level": "raw_ocr",
            }
        ])
        
        candidates = extract_ocr_table_candidates(ocr_df)
        self.assertFalse(candidates.empty)
        method = candidates.iloc[0]["extraction_method"]
        self.assertIn("paddleocr", method)
        self.assertIn("candidate", method)

if __name__ == "__main__":
    unittest.main()
