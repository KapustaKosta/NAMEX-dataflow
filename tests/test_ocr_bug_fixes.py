
import unittest
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add PROJECT_DIR to sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.ocr_table_candidates import extract_ocr_table_candidates
from src.ocr_engines.base import OcrSettings
from src.ocr_engines.conversion import ocr_page_results_to_dataframe
from src.ocr_engines.base import OcrPageResult

class TestYandexVisionFix(unittest.TestCase):
    def test_yandex_candidate_extraction(self):
        """Verify that Yandex Vision raw rows produce candidates."""
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
                "ocr_engine": "yandex_vision",
                "extraction_method": "yandex_vision_ocr",
                "extraction_level": "raw_ocr",
            }
        ])
        
        candidates = extract_ocr_table_candidates(ocr_df)
        self.assertFalse(candidates.empty, "Yandex raw rows should produce candidates")
        self.assertEqual(candidates.iloc[0]["ocr_engine"], "yandex_vision")
        self.assertIn("yandex_vision", candidates.iloc[0]["extraction_method"])

    def test_yandex_fallback_candidate(self):
        """Verify that Yandex Vision raw rows produce fallback candidates if no blocks found."""
        # Text with enough numbers but no clear heading
        fallback_text = "\n".join([f"Line with number {i} and value {i*10.5}" for i in range(10)])
        ocr_df = pd.DataFrame([
            {
                "source_file": "test.pdf",
                "page": 2,
                "evidence_text": fallback_text,
                "ocr_engine": "yandex_vision",
                "extraction_method": "yandex_vision_ocr",
                "extraction_level": "raw_ocr",
            }
        ])
        
        candidates = extract_ocr_table_candidates(ocr_df)
        self.assertFalse(candidates.empty, "Yandex raw rows should produce fallback candidates")
        self.assertEqual(candidates.iloc[0]["ocr_engine"], "yandex_vision")
        self.assertIn("fallback", candidates.iloc[0]["ocr_block_id"])
        self.assertEqual(candidates.iloc[0]["extraction_method"], "yandex_vision_ocr_candidate_fallback")

class TestOcrEngineFieldPropagation(unittest.TestCase):
    def test_ocr_engine_field_in_conversion(self):
        """Verify that ocr_engine field is present in converted dataframe."""
        results = [
            OcrPageResult(page=1, text="text", engine="paddleocr")
        ]
        df = ocr_page_results_to_dataframe(results, "test.pdf")
        self.assertIn("ocr_engine", df.columns)
        self.assertEqual(df.iloc[0]["ocr_engine"], "paddleocr")

if __name__ == "__main__":
    unittest.main()
