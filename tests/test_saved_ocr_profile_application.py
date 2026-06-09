import unittest
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add PROJECT_DIR to sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.user_profile_builder import apply_user_profile

class TestSavedOCRProfileApplication(unittest.TestCase):
    def test_saved_yandex_profile_applies_correctly(self):
        """Verify that a saved profile built with yandex_vision correctly matches OCR candidates."""
        
        # 1. Arrange a mocked saved profile and OCR candidates
        saved_profile = {
            "profile_name": "Тарифы ММТП",
            "extraction": {
                "source": "ocr",
                "ocr": {
                    "required": True,
                    "engine": "yandex_vision",
                    "lang": "rus+eng",
                    "pages": "auto",
                    "dpi": 300
                }
            },
            "blocks": [
                {
                    "selector": {
                        "block_uids": ["ocr_candidate:2:ocr_p2_fallback"]
                    },
                    "column_mapping": {
                        "column_1": {"role": "name"},
                        "column_2": {"role": "value", "value_type": "numeric"}
                    }
                }
            ]
        }
        
        mock_ocr_candidates = pd.DataFrame([
            {
                "ocr_block_id": "ocr_p2_fallback",
                "source_kind": "ocr_candidate",
                "page": 2,
                "block_title": "Page 2",
                "extraction_method": "yandex_vision_ocr_candidate",
                "evidence_text": "Услуга 1 | 100",
                "block_text": "Услуга 1 | 100"
            }
        ])
        
        # 2. Act: Apply the user profile
        result = apply_user_profile(
            {"document_key": "test_doc"},
            saved_profile,
            ocr_candidates_df=mock_ocr_candidates
        )
        
        # 3. Assert
        self.assertEqual(result["extraction_source"], "ocr")
        self.assertFalse(result["structured_rows"].empty, "Structured rows should not be empty")
        
        first_row = result["structured_rows"].iloc[0]
        self.assertEqual(first_row["name"], "Услуга 1")
        self.assertEqual(first_row["value"], 100.0)
        self.assertEqual(first_row["extraction_method"], "user_profile_parser")

if __name__ == "__main__":
    unittest.main()
