
import unittest
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to sys.path
sys.path.append(str(Path.cwd()))

from src.user_profile_builder import (
    _source_rows_from_ocr_candidates,
    apply_user_profile,
    apply_user_profile_to_sources
)
from src.ocr_table_candidates import extract_ocr_table_candidates

class TestOcrProfilePersistence(unittest.TestCase):
    def test_apply_saved_ocr_profile(self):
        # 1. Simulate OCR results
        ocr_df = pd.DataFrame([
            {
                "source_file": "test.pdf",
                "page": 2,
                "evidence_text": (
                    "Тарифы на погрузочно-разгрузочные работы и иные работы услуги сера\n"
                    "Наименование услуги | Ед. изм. | Ставка\n"
                    "Погрузка серы навалом | тонн | 1 200\n"
                    "Выгрузка серы из вагонов | тонн | 1 020\n"
                    "Хранение серы на складе | тонн/сут | 700\n"
                    "Дополнительные услуги по перевалке | тонн | 150\n"
                    "Итого по разделу 1 | тонн | 3 070\n"
                    "Конец таблицы"
                ),
                "extraction_method": "tesseract_ocr",
                "extraction_level": "raw_ocr",
            }
        ])
        
        # 2. Generate candidates (fallback logic)
        candidates_df = extract_ocr_table_candidates(ocr_df)
        self.assertFalse(candidates_df.empty)
        # Find fallback candidate for page 2
        fallback_row = candidates_df[candidates_df["ocr_block_id"] == "ocr_p2_fallback"].iloc[0]
        block_uid = f"ocr_candidate:2:ocr_p2_fallback"
        
        # Get UIDs for specific rows
        source_rows = _source_rows_from_ocr_candidates(candidates_df)
        # Filter for the fallback block
        block_rows = [r for r in source_rows if r["block_uid"] == block_uid]
        # Select first 3 numeric rows
        selected_uids = [r["row_uid"] for r in block_rows if r["numeric_tokens"]][:3]
        
        # 3. Simulate Profile Builder Config
        profile_config = {
            "profile_name": "test_ocr_persistence",
            "extraction": {
                "source": "ocr",
                "ocr": {"engine": "tesseract", "pages": [2], "dpi": 300}
            },
            "blocks": [
                {
                    "selector": {"block_uids": [block_uid]},
                    "row_selection": {
                        "use_manual_rows": True,
                        "selected_row_uids": selected_uids
                    },
                    "token_mapping": {
                        "token_1": {"enabled": True, "role": "value", "metric": "volume"}
                    }
                }
            ]
        }
        
        # 4. Apply profile (normal mode)
        # We simulate ocr_runner returning the same OCR results
        def mock_ocr_runner(doc, config):
            return {"ocr_result_df": ocr_df, "ocr_candidates_df": candidates_df}
            
        result = apply_user_profile(
            {"file_path": "test.pdf"},
            profile_config,
            ocr_runner=mock_ocr_runner
        )
        
        structured_rows = result["structured_rows"]
        
        self.assertFalse(structured_rows.empty, "Structured rows should NOT be empty")
        self.assertEqual(len(structured_rows), 3)
        self.assertEqual(float(structured_rows.iloc[0]["value"]), 1200.0)
        self.assertEqual(float(structured_rows.iloc[1]["value"]), 1020.0)
        self.assertEqual(float(structured_rows.iloc[2]["value"]), 700.0)

if __name__ == "__main__":
    unittest.main()
