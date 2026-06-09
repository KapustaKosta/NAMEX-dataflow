
import unittest
import pandas as pd
from pathlib import Path
import sys

# Add src to sys.path
sys.path.append(str(Path.cwd()))

from src.user_profile_builder import (
    _source_rows_from_ocr_candidates,
    apply_table_reconstruction,
    apply_row_filters,
    apply_user_profile_to_sources,
    source_row_uid
)

class TestOcrProfileFlow(unittest.TestCase):
    def test_ocr_row_uid_stability_after_reconstruction(self):
        ocr_candidates_df = pd.DataFrame([
            {
                "source_file": "test.pdf",
                "page": 1,
                "ocr_block_id": "ocr_p1_b1",
                "block_title": "Test Table",
                "block_text": "Row1 | 100\nRow2 | 200",
                "preview": "Row1 | 100...",
            }
        ])
        
        # 1. Generate initial source rows
        source_rows = _source_rows_from_ocr_candidates(ocr_candidates_df)
        self.assertEqual(len(source_rows), 2)
        
        uid1 = source_rows[0]["row_uid"]
        uid2 = source_rows[1]["row_uid"]
        
        self.assertTrue(uid1.startswith("ocr_candidate:1:ocr_p1_b1:row:"))
        
        # 2. Apply reconstruction (split by |)
        reconstruction = {"method": "split_by_regex", "pattern": r"\|"}
        rebuilt_rows = apply_table_reconstruction(source_rows, reconstruction)
        
        self.assertEqual(len(rebuilt_rows), 2)
        self.assertEqual(rebuilt_rows[0]["row_uid"], uid1)
        self.assertEqual(rebuilt_rows[1]["row_uid"], uid2)
        self.assertEqual(rebuilt_rows[0]["cells"], ["Row1", "100"])
        
        # 3. Apply row filters (manual selection)
        row_filters = {
            "use_manual_rows": True,
            "selected_row_uids": [uid1]
        }
        filtered_rows = apply_row_filters(rebuilt_rows, row_filters)
        self.assertEqual(len(filtered_rows), 1)
        self.assertEqual(filtered_rows[0]["row_uid"], uid1)

    def test_ocr_profile_produces_rows(self):
        ocr_candidates_df = pd.DataFrame([
            {
                "source_file": "test.pdf",
                "page": 1,
                "ocr_block_id": "ocr_p1_b1",
                "block_title": "Test Table",
                "block_text": "Item1 100 200\nItem2 300 400",
            }
        ])
        
        # Get UIDs first
        source_rows = _source_rows_from_ocr_candidates(ocr_candidates_df)
        uid1 = source_rows[0]["row_uid"]
        
        profile_config = {
            "profile_name": "test_profile",
            "extraction": {"source": "ocr"},
            "blocks": [
                {
                    "selector": {"block_uids": ["ocr_candidate:1:ocr_p1_b1"]},
                    "row_selection": {
                        "use_manual_rows": True,
                        "selected_row_uids": [uid1]
                    },
                    "table_reconstruction": {"method": "split_by_regex", "pattern": r"\s+"},
                    "column_mapping": {
                        "column_1": {"role": "name"},
                        "column_2": {"role": "value", "metric": "price"}
                    }
                }
            ]
        }
        
        # Apply profile
        output_df = apply_user_profile_to_sources(None, ocr_candidates_df, profile_config)
        
        self.assertFalse(output_df.empty, "Output should not be empty")
        self.assertEqual(len(output_df), 1)
        self.assertEqual(output_df.iloc[0]["name"], "Item1")
        self.assertEqual(float(output_df.iloc[0]["value"]), 100.0)

    def test_ocr_token_mapping_produces_rows(self):
        ocr_candidates_df = pd.DataFrame([
            {
                "source_file": "test.pdf",
                "page": 1,
                "ocr_block_id": "ocr_p1_b1",
                "block_title": "Test Table",
                "block_text": "Item1 100\nItem2 300",
            }
        ])
        
        source_rows = _source_rows_from_ocr_candidates(ocr_candidates_df)
        uid1 = source_rows[0]["row_uid"]
        
        profile_config = {
            "profile_name": "test_token_profile",
            "extraction": {"source": "ocr"},
            "blocks": [
                {
                    "selector": {"block_uids": ["ocr_candidate:1:ocr_p1_b1"]},
                    "row_selection": {
                        "use_manual_rows": True,
                        "selected_row_uids": [uid1]
                    },
                    "token_mapping": {
                        "token_1": {"enabled": True, "role": "value", "metric": "volume"}
                    }
                }
            ]
        }
        
        output_df = apply_user_profile_to_sources(None, ocr_candidates_df, profile_config)
        
        self.assertFalse(output_df.empty, "Output should not be empty with token mapping")
        self.assertEqual(len(output_df), 1)
        self.assertEqual(output_df.iloc[0]["name"], "Item1")
        self.assertEqual(float(output_df.iloc[0]["value"]), 100.0)

if __name__ == "__main__":
    unittest.main()
