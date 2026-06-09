import unittest
import pandas as pd
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.user_profile_builder import apply_table_reconstruction, extract_numeric_tokens, text_without_numeric_tokens

class TestTariffReconstruction(unittest.TestCase):
    def test_pair_name_row_with_following_value_row(self):
        """Verify that name/code rows are correctly paired with following value/unit rows."""
        
        source_rows = [
            {
                "source_row_id": "row30",
                "evidence_text": "2.1. Зерновые грузы",
                "cells": ["2.1. Зерновые грузы"],
                "numeric_tokens": ["2.1"],
                "text_part": "Зерновые грузы"
            },
            {
                "source_row_id": "row31",
                "evidence_text": "1 200",
                "cells": ["1 200"],
                "numeric_tokens": ["1 200"],
                "text_part": ""
            },
            {
                "source_row_id": "row32",
                "evidence_text": "2.1.1. Зерновые грузы, прибывшие ж/д транспортом",
                "cells": ["2.1.1. Зерновые грузы, прибывшие ж/д транспортом"],
                "numeric_tokens": ["2.1.1"],
                "text_part": "Зерновые грузы, прибывшие ж/д транспортом"
            },
            {
                "source_row_id": "row34",
                "evidence_text": "1 020",
                "cells": ["1 020"],
                "numeric_tokens": ["1 020"],
                "text_part": ""
            },
            {
                "source_row_id": "row35",
                "evidence_text": "2.2. Выгрузка зерна из трюма на автотранспорт",
                "cells": ["2.2. Выгрузка зерна из трюма на автотранспорт"],
                "numeric_tokens": ["2.2"],
                "text_part": "Выгрузка зерна из трюма на автотранспорт"
            },
            {
                "source_row_id": "row36",
                "evidence_text": "700",
                "cells": ["700"],
                "numeric_tokens": ["700"],
                "text_part": ""
            },
            {
                "source_row_id": "row37",
                "evidence_text": "2.3. Хранение на закрытом складе зерновых грузов с 16 суток",
                "cells": ["2.3. Хранение на закрытом складе зерновых грузов с 16 суток"],
                "numeric_tokens": ["2.3"],
                "text_part": "Хранение на закрытом складе зерновых грузов с 16 суток"
            },
            {
                "source_row_id": "row38",
                "evidence_text": "тн/сут",
                "cells": ["тн/сут"],
                "numeric_tokens": [],
                "text_part": "тн/сут"
            },
            {
                "source_row_id": "row39",
                "evidence_text": "35",
                "cells": ["35"],
                "numeric_tokens": ["35"],
                "text_part": ""
            }
        ]
        
        recon_config = {"method": "pair_name_row_with_following_value_row"}
        rebuilt = apply_table_reconstruction(source_rows, recon_config)
        
        self.assertEqual(len(rebuilt), 4)
        
        # Check first row (2.1)
        self.assertEqual(rebuilt[0]["reconstructed_code"], "2.1")
        self.assertEqual(rebuilt[0]["reconstructed_name"], "Зерновые грузы")
        self.assertEqual(rebuilt[0]["numeric_tokens"], ["1 200"])
        self.assertEqual(rebuilt[0]["cells"][2], "1 200") # Value at index 2
        
        # Check fourth row (2.3) with unit
        self.assertEqual(rebuilt[3]["reconstructed_code"], "2.3")
        self.assertEqual(rebuilt[3]["reconstructed_name"], "Хранение на закрытом складе зерновых грузов с 16 суток")
        self.assertEqual(rebuilt[3]["reconstructed_unit"], "тн/сут")
        self.assertEqual(rebuilt[3]["numeric_tokens"], ["35"])
        self.assertEqual(rebuilt[3]["cells"][2], "35")
        self.assertEqual(rebuilt[3]["cells"][3], "тн/сут")

    def test_selection_keys_with_reconstructed_ids(self):
        """Verify that selection keys correctly include IDs of consumed rows."""
        source_rows = [
            {
                "source_row_id": "id1",
                "source_kind": "ocr_candidate",
                "page": 1,
                "table_id": "b1",
                "evidence_text": "2.1. Service",
                "numeric_tokens": ["2.1"]
            },
            {
                "source_row_id": "id2",
                "source_kind": "ocr_candidate",
                "page": 1,
                "table_id": "b1",
                "evidence_text": "100",
                "numeric_tokens": ["100"]
            }
        ]
        
        from src.user_profile_builder import source_row_uid, _source_row_selection_keys
        uid1 = source_row_uid(source_rows[0])
        uid2 = source_row_uid(source_rows[1])
        
        rebuilt = apply_table_reconstruction(source_rows, {"method": "pair_name_row_with_following_value_row"})
        
        keys = _source_row_selection_keys(rebuilt[0])
        
        self.assertIn(uid1, keys)
        self.assertIn(uid2, keys)

class TestSavedProfileReconstructionOrder(unittest.TestCase):
    def test_saved_profile_pairs_correctly_from_anchor_selection(self):
        """Verify that only selecting name rows is sufficient to produce paired rows in production."""
        
        # 1. Arrange: full block of rows
        source_rows_df = pd.DataFrame([
            {"ocr_block_id": "b1", "page": 1, "block_text": "2.1. Service A\n1 200\n2.2. Service B\n35", "source_file": "f.pdf"}
        ])
        
        # 2. Arrange: saved profile with only anchor rows selected
        profile = {
            "profile_name": "reconstruction_test",
            "extraction": {"source": "ocr", "ocr": {"engine": "yandex_vision"}},
            "blocks": [
                {
                    "selector": {"block_uids": ["ocr_candidate:1:b1"]},
                    "row_selection": {
                        "use_manual_rows": True,
                        "selected_row_uids": [
                            "ocr_candidate:1:b1:row:1", # Service A
                            "ocr_candidate:1:b1:row:3"  # Service B
                        ]
                    },
                    "table_reconstruction": {"method": "pair_name_row_with_following_value_row"},
                    "column_mapping": {
                        "column_1": {"role": "code"},
                        "column_2": {"role": "name"},
                        "column_3": {"role": "value"}
                    }
                }
            ]
        }
        
        from src.user_profile_builder import apply_user_profile
        
        # 3. Act: apply profile
        result = apply_user_profile(
            {"document_key": "d1"},
            profile,
            ocr_candidates_df=source_rows_df
        )
        
        # 4. Assert
        rows = result["structured_rows"]
        self.assertEqual(len(rows), 2)
        
        self.assertEqual(rows.iloc[0]["code"], "2.1")
        self.assertEqual(rows.iloc[0]["value"], 1200.0)
        
        self.assertEqual(rows.iloc[1]["code"], "2.2")
        self.assertEqual(rows.iloc[1]["value"], 35.0)


if __name__ == "__main__":
    unittest.main()
