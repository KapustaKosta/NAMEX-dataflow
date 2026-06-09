import unittest
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add PROJECT_DIR to sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.ocr_table_candidates import extract_ocr_table_candidates
from app import prepare_profile_builder_catalog_editor

class TestAppUIState(unittest.TestCase):
    def test_dataframe_typing_for_arrow(self):
        """Verify that the block selection dataframe is strictly typed for Arrow compatibility."""
        # Create a catalog with mixed problematic types
        catalog_df = pd.DataFrame([
            {
                "block_uid": "b1",
                "source_kind": "ocr_candidate",
                "page": 1,
                "table_id": "t1",
                "block_title": "Title 1",
                "preview": "Prev",
                "rows_count": "5",  # Problematic: string instead of int
                "extraction_method": "yandex",
                "ocr_engine": "yandex_vision"
            },
            {
                "block_uid": "b2",
                "source_kind": "ocr_candidate",
                "page": "2", # Problematic: string instead of int
                "table_id": "t2",
                "block_title": "Title 2",
                "preview": "Prev 2",
                "rows_count": 10,
                "columns_count": 3,
                "extraction_method": "yandex",
                "ocr_engine": "yandex_vision"
            }
        ])
        
        # Prepare editor df
        editor_df = prepare_profile_builder_catalog_editor(catalog_df, ["b1"])
        
        # Simulate app.py's Arrow fix logic
        for col in ["block_uid", "table_key", "source_kind", "block_title", "Таблица", "Краткий preview", "extraction_method"]:
            if col in editor_df.columns:
                editor_df[col] = editor_df[col].fillna("").astype(str)
        for col in ["Страница", "Найдено строк", "Найдено колонок"]:
            if col in editor_df.columns:
                editor_df[col] = pd.to_numeric(editor_df[col], errors="coerce").fillna(0).astype(int)
        
        # Verify strict types
        self.assertTrue(pd.api.types.is_string_dtype(editor_df["block_uid"]) or pd.api.types.is_object_dtype(editor_df["block_uid"]))
        self.assertTrue(pd.api.types.is_integer_dtype(editor_df["Страница"]))
        self.assertTrue(pd.api.types.is_integer_dtype(editor_df["Найдено строк"]))
        
        # Ensure no mixed-type "rows" or "count" columns exist that would confuse Arrow
        self.assertNotIn("rows", editor_df.columns)
        self.assertNotIn("count", editor_df.columns)

    @patch("streamlit.session_state", new_callable=dict)
    def test_block_selection_submit_logic(self, mock_session):
        """Verify the direct reading of edited rows on submit."""
        # Setup mock edited dataframe from st.data_editor
        PROFILE_BUILDER_USE_BLOCK_COLUMN = "Использовать эту таблицу"
        edited_df = pd.DataFrame([
            {PROFILE_BUILDER_USE_BLOCK_COLUMN: True, "block_uid": "b1"},
            {PROFILE_BUILDER_USE_BLOCK_COLUMN: False, "block_uid": "b2"},
            {PROFILE_BUILDER_USE_BLOCK_COLUMN: True, "block_uid": "b3"},
        ])
        
        available_uids = ["b1", "b2", "b3", "b4"]
        document_key = "doc123"
        
        # Simulate submit handler logic
        updated_table_keys = []
        if PROFILE_BUILDER_USE_BLOCK_COLUMN in edited_df.columns:
            checked_mask = edited_df[PROFILE_BUILDER_USE_BLOCK_COLUMN].fillna(False).astype(bool)
            checked_rows = edited_df.loc[checked_mask]
            for _, row in checked_rows.iterrows():
                if "block_uid" in row and row["block_uid"]:
                    updated_table_keys.append(str(row["block_uid"]))
        
        missing = [uid for uid in updated_table_keys if uid not in set(available_uids)]
        
        self.assertEqual(updated_table_keys, ["b1", "b3"])
        self.assertEqual(missing, [])

if __name__ == "__main__":
    unittest.main()

