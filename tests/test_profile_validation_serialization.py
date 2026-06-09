import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import build_profile_builder_config

class TestProfileValidationSerialization(unittest.TestCase):
    @patch("app.profile_builder_extraction_config")
    @patch("app.build_profile_builder_column_mapping")
    @patch("app.build_profile_builder_token_mapping")
    @patch("app.profile_builder_reconstruction_config")
    @patch("app.build_profile_builder_row_filters")
    @patch("streamlit.session_state", new_callable=dict)
    def test_dynamic_required_fields_resolution(
        self,
        mock_session,
        mock_row_filters,
        mock_recon,
        mock_token_map,
        mock_col_map,
        mock_ext_config
    ):
        """Verify that required_fields adjusts to missing name/value when they are not mapped."""
        mock_ext_config.return_value = {"source": "ocr", "ocr": {"engine": "yandex_vision"}}
        mock_col_map.return_value = {"column_1": {"role": "code"}}
        mock_token_map.return_value = {} # No token mapping
        mock_recon.return_value = {}
        mock_row_filters.return_value = {}
        
        config = build_profile_builder_config(
            "doc123",
            ["ocr_candidate:1:ocr_p1_b1"],
            [],
            5,
            use_token_mapping=False
        )
        
        # Because we only mapped "code", we should not require "name" or "value"
        # However, our fallback injects ["name", "value"] if no required fields were identified.
        # But wait, "code" maps to "name" in the resolution logic. Let's see:
        # role "code" -> requires "name".
        self.assertIn("name", config["validation"]["required_fields"])
        self.assertNotIn("value", config["validation"]["required_fields"])

    @patch("app.profile_builder_extraction_config")
    @patch("app.build_profile_builder_column_mapping")
    @patch("app.build_profile_builder_token_mapping")
    @patch("app.profile_builder_reconstruction_config")
    @patch("app.build_profile_builder_row_filters")
    @patch("streamlit.session_state", new_callable=dict)
    def test_dynamic_required_fields_resolution_both(
        self,
        mock_session,
        mock_row_filters,
        mock_recon,
        mock_token_map,
        mock_col_map,
        mock_ext_config
    ):
        mock_ext_config.return_value = {"source": "ocr"}
        mock_col_map.return_value = {"column_1": {"role": "name"}, "column_2": {"role": "value"}}
        mock_token_map.return_value = {}
        mock_recon.return_value = {}
        mock_row_filters.return_value = {}
        
        config = build_profile_builder_config("doc1", [], [], 2)
        
        self.assertIn("name", config["validation"]["required_fields"])
        self.assertIn("value", config["validation"]["required_fields"])

if __name__ == "__main__":
    unittest.main()
