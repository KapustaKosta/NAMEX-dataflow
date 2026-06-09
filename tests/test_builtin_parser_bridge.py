import unittest
import pandas as pd
from pathlib import Path
import sys

# Add PROJECT_DIR to sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from unittest.mock import MagicMock, patch
from src.user_profile_builder import apply_user_profile

class TestBuiltinParserBridge(unittest.TestCase):
    @patch("src.parsers.fish_market_report.parse_fish_market_report")
    def test_fish_market_report_bridge(self, mock_parser):
        # Mock raw rows
        raw_rows = pd.DataFrame([
            {"source_file": "test.pdf", "page": 1, "evidence_text": "Some text", "extraction_method": "pdfplumber"},
        ])
        
        mock_parser.return_value = pd.DataFrame([{"commodity": "Test Fish"}])
        
        profile_config = {
            "profile_name": "fish_market_test",
            "parser": {
                "type": "builtin",
                "name": "fish_market_report"
            },
            "extraction": {"source": "pdf_text_layer"}
        }
        
        result = apply_user_profile(
            document={"raw_rows": raw_rows},
            profile_config=profile_config,
            raw_rows=raw_rows
        )
        
        self.assertTrue(mock_parser.called)
        structured_rows = result["structured_rows"]
        self.assertEqual(structured_rows.iloc[0]["commodity"], "Test Fish")

if __name__ == "__main__":
    unittest.main()
