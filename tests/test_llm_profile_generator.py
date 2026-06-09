import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.llm_profile_generator import LLMProfileGenerator, validate_generated_profile
import yaml

class TestLLMProfileGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = LLMProfileGenerator(api_key="test_key")
        self.mock_ocr_candidates = pd.DataFrame([
            {
                "page": 2,
                "ocr_block_id": "block1",
                "block_text": "row 30: 2.1. Зерновые грузы\nrow 31: 1 200\nrow 35: 2.2. Выгрузка зерна\nrow 36: 700\nrow 40: 2.4. Погрузка на автотранспорт\nrow 41: 1 000",
                "rows": [
                    {"row_index": 30, "text": "2.1. Зерновые грузы"},
                    {"row_index": 31, "text": "1 200"},
                    {"row_index": 35, "text": "2.2. Выгрузка зерна"},
                    {"row_index": 36, "text": "700"},
                    {"row_index": 40, "text": "2.4. Погрузка на автотранспорт"},
                    {"row_index": 41, "text": "1 000"},
                ]
            }
        ])

    @patch.object(LLMProfileGenerator, '_call_openai')
    def test_generate_profile_success(self, mock_call):
        mock_yaml = """
profile_name: mmpt_grain
display_name: ММТП Зерновые
extraction:
  source: ocr
  ocr:
    engine: yandex_vision
    pages: [2]
blocks:
  - selector:
      block_uids: ["ocr_candidate:2:block1"]
    row_filters:
      include:
        any:
          - contains: "Зерновые"
          - contains: "Выгрузка зерна"
      exclude:
        any:
          - contains: "автотранспорт"
    column_mapping:
      column_1: {role: "name"}
      column_2: {role: "value"}
"""
        mock_call.return_value = mock_yaml
        
        doc_context = {"ocr_candidates_df": self.mock_ocr_candidates}
        instruction = "Зерновые на стр 2, кроме автотранспорта"
        
        profile = self.generator.generate_profile(doc_context, instruction)
        
        self.assertEqual(profile["profile_name"], "mmpt_grain")
        self.assertEqual(profile["extraction"]["source"], "ocr")
        self.assertEqual(len(profile["blocks"]), 1)
        
        errors = validate_generated_profile(profile)
        self.assertEqual(len(errors), 0)

    def test_validate_generated_profile_errors(self):
        invalid_profile = {"profile_name": "test"}
        errors = validate_generated_profile(invalid_profile)
        self.assertIn("Missing display_name", errors)
        self.assertIn("Missing extraction.source", errors)
        self.assertIn("Missing blocks or tables configuration", errors)

    def test_semantic_filter_application(self):
        from src.user_profile_builder import apply_row_filters
        
        source_rows = [
            {"evidence_text": "2.1. Зерновые грузы", "row_uid": "r1"},
            {"evidence_text": "2.2. Выгрузка зерна", "row_uid": "r2"},
            {"evidence_text": "2.4. Погрузка на автотранспорт", "row_uid": "r3"},
        ]
        
        row_filters = {
            "include": {"any": [{"contains": "Зерновые"}, {"contains": "Выгрузка"}]},
            "exclude": {"any": [{"contains": "автотранспорт"}]}
        }
        
        filtered = apply_row_filters(source_rows, row_filters)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["evidence_text"], "2.1. Зерновые грузы")
        self.assertEqual(filtered[1]["evidence_text"], "2.2. Выгрузка зерна")

if __name__ == "__main__":
    unittest.main()
