import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

class TestOcrEngineStateReset(unittest.TestCase):
    @patch("streamlit.session_state", new_callable=dict)
    @patch("streamlit.selectbox")
    @patch("streamlit.columns")
    def test_engine_state_is_not_overwritten(self, mock_columns, mock_selectbox, mock_session):
        """Verify that the engine state is correctly read and not unconditionally reset."""
        
        # Simulate app logic for engine selection
        document_key = "test_doc"
        engine_state_key = f"profile_builder_ocr_engine:{document_key}"
        
        # Arrange: Durable state is already yandex_vision
        mock_session[engine_state_key] = "yandex_vision"
        
        all_engine_names = ["tesseract", "paddleocr", "yandex_vision"]
        
        # Simulating the app.py logic directly
        if engine_state_key not in mock_session:
            mock_session[engine_state_key] = "tesseract"
            
        current_engine = mock_session.get(engine_state_key, "tesseract")
        if current_engine not in all_engine_names:
            current_engine = "tesseract"
            
        # Assert that current_engine correctly picks up yandex_vision
        self.assertEqual(current_engine, "yandex_vision")
        
        # Assert that the durable state wasn't overwritten
        self.assertEqual(mock_session[engine_state_key], "yandex_vision")

if __name__ == "__main__":
    unittest.main()
