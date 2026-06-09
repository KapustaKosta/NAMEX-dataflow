
import unittest
import pandas as pd
from pathlib import Path
import sys
import os
from unittest.mock import MagicMock, patch

# Add PROJECT_DIR to sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.ocr_engines.paddle_engine import PaddleOCREngine
from src.ocr_engines.yandex_vision_engine import YandexVisionEngine
from src.ocr_engines.base import OcrSettings

class TestOcrRuntimeFixes(unittest.TestCase):
    @patch("requests.post")
    @patch("fitz.open")
    @patch("src.ocr_engines.yandex_vision_engine._get_yandex_credentials")
    @patch("builtins.open", new_callable=MagicMock)
    def test_yandex_vision_parsing_variations(self, mock_open, mock_creds, mock_fitz_open, mock_post):
        """Verify that Yandex Vision correctly parses various response shapes."""
        mock_creds.return_value = {"api_key": "fake-key"}
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake-image-bytes"
        mock_open.return_value.__enter__.return_value = mock_file
        
        mock_doc = mock_fitz_open.return_value
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        mock_pix = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix
        
        engine = YandexVisionEngine()
        settings = OcrSettings(pages=[1])
        dummy_pdf = Path("dummy_yandex_parse_test.pdf")
        dummy_pdf.touch()

        try:
            # 1. Test nested "result" with "fullText"
            mock_post.return_value.json.return_value = {
                "result": {
                    "textAnnotation": {
                        "fullText": "Nested Full Text"
                    }
                }
            }
            results = engine.recognize_pdf(str(dummy_pdf), settings)
            self.assertEqual(results[0].text, "Nested Full Text")

            # 2. Test flat "textAnnotation" with "fullText"
            mock_post.return_value.json.return_value = {
                "textAnnotation": {
                    "fullText": "Flat Full Text"
                }
            }
            results = engine.recognize_pdf(str(dummy_pdf), settings)
            self.assertEqual(results[0].text, "Flat Full Text")

            # 3. Test lines text fallback
            mock_post.return_value.json.return_value = {
                "result": {
                    "textAnnotation": {
                        "blocks": [
                            {"lines": [{"text": "Line 1"}, {"text": "Line 2"}]}
                        ]
                    }
                }
            }
            results = engine.recognize_pdf(str(dummy_pdf), settings)
            self.assertEqual(results[0].text, "Line 1\nLine 2")

            # 4. Test empty text exception
            mock_post.return_value.json.return_value = {
                "result": {"textAnnotation": {"blocks": []}}
            }
            mock_post.return_value.status_code = 200
            
            with self.assertRaisesRegex(RuntimeError, "Yandex Vision returned no text. Response shape:"):
                engine.recognize_pdf(str(dummy_pdf), settings)

        finally:
            if dummy_pdf.exists():
                dummy_pdf.unlink()

if __name__ == "__main__":
    unittest.main()
