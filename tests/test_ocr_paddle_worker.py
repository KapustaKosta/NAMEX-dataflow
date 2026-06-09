import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.ocr_engines.paddle_engine import PaddleOCREngine
from src.ocr_engines.base import OcrSettings
import app  # Just to verify it can be imported after adding 'import re'

class TestPaddleOCRWorker(unittest.TestCase):
    @patch("subprocess.run")
    @patch("fitz.open")
    def test_paddle_ocr_engine_calls_worker(self, mock_fitz_open, mock_subprocess_run):
        # Mock fitz
        mock_doc = mock_fitz_open.return_value
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        
        # Mock subprocess worker response
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "success": True,
            "results": [
                {
                    "page": 1,
                    "lines": [
                        {"text": "Hello Paddle", "confidence": 0.99}
                    ]
                }
            ]
        })
        mock_subprocess_run.return_value = mock_proc
        
        engine = PaddleOCREngine()
        settings = OcrSettings(pages=[1])
        
        dummy_pdf = Path("dummy_worker_test.pdf")
        dummy_pdf.touch()
        try:
            results = engine.recognize_pdf(str(dummy_pdf), settings)
            
            # Verify subprocess was called
            self.assertTrue(mock_subprocess_run.called)
            args, kwargs = mock_subprocess_run.call_args
            
            self.assertIn("paddle_worker.py", args[0][1])
            input_data = json.loads(kwargs["input"])
            
            self.assertFalse(input_data["use_gpu"])
            self.assertEqual(input_data["lang"], "ru")
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].text, "Hello Paddle")
            self.assertEqual(results[0].confidence, 0.99)
            self.assertEqual(results[0].engine, "paddleocr")
        finally:
            if dummy_pdf.exists():
                dummy_pdf.unlink()

if __name__ == "__main__":
    unittest.main()
