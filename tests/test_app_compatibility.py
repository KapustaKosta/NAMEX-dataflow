import unittest
import pandas as pd
from src.user_profile_builder import apply_user_profile

class TestAppCompatibility(unittest.TestCase):
    def test_apply_user_profile_returns_expected_keys(self):
        profile_config = {
            "profile_name": "test",
            "extraction": {"source": "pdf_text_layer"}
        }
        result = apply_user_profile(
            document={"raw_rows": pd.DataFrame()},
            profile_config=profile_config,
            raw_rows=pd.DataFrame()
        )
        # These are the keys app.py line 3085+ expects
        self.assertIn("ocr_ran", result)
        self.assertIn("status", result)
        self.assertIn("structured_rows", result)

if __name__ == "__main__":
    unittest.main()
