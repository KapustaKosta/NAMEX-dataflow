import unittest
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add PROJECT_DIR to sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.user_profile_builder import apply_row_filters

class TestPreviewStateFixes(unittest.TestCase):
    def test_apply_row_filters_does_not_take_token_mapping(self):
        """Verify that apply_row_filters accepts exactly the arguments we pass from app.py."""
        source_rows = [
            {"evidence_text": "Row 1", "cells": ["A", "B"]},
            {"evidence_text": "Row 2", "cells": ["C", "D"]},
        ]
        row_filters = [{"type": "keep_text_contains", "text": "Row 1"}]
        column_mapping = {"col1": {"role": "commodity", "index": 0}}
        
        # This should execute without TypeError
        try:
            filtered = apply_row_filters(
                source_rows,
                row_filters,
                column_mapping
            )
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["evidence_text"], "Row 1")
        except TypeError as e:
            self.fail(f"apply_row_filters raised TypeError: {e}")

if __name__ == "__main__":
    unittest.main()
