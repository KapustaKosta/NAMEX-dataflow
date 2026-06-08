"""Test multi-page OCR processing and candidate extraction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.extract_ocr import extract_ocr_pages, get_pdf_page_count
from src.ocr_table_candidates import extract_ocr_table_candidates
from src.validate import validate_extracted_data


class TestOCRMultipage(unittest.TestCase):
    """Test suite for multi-page OCR processing."""

    def test_extract_ocr_table_candidates_processes_all_pages(self) -> None:
        """Verify that extract_ocr_table_candidates processes OCR results from all pages.
        
        Regression test for: OCR only processed first page when raw_rows was empty.
        When a multi-page PDF has OCR results for pages 1, 2, and 3, all candidates
        should be extracted, not just from page 1.
        """
        ocr_df = pd.DataFrame(
            [
                {
                    "source_file": "multipage.pdf",
                    "source_type": "pdf",
                    "page": 1,
                    "row_id": 1,
                    "evidence_text": (
                        "Производство продукции растениеводства, тыс. тонн\n"
                        "Культура 2020 2021 2022 2023 2024\n"
                        "Пшеница 14256,1 11814,5 16404,2 12110,8 18450,0\n"
                        "Ячмень 3659,1 2377,2 3287,1 2590,5 3800,4\n"
                        "Источник: статистика\n"
                        "\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                    "section_name": "ocr_page_text",
                },
                {
                    "source_file": "multipage.pdf",
                    "source_type": "pdf",
                    "page": 2,
                    "row_id": 2,
                    "evidence_text": (
                        "Зерновые грузы 2020-2024 млн тонн\n"
                        "Наименование 2020 2021 2022 2023 2024\n"
                        "Пшеница 1250 1350 1480 1650 1850\n"
                        "Рожь 450 520 580 620 650\n"
                        "Ячмень 850 920 1050 1200 1350\n"
                        "Кукуруза 180 220 280 350 420\n"
                        "Всего 2730 3010 3390 3820 4270\n"
                        "Источник: статистика\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                    "section_name": "ocr_page_text",
                },
                {
                    "source_file": "multipage.pdf",
                    "source_type": "pdf",
                    "page": 3,
                    "row_id": 3,
                    "evidence_text": (
                        "Использование техники и оборудования\n"
                        "Металлопрокат тарифы 2024\n"
                        "Наименование Тариф руб/тонна Объем тонн\n"
                        "Листовой прокат 2500 15000\n"
                        "Трубный 3000 8500\n"
                        "Профиль 2800 12000\n"
                        "Арматура 2200 25000\n"
                        "Приложение 1: Дополнительные услуги\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                    "section_name": "ocr_page_text",
                },
            ]
        )

        candidates = extract_ocr_table_candidates(ocr_df)

        self.assertFalse(candidates.empty, "Candidates DataFrame should not be empty")
        
        pages_in_candidates = sorted(candidates["page"].unique().tolist())
        self.assertEqual(pages_in_candidates, [1, 2, 3], f"Expected candidates from pages [1, 2, 3], got {pages_in_candidates}")
        
        candidates_per_page = candidates.groupby("page").size().to_dict()
        self.assertGreaterEqual(candidates_per_page.get(1, 0), 1, "Should have at least one candidate from page 1")
        self.assertGreaterEqual(candidates_per_page.get(2, 0), 1, "Should have at least one candidate from page 2")
        self.assertGreaterEqual(candidates_per_page.get(3, 0), 1, "Should have at least one candidate from page 3")
        
        block_ids = candidates["ocr_block_id"].tolist()
        self.assertTrue(any("p1" in bid for bid in block_ids), "Block IDs should include page 1 reference")
        self.assertTrue(any("p2" in bid for bid in block_ids), "Block IDs should include page 2 reference")
        self.assertTrue(any("p3" in bid for bid in block_ids), "Block IDs should include page 3 reference")
        
        for idx, row in candidates.iterrows():
            self.assertIsNotNone(row["page"], f"Candidate {idx} should have page number")
            self.assertIn(row["candidate_type"], ["table", "paragraph", "mixed", "chart_text", "unknown"])
            self.assertGreater(row["rows_count"], 0, f"Candidate {idx} should have rows")
            self.assertGreater(row["numbers_count"], 0, f"Candidate {idx} should have numbers")
            self.assertGreater(row["score"], 0, f"Candidate {idx} should have positive score")

    def test_ocr_candidates_preserve_page_number_in_block_uid(self) -> None:
        """Verify that block_uid uniquely identifies blocks across pages.
        
        If two pages have identically named blocks, they should have different
        block_uids due to different page numbers.
        """
        ocr_df = pd.DataFrame(
            [
                {
                    "source_file": "test.pdf",
                    "source_type": "pdf",
                    "page": 1,
                    "row_id": 1,
                    "evidence_text": (
                        "Экспорт продуктов, млн долл\n"
                        "Год 2020 2021 2022 2023 2024\n"
                        "Пшеница 1000 1100 1200 1300 1400\n"
                        "Масло 900 950 1000 1050 1100\n"
                        "Источник: данные\n"
                        "\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                    "section_name": "ocr_page_text",
                },
                {
                    "source_file": "test.pdf",
                    "source_type": "pdf",
                    "page": 2,
                    "row_id": 2,
                    "evidence_text": (
                        "Экспорт продуктов, млн долл\n"
                        "Год 2020 2021 2022 2023 2024\n"
                        "Пшеница 2000 2200 2400 2600 2800\n"
                        "Масло 1800 1900 2000 2100 2200\n"
                        "Источник: данные\n"
                        "\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                    "section_name": "ocr_page_text",
                },
            ]
        )

        candidates = extract_ocr_table_candidates(ocr_df)
        
        self.assertEqual(len(candidates), 2, "Should have exactly 2 candidates")
        
        # Block UIDs should be different even though titles are the same
        block_uids = candidates["ocr_block_id"].tolist()
        self.assertNotEqual(block_uids[0], block_uids[1], 
                          f"Block UIDs should differ across pages, got {block_uids}")
        
        # Verify page info is encoded in block_uid
        self.assertIn("p1", block_uids[0], f"Page 1 block should have 'p1' in UID: {block_uids[0]}")
        self.assertIn("p2", block_uids[1], f"Page 2 block should have 'p2' in UID: {block_uids[1]}")

    def test_extract_ocr_table_candidates_creates_fallback_for_unstructured_pages(self) -> None:
        """Test that fallback candidates are created for pages without well-formed blocks.
        
        This simulates a real scenario where:
        - Page 1 has well-structured tabular data → creates normal block candidates
        - Page 2 and 3 have data but poor structure → should create fallback candidates
        
        This ensures that all pages with data are represented, even if extraction didn't find
        well-formed blocks.
        """
        ocr_df = pd.DataFrame(
            [
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 1,
                    "row_id": 1,
                    "evidence_text": (
                        "Таблица с хорошей структурой\n"
                        "Товар 2022 2023 2024 2025\n"
                        "Пшеница 100.5 150.3 200.8 250.1\n"
                        "Ячмень 50.2 60.5 80.3 100.0\n"
                        "Рожь 30.1 35.2 40.5 45.0\n"
                        "Кукуруза 25.5 30.0 35.5 40.0\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                },
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 2,
                    "row_id": 2,
                    "evidence_text": (
                        "Зерновые грузы по портам\n"
                        "Санкт-Петербург: тарифы 2500 руб/тонна контейнер объем 1500 тонн\n"
                        "Новороссийск: тарифы 2200 руб/тонна контейнер объем 2000 тонн\n"
                        "Владивосток: тарифы 3500 руб/тонна контейнер объем 800 тонн\n"
                        "Хранение на складе 50 руб/тонна сутки минимум 100 тонн\n"
                        "Подача вагонов 300 руб сутки авансовый платеж обязателен\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                },
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 3,
                    "row_id": 3,
                    "evidence_text": (
                        "Использование техники и оборудования\n"
                        "Погрузчик вилочный 200 руб час минимум 4 часа смены\n"
                        "Автотранспорт доставка 50 км 3500 руб за борт каждый км свыше 50 км 100 руб\n"
                        "Металлопрокат листовой 2000 руб тонна трубный прокат 2500 руб тонна\n"
                        "Арматура 1800 руб тонна уголок 1700 руб тонна швеллер 1900 руб тонна\n"
                        "Насыпные грузы сортировка и фасовка 150 руб тонна\n"
                        "Дополнительные услуги приложение 1 согласно тарифам 2024\n"
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                },
            ]
        )

        candidates = extract_ocr_table_candidates(ocr_df)

        self.assertFalse(candidates.empty, "Candidates should exist for all pages")
        
        pages = sorted(candidates["page"].unique().tolist())
        self.assertEqual(pages, [1, 2, 3], f"Expected candidates from all three pages, got {pages}")
        
        page_1_candidates = candidates[candidates["page"] == 1]
        page_2_candidates = candidates[candidates["page"] == 2]
        page_3_candidates = candidates[candidates["page"] == 3]
        
        self.assertGreater(len(page_1_candidates), 0, "Page 1 should have candidates (structured data)")
        self.assertGreater(len(page_2_candidates), 0, "Page 2 should have fallback candidates (tariffs/contracts)")
        self.assertGreater(len(page_3_candidates), 0, "Page 3 should have fallback candidates (equipment/services)")
        
        page_2_has_fallback = any("fallback" in uid for uid in page_2_candidates["ocr_block_id"])
        page_3_has_fallback = any("fallback" in uid for uid in page_3_candidates["ocr_block_id"])
        
        self.assertTrue(page_2_has_fallback, "Page 2 should have at least one fallback candidate")
        self.assertTrue(page_3_has_fallback, "Page 3 should have at least one fallback candidate")
        
        for idx, row in page_2_candidates.iterrows():
            self.assertIsNotNone(row["rows_count"])
            self.assertGreater(row["rows_count"], 0, f"Page 2 candidate {idx} should have rows")
            self.assertGreater(row["numbers_count"], 0, f"Page 2 candidate {idx} should have numbers")
        
        for idx, row in page_3_candidates.iterrows():
            self.assertIsNotNone(row["rows_count"])
            self.assertGreater(row["rows_count"], 0, f"Page 3 candidate {idx} should have rows")
            self.assertGreater(row["numbers_count"], 0, f"Page 3 candidate {idx} should have numbers")


