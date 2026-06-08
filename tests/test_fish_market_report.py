from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.complex_mapping import (
    TRADE_2023_2024_PRESET,
    apply_complex_mapping,
    build_mapping_preview,
    suggest_mapping_preset,
    token_mapping_from_preset,
)
from src.numeric_token_reconstruction import (
    FAILED_RECONSTRUCTION_WARNING,
    RECONSTRUCTION_WARNING,
    reconstruct_numeric_tokens,
)
from src.extract_pdf import extract_pdf
from src.coverage_summary import build_coverage_summary, coverage_counts
from src.document_profiles import PROFILE_PARSER_CONFIDENCE_THRESHOLD, detect_document_profile_with_confidence
from src.export import export_to_csv
from src.extract_ocr import (
    OCRLanguageError,
    OCRUnavailableError,
    TESSERACT_INSTALL_MESSAGE,
    configure_tesseract,
    extract_ocr,
    extract_ocr_pages,
    get_available_tesseract_languages,
    get_tesseract_cmd,
    is_language_available,
    is_tesseract_available,
)
from src.normalize import normalize_dataframe
from src.ocr_table_candidates import (
    analyze_ocr_candidate,
    detect_paragraph_like_block,
    extract_ocr_table_candidates,
    score_ocr_candidate,
)
from src.pdf_quality import detect_bad_text_layer
from src.profile_draft import build_profile_draft, dump_profile_draft_json, dump_profile_draft_yaml
from src.profile_parser_prototype import (
    COMPLEX_TRADE_WARNING,
    OCR_DECIMAL_WARNING,
    THOUSAND_TONS_LARGE_WARNING,
    normalize_ocr_number,
    parse_russian_number as parse_prototype_russian_number,
    parse_sections_from_draft,
)
from src.raw_table_analysis import build_raw_table_summary, score_raw_table
from src.parsers.fish_market_report import (
    parse_fish_market_report,
    parse_russian_number,
    split_multiline_table_row,
)
from src.source_registry import get_expected_sections, get_source_config, load_source_registry
from src.user_profile_builder import (
    apply_table_reconstruction,
    apply_user_profile,
    apply_user_profile_to_sources,
    build_profile_table_catalog,
    dump_user_profile_yaml,
    load_user_profile_file,
    normalize_user_number,
    profile_matches_document,
    save_user_profile,
    select_user_profile_export_columns,
    select_source_rows,
    select_source_rows_for_block_uids,
    source_rows_to_preview_df,
    source_row_uid,
)
from src.validate import validate_extracted_data
from app import (
    PROFILE_BUILDER_STEPS,
    apply_mapped_review_edits,
    audit_trail_coverage,
    build_processing_dashboard_summary,
    build_processing_funnel,
    count_strong_ocr_candidates,
    format_coverage_for_ui,
    format_ocr_candidates_for_ui,
    format_table_summary_for_ui,
    mapped_review_summary,
    mapped_review_editor_summary,
    ocr_candidate_diagnostics,
    prepare_profile_builder_catalog_editor,
    prepare_compact_review_editor_df,
    prepare_review_editor_df,
    prepare_reviewed_mapped_export,
    profile_builder_catalog_for_source,
    profile_builder_get_source,
    profile_builder_selected_block_uids_from_editor,
    profile_builder_source_state_key,
    profile_builder_source_widget_key,
    rename_columns_for_ui,
    review_editor_has_unsaved_changes,
    restore_mapped_review_original_values,
    restore_status_columns,
    select_mapped_rows_for_review,
    select_profile_candidate_export_columns,
    select_ocr_export_columns,
    select_ocr_candidate_export_columns,
    select_raw_export_columns,
    select_technical_raw_export_columns,
    split_rows_by_extraction_level,
    translate_status_columns,
)


class FishMarketReportParserTests(unittest.TestCase):
    def test_parse_russian_number_handles_dash(self) -> None:
        cases = {
            "1 947,4": 1947.4,
            "1106,6": 1106.6,
            "550,00": 550.0,
            "+ 33,3%": 33.3,
            "+33,3%": 33.3,
            "- 3,9%": -3.9,
            "-3,9%": -3.9,
            "0,0%": 0.0,
            "81,93": 81.93,
            "-": None,
        }
        for raw_value, expected in cases.items():
            self.assertEqual(parse_russian_number(raw_value), expected)

    def test_catch_block_with_dash_keeps_missing_metric(self) -> None:
        parsed = parse_fish_market_report(
            ["Рыбные ряды", "", "Лососевые*", "335,6 | - 44,9% | -"],
            source_file="report.pdf",
            page=1,
        )
        validated = validate_extracted_data(parsed)

        self.assertEqual(len(parsed), 3)
        volume = parsed[parsed["indicator"] == "catch_volume"].iloc[0]
        yoy = parsed[parsed["indicator"] == "yoy_change"].iloc[0]
        quota = parsed[parsed["indicator"] == "quota_utilization"].iloc[0]

        self.assertEqual(volume["commodity"], "Лососевые*")
        self.assertEqual(volume["value"], 335.6)
        self.assertEqual(yoy["value"], -44.9)
        self.assertTrue(quota.isna()["value"])
        self.assertEqual(quota["unit"], "percent")
        self.assertEqual(quota["evidence_text"], "Лососевые* | 335,6 | -44,9% | -")
        self.assertEqual(validated.loc[validated["indicator"] == "quota_utilization", "validation_status"].iloc[0], "warning")

    def test_split_multiline_table_row(self) -> None:
        rows = split_multiline_table_row(
            [
                "Скумбрия атлант.\nМинтай",
                "300,00\n174,00",
                "0,0%\n- 0,6%",
                "0,0%\n- 0,6%",
            ]
        )
        self.assertEqual(
            rows,
            [
                ["Скумбрия атлант.", "300,00", "0,0%", "0,0%"],
                ["Минтай", "174,00", "- 0,6%", "- 0,6%"],
            ],
        )

    def test_real_report_2026_06_01_still_extracts_75_rows(self) -> None:
        pdf_path = PROJECT_DIR / "data" / "examples" / "monitopring.rinka.ribi.na.01.06.2026.pdf"
        parsed = extract_pdf(str(pdf_path))
        validated = validate_extracted_data(normalize_dataframe(parsed))

        self.assertEqual(len(parsed), 75)
        self.assertEqual(set(parsed["extraction_level"]), {"structured"})
        self.assertEqual(set(validated["validation_status"]), {"passed"})
        metadata = parsed.attrs["profile_detection"]
        self.assertEqual(metadata["profile_name"], "fish_market_report")
        self.assertGreaterEqual(metadata["profile_confidence"], PROFILE_PARSER_CONFIDENCE_THRESHOLD)
        self.assertEqual(metadata["selected_extraction_strategy"], "profile_parser")

    def test_real_report_2026_01_12_extracts_center_and_salmon_rows(self) -> None:
        pdf_path = PROJECT_DIR / "data" / "examples" / "monitoring_12_01_2026.pdf"
        parsed = extract_pdf(str(pdf_path))
        validated = validate_extracted_data(normalize_dataframe(parsed))

        self.assertGreaterEqual(len(parsed), 74)
        center_prices = parsed[
            (parsed["region"] == "Центр")
            & (parsed["indicator"] == "wholesale_price")
        ][["commodity", "value", "unit"]]
        self.assertEqual(
            center_prices.to_records(index=False).tolist(),
            [
                ("Скумбрия атлант.", 300.0, "RUB/kg"),
                ("Минтай", 174.0, "RUB/kg"),
                ("Мойва атлант.", 210.0, "RUB/kg"),
                ("Сельдь атлант.", 140.0, "RUB/kg"),
            ],
        )

        salmon = parsed[parsed["commodity"] == "Лососевые*"]
        self.assertEqual(set(salmon["indicator"]), {"catch_volume", "yoy_change", "quota_utilization"})
        self.assertIn("warning", set(validated.loc[validated["commodity"] == "Лососевые*", "validation_status"]))

        coverage = build_coverage_summary(validated)
        found_count, total_count, missing_blocks = coverage_counts(coverage)
        self.assertEqual((found_count, total_count, missing_blocks), (6, 6, []))

        catch_coverage = coverage[coverage["section_name"] == "catch_main_species"].iloc[0]
        self.assertEqual(catch_coverage["actual_rows"], 15)
        self.assertEqual(catch_coverage["warning_rows"], 1)

        csv_bytes = export_to_csv(validated)
        self.assertIn("section_name", csv_bytes.decode("utf-8-sig").splitlines()[0])

    def test_detect_profile_requires_strong_fish_market_markers(self) -> None:
        strong = detect_document_profile_with_confidence(
            "Рыбные ряды\nОбзор ситуации на рынке рыбы\nНАЦРЫБРЕСУРС",
            "monitopring.rinka.ribi.na.01.06.2026.pdf",
        )
        weak = detect_document_profile_with_confidence(
            "В обзоре ВЭД упоминается рыба, рынок рыбы и рыбной продукции.",
            "obzor_ved_kazahstan_2025.pdf",
        )

        self.assertEqual(strong["profile_name"], "fish_market_report")
        self.assertGreaterEqual(strong["profile_confidence"], PROFILE_PARSER_CONFIDENCE_THRESHOLD)
        self.assertEqual(weak["profile_name"], "generic_pdf")
        self.assertLess(weak["profile_confidence"], PROFILE_PARSER_CONFIDENCE_THRESHOLD)

    def test_fish_parser_skips_rows_without_strong_profile_markers(self) -> None:
        parsed = parse_fish_market_report(
            ["Лососевые*", "335,6 | - 44,9% | -"],
            source_file="obzor_ved_kazahstan_2025.pdf",
            page=1,
        )

        self.assertTrue(parsed.empty)

    def test_score_raw_table_prioritizes_trade_tables_over_contents(self) -> None:
        useful = score_raw_table(
            [
                "Код ТН ВЭД | Товар | Экспорт, тыс. долл. | Импорт, тыс. долл.",
                "0303 | Рыба мороженая | 1250,4 | 830,1",
                "1604 | Готовая продукция | 520,0 | 118,3",
            ]
        )
        contents = score_raw_table(
            [
                "Содержание | Стр.",
                "Обзор рынка | 3",
                "Динамика | 4",
                "Контакты | 42",
            ]
        )

        self.assertGreaterEqual(useful["table_score"], 0.5)
        self.assertLess(contents["table_score"], 0.5)
        self.assertIn("числовые", useful["table_reason"])

    def test_bad_text_layer_detector_and_table_score_handle_cid_tokens(self) -> None:
        cid_text = " ".join(f"(cid:{index})" for index in range(40))
        quality = detect_bad_text_layer(cid_text)
        scored = score_raw_table(
            [
                "(cid:1473) | (cid:1474) | (cid:1475)",
                "(cid:1476) | (cid:1477) | (cid:1478)",
                "(cid:1479) | (cid:1480) | (cid:1481)",
            ]
        )

        self.assertTrue(quality["bad_text_layer"])
        self.assertGreater(quality["cid_ratio"], 0.05)
        self.assertLessEqual(scored["table_score"], 0.3)
        self.assertEqual(scored["text_layer_quality"], "bad")
        self.assertIn("OCR", scored["table_reason"])

    def test_obzor_ved_kazakhstan_uses_generic_pdf_in_auto_mode(self) -> None:
        pdf_path = PROJECT_DIR / "data" / "examples" / "obzor_ved_kazahstan_2025.pdf"
        parsed = extract_pdf(str(pdf_path))
        metadata = parsed.attrs["profile_detection"]
        coverage = build_coverage_summary(validate_extracted_data(normalize_dataframe(parsed)), profile=metadata["profile_name"])

        self.assertEqual(metadata["profile_name"], "generic_pdf")
        self.assertLess(metadata["profile_confidence"], PROFILE_PARSER_CONFIDENCE_THRESHOLD)
        self.assertEqual(metadata["selected_extraction_strategy"], "pdfplumber")
        self.assertNotIn("fish_market_report_parser", set(parsed["extraction_method"].dropna()))
        self.assertTrue(coverage.empty)

    def test_generic_pdf_rows_are_raw_extracted_not_failed(self) -> None:
        pdf_path = PROJECT_DIR / "data" / "examples" / "obzor_ved_kazahstan_2025.pdf"
        parsed = extract_pdf(str(pdf_path))
        validated = validate_extracted_data(normalize_dataframe(parsed))
        structured_rows, raw_rows = split_rows_by_extraction_level(validated)
        raw_export = select_raw_export_columns(raw_rows)
        technical_raw_export = select_technical_raw_export_columns(raw_rows)
        table_summary = build_raw_table_summary(raw_rows)
        ocr_tables = table_summary[table_summary["text_layer_quality"] == "bad"]
        candidate_export = select_profile_candidate_export_columns(
            table_summary[
                (table_summary["table_score"] >= 0.5)
                & (table_summary["text_layer_quality"] != "bad")
            ]
        )
        table_summary_ui = format_table_summary_for_ui(table_summary)
        metadata = parsed.attrs["profile_detection"]

        self.assertLess(len(parsed), 893)
        self.assertTrue(structured_rows.empty)
        self.assertEqual(len(raw_rows), len(validated))
        self.assertEqual(set(parsed["extraction_level"].dropna()), {"raw"})
        self.assertTrue(metadata["text_layer_quality"]["bad_text_layer"])
        self.assertIn("raw_page_text", set(parsed["section_name"].dropna()))
        self.assertIn("raw_pdf_table", set(parsed["section_name"].dropna()))
        self.assertIn("pdfplumber_text", set(parsed["extraction_method"].dropna()))
        self.assertIn("pdfplumber_table", set(parsed["extraction_method"].dropna()))
        self.assertTrue(parsed.loc[parsed["extraction_method"] == "pdfplumber_table", "table_id"].notna().all())
        self.assertTrue(parsed.loc[parsed["extraction_method"] == "pdfplumber_table", "row_index_in_table"].notna().all())
        self.assertEqual(set(validated["text_layer_quality"].dropna()), {"bad"})
        self.assertEqual(set(validated["validation_status"].dropna()), {"raw_extracted"})
        self.assertEqual(set(validated["review_status"].dropna()), {"needs_ocr"})
        self.assertNotIn("failed", set(validated["validation_status"].dropna()))
        self.assertFalse(table_summary.empty)
        self.assertIn("table_id", table_summary.columns)
        self.assertIn("table_score", table_summary.columns)
        self.assertIn("text_layer_quality", table_summary.columns)
        self.assertIn("table_id", table_summary_ui.columns)
        self.assertFalse(ocr_tables.empty)
        self.assertTrue((ocr_tables["table_score"] <= 0.3).all())
        self.assertNotIn("bad", set(candidate_export.get("text_layer_quality", [])))
        self.assertEqual(
            list(technical_raw_export.columns),
            [
                "source_file",
                "source_type",
                "page",
                "section_name",
                "table_id",
                "row_id",
                "row_index_in_table",
                "evidence_text",
                "extraction_method",
                "extraction_level",
                "text_layer_quality",
                "text_layer_warning",
                "confidence",
                "table_score",
                "table_reason",
                "validation_status",
                "review_status",
            ],
        )
        self.assertIn("Файл", raw_export.columns)
        self.assertIn("Статус проверки", raw_export.columns)
        if not candidate_export.empty:
            self.assertEqual(
                list(candidate_export.columns),
                [
                    "source_file",
                    "table_id",
                    "page",
                    "table_score",
                    "table_reason",
                    "preview",
                    "raw_rows_count",
                ],
            )

    def test_manual_fish_profile_can_run_profile_parser(self) -> None:
        pdf_path = PROJECT_DIR / "data" / "examples" / "monitoring_12_01_2026.pdf"
        parsed = extract_pdf(str(pdf_path), profile_override="fish_market_report")
        metadata = parsed.attrs["profile_detection"]

        self.assertEqual(metadata["profile_name"], "fish_market_report")
        self.assertEqual(metadata["profile_selection"], "manual")
        self.assertEqual(metadata["selected_extraction_strategy"], "profile_parser")
        self.assertIn("fish_market_report_parser", set(parsed["extraction_method"].dropna()))

    def test_ocr_fallback_placeholder_returns_standard_empty_dataframe(self) -> None:
        ocr_df = extract_ocr(str(PROJECT_DIR / "data" / "examples" / "obzor_ved_kazahstan_2025.pdf"))

        self.assertTrue(ocr_df.empty)
        self.assertIn("source_file", ocr_df.columns)
        self.assertIn("text_layer_quality", ocr_df.columns)

    def test_ocr_page_extraction_contract_without_running_tesseract(self) -> None:
        empty_ocr = extract_ocr_pages(
            str(PROJECT_DIR / "data" / "examples" / "obzor_ved_kazahstan_2025.pdf"),
            pages=[],
        )
        self.assertTrue(empty_ocr.empty)
        self.assertIsInstance(is_tesseract_available(), bool)

        synthetic = pd.DataFrame(
            [
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 5,
                    "section_name": "ocr_page_text",
                    "evidence_text": "OCR text",
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                    "confidence": 0.6,
                    "text_layer_quality": "ocr",
                },
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 6,
                    "section_name": "ocr_page_text",
                    "evidence_text": None,
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                    "confidence": 0.2,
                    "text_layer_quality": "ocr",
                },
            ]
        )
        validated = validate_extracted_data(synthetic)
        structured_rows, raw_rows = split_rows_by_extraction_level(validated)
        ocr_export = select_ocr_export_columns(validated)

        self.assertTrue(structured_rows.empty)
        self.assertEqual(len(raw_rows), 2)
        self.assertEqual(set(validated["validation_status"]), {"raw_extracted"})
        self.assertIn("needs_profile_setup", set(validated["review_status"]))
        self.assertIn("manual_required", set(validated["review_status"]))
        self.assertEqual(
            list(ocr_export.columns),
            [
                "source_file",
                "source_type",
                "page",
                "section_name",
                "evidence_text",
                "extraction_method",
                "extraction_level",
                "confidence",
                "validation_status",
                "review_status",
            ],
        )

    def test_extract_ocr_table_candidates_finds_table_like_blocks(self) -> None:
        ocr_df = pd.DataFrame(
            [
                {
                    "source_file": "obzor_ved_kazahstan_2025.pdf",
                    "page": 5,
                    "row_id": 1,
                    "evidence_text": (
                        "Производство продукции растениеводства, тыс. тонн\n"
                        "Культура 2020 2021 2022 2023 2024\n"
                        "Пшеница 14256,1 11814,5 16404,2 12110,8 18450,0\n"
                        "Ячмень 3659,1 2377,2 3287,1 2590,5 3800,4\n"
                        "Картофель 4006,8 4031,7 4080,2 4111,0 4200,3\n"
                        "Источник: БНС АСПР РК\n"
                        "\n"
                        "Обычный текст страницы без таблицы."
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                }
            ]
        )

        candidates = extract_ocr_table_candidates(ocr_df)
        candidate_export = select_ocr_candidate_export_columns(candidates)
        candidate_ui = format_ocr_candidates_for_ui(candidates)

        self.assertFalse(candidates.empty)
        self.assertEqual(candidates.loc[0, "block_title"], "Производство продукции растениеводства, тыс. тонн")
        self.assertEqual(candidates.loc[0, "candidate_type"], "table")
        self.assertEqual(candidates.loc[0, "extraction_method"], "tesseract_ocr_candidate")
        self.assertEqual(candidates.loc[0, "extraction_level"], "ocr_candidate")
        self.assertEqual(candidates.loc[0, "review_status"], "needs_profile_setup")
        self.assertGreaterEqual(candidates.loc[0, "table_score"], 0.6)
        self.assertGreaterEqual(candidates.loc[0, "information_score"], 0.6)
        self.assertGreaterEqual(candidates.loc[0, "score"], 0.6)
        self.assertGreaterEqual(candidates.loc[0, "rows_count"], 5)
        self.assertGreaterEqual(candidates.loc[0, "numbers_count"], 8)
        self.assertIn("похоже на таблицу", candidates.loc[0, "reason"])
        self.assertIn("годы", candidates.loc[0, "reason"])
        self.assertIn("\n", candidates.loc[0, "preview"])
        self.assertLessEqual(len(candidates.loc[0, "preview"].splitlines()), 8)
        self.assertIn("Заголовок блока", candidate_ui.columns)
        self.assertIn("Тип кандидата", candidate_ui.columns)
        self.assertIn("Table score", candidate_ui.columns)
        self.assertEqual(
            list(candidate_export.columns),
            [
                "source_file",
                "page",
                "ocr_block_id",
                "block_title",
                "candidate_type",
                "block_text",
                "preview",
                "rows_count",
                "numbers_count",
                "table_score",
                "information_score",
                "reason",
                "review_status",
            ],
        )

    def test_score_ocr_candidate_penalizes_plain_paragraphs(self) -> None:
        strong_score, strong_reason = score_ocr_candidate(
            "Импорт Казахстана из России, 2023-2024 гг.\n"
            "Товар 2023 2024\n"
            "Пшеница 10,5 12,1\n"
            "Масло 5,0 6,3\n"
            "Итого млн долл США 15,5 18,4"
        )
        weak_score, weak_reason = score_ocr_candidate(
            "Это обычный абзац с кратким описанием ситуации на рынке без устойчивой табличной структуры."
        )

        self.assertGreaterEqual(strong_score, 0.6)
        self.assertIn("импорт", strong_reason)
        self.assertLessEqual(weak_score, 0.2)
        self.assertIn("обычный абзац", weak_reason)

    def test_ocr_candidate_scoring_keeps_informative_paragraphs_out_of_best_tables(self) -> None:
        paragraph_analysis = analyze_ocr_candidate(
            "Импорт продукции АПК\n"
            "Импорт продукции АПК Казахстана в значительной степени диверсифицирован. "
            "В 2024 г. на топ-3 страны пришлось 36,0% импорта, при этом поставки "
            "распределены между несколькими группами товаров и рынками.\n"
            "В топ-3 стран-экспортёров продуктов питания вошли Россия, Китай и Беларусь."
        )

        self.assertEqual(paragraph_analysis["candidate_type"], "paragraph")
        self.assertGreaterEqual(paragraph_analysis["information_score"], 0.6)
        self.assertLess(paragraph_analysis["table_score"], 0.6)
        self.assertIn("информативный текстовый блок", paragraph_analysis["reason"])

    def test_retail_trade_paragraph_does_not_get_high_table_score(self) -> None:
        block_text = (
            "Розничная торговля\n"
            "В 2024 г. оборот розничной торговли продовольственными товарами вырос на 8,4%, "
            "при этом наибольший вклад внесли крупные города и приграничные регионы.\n"
            "По сравнению с 2023 г. продажи отдельных категорий увеличились на 6,1%, "
            "а доля импортной продукции в ассортименте торговых сетей оставалась высокой.\n"
            "В то же время динамика спроса зависела от доходов населения, сезонности, "
            "логистических расходов и изменения потребительских предпочтений."
        )

        paragraph_metrics = detect_paragraph_like_block(block_text)
        analysis = analyze_ocr_candidate(block_text)

        self.assertTrue(paragraph_metrics["is_paragraph_like"])
        self.assertIn(analysis["candidate_type"], ["paragraph", "mixed"])
        self.assertLessEqual(analysis["table_score"], 0.45)
        self.assertGreaterEqual(analysis["information_score"], 0.7)
        self.assertIn("связный текст", analysis["reason"])

    def test_country_exporters_table_gets_high_table_score(self) -> None:
        analysis = analyze_ocr_candidate(
            "Основные страны-экспортеры продукции АПК в Казахстан, 2020-2024 гг., млн долл. США\n"
            "1 Россия 2072.4 2678.2 3207.1 3078.1 3321.8\n"
            "2 Китай 158.3 165.2 257.4 381.8 388.3\n"
            "3 Беларусь 116.4 139.8 144.2 188.6 205.7"
        )

        self.assertEqual(analysis["candidate_type"], "table")
        self.assertGreaterEqual(analysis["table_score"], 0.75)
        self.assertIn("похоже на таблицу", analysis["reason"])

    def test_diversified_import_sentence_is_paragraph_candidate(self) -> None:
        analysis = analyze_ocr_candidate(
            "Импорт продукции АПК Казахстана в значительной степени диверсифицирован.\n"
            "В 2024 г. поставки распределялись между несколькими товарными группами и странами, "
            "а доля топ-3 направлений не формировала устойчивую табличную структуру.\n"
            "Такие сведения важны для аналитического обзора, но строки являются связным текстом."
        )

        self.assertEqual(analysis["candidate_type"], "paragraph")
        self.assertLessEqual(analysis["table_score"], 0.45)
        self.assertIn("информативный текстовый блок", analysis["reason"])

    def test_extract_ocr_candidates_keeps_paragraphs_but_not_as_strong_tables(self) -> None:
        ocr_df = pd.DataFrame(
            [
                {
                    "source_file": "obzor_ved_kazahstan_2025.pdf",
                    "page": 9,
                    "row_id": 1,
                    "evidence_text": (
                        "Импорт продукции АПК\n"
                        "Импорт продукции АПК Казахстана в значительной степени диверсифицирован. "
                        "В 2024 г. на топ-3 страны пришлось 36,0% импорта, "
                        "а остальные поставки распределены между другими направлениями.\n"
                        "В топ-3 стран-экспортёров продуктов питания вошли Россия, Китай и Беларусь."
                    ),
                    "extraction_method": "tesseract_ocr",
                    "extraction_level": "raw_ocr",
                }
            ]
        )

        candidates = extract_ocr_table_candidates(ocr_df)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.loc[0, "candidate_type"], "paragraph")
        self.assertGreaterEqual(candidates.loc[0, "information_score"], 0.6)
        self.assertLess(candidates.loc[0, "table_score"], 0.6)
        self.assertEqual(count_strong_ocr_candidates(candidates), 0)

    def test_ocr_candidate_diagnostics_reports_saved_candidates(self) -> None:
        ocr_result_df = pd.DataFrame(
            [
                {"page": 7, "evidence_text": "Производство продукции растениеводства"},
                {"page": 8, "evidence_text": "Производство продукции животноводства"},
            ]
        )
        ocr_candidates_df = pd.DataFrame(
            [
                {"block_title": "Производство продукции растениеводства", "score": 0.7},
                {"block_title": "Производство продукции животноводства", "score": 0.7},
                {"block_title": "Слабый кандидат", "score": 0.4},
            ]
        )

        diagnostics = ocr_candidate_diagnostics(ocr_result_df, ocr_candidates_df)

        self.assertEqual(count_strong_ocr_candidates(ocr_candidates_df), 2)
        self.assertEqual(diagnostics["ocr_rows_count"], 2)
        self.assertEqual(diagnostics["ocr_candidates_count"], 3)
        self.assertEqual(diagnostics["max_table_score"], 0.7)
        self.assertEqual(diagnostics["strong_table_candidates"], 2)
        self.assertEqual(
            diagnostics["first_5_block_titles"][:2],
            ["Производство продукции растениеводства", "Производство продукции животноводства"],
        )

    def test_build_profile_draft_from_selected_ocr_candidates(self) -> None:
        selected_candidates = pd.DataFrame(
            [
                {
                    "source_file": "obzor_ved_kazahstan_2025.pdf",
                    "page": 11,
                    "block_title": "Основные страны-экспортеры продукции АПК в Казахстан, 2020-2024 гг., млн долл. США",
                    "candidate_type": "table",
                    "table_score": 0.92,
                    "information_score": 0.95,
                    "preview": (
                        "1 Россия 2072.4 2678.2 3207.1 3078.1 3321.8\n"
                        "2 Китай 158.3 165.2 257.4 381.8 388.3"
                    ),
                },
                {
                    "source_file": "obzor_ved_kazahstan_2025.pdf",
                    "page": 12,
                    "block_title": "Товарная структура импорта продукции АПК Казахстана, 2024 г.",
                    "candidate_type": "table",
                    "table_score": 0.88,
                    "information_score": 0.9,
                    "preview": "1 Пшеница 120.0 18,2%\n2 Масло 92.0 14,0%",
                },
            ]
        )

        draft = build_profile_draft(
            source_file="obzor_ved_kazahstan_2025.pdf",
            selected_candidates_df=selected_candidates,
            profile_name="agro_kazakhstan_review",
        )

        self.assertEqual(draft["profile_name"], "agro_kazakhstan_review")
        self.assertEqual(draft["document_profile"], "agro_kazakhstan_review")
        self.assertTrue(draft["requires_ocr"])
        self.assertEqual(draft["extraction_strategy"], "ocr_profile_parser")
        self.assertEqual(draft["metadata"]["created_from"], "ocr_candidates")
        self.assertEqual(draft["profile_draft_summary"]["total_sections"], 2)
        self.assertEqual(draft["profile_draft_summary"]["good_sections"], 2)
        self.assertFalse(draft["profile_draft_summary"]["has_warnings"])
        self.assertTrue(draft["profile_draft_summary"]["ready_for_parser_prototype"])
        self.assertTrue(draft["profile_draft_summary"]["requires_developer_review"])
        self.assertEqual(len(draft["target_sections"]), 2)
        first_section = draft["target_sections"][0]
        second_section = draft["target_sections"][1]
        self.assertEqual(first_section["section_id"], "main_export_countries")
        self.assertEqual(first_section["section_quality"], "good")
        self.assertGreaterEqual(first_section["section_confidence"], 0.9)
        self.assertEqual(first_section["section_warnings"], [])
        self.assertEqual(first_section["unit_hint"], "млн долл. США")
        self.assertEqual(
            first_section["expected_fields"],
            ["rank", "name", "value_2020", "value_2021", "value_2022", "value_2023", "value_2024"],
        )
        self.assertEqual(second_section["section_id"], "import_commodity_structure")
        self.assertEqual(second_section["expected_fields"], ["rank", "commodity", "value", "share_pct"])
        self.assertIn("OCR table-like block", first_section["parser_hint"])
        self.assertIn("numeric columns must be parseable", first_section["validation_rules"])

    def test_profile_draft_serializers_include_metadata_and_sections(self) -> None:
        draft = build_profile_draft(
            source_file="obzor_ved_kazahstan_2025.pdf",
            selected_candidates_df=pd.DataFrame(
                [
                    {
                        "page": 7,
                        "block_title": "Производство продукции растениеводства, тыс. тонн",
                        "candidate_type": "table",
                        "table_score": 0.85,
                        "information_score": 0.9,
                        "preview": "Пшеница 14256,1 11814,5 16404,2 12110,8 18450,0",
                    }
                ]
            ),
        )

        yaml_text = dump_profile_draft_yaml(draft)
        json_text = dump_profile_draft_json(draft)

        self.assertIn("profile_name:", yaml_text)
        self.assertIn("profile_draft_summary:", yaml_text)
        self.assertIn("section_quality:", yaml_text)
        self.assertIn("crop_production", yaml_text)
        self.assertIn("draft profile, requires developer review", yaml_text)
        self.assertIn('"profile_draft_summary"', json_text)
        self.assertIn('"target_sections"', json_text)
        self.assertIn("crop_production", json_text)

    def _mock_user_profile_config(self) -> dict:
        return {
            "profile_name": "user_tariffs_test",
            "display_name": "User tariffs test",
            "document_match": {
                "keywords": ["Зерно, бобовые и семена", "Тариф в рублях РФ"],
            },
            "tables": [
                {
                    "section_name": "tariffs",
                    "table_selector": {"text_contains": "Зерно"},
                    "row_filters": [
                        {"type": "keep_after", "text": "Зерно, бобовые и семена"},
                        {"type": "skip_empty_code"},
                        {"type": "skip_empty_value_columns"},
                        {"type": "skip_dash_values"},
                    ],
                    "column_mapping": {
                        "column_1": {"role": "code"},
                        "column_2": {"role": "name"},
                        "column_3": {"role": "unit"},
                        "column_4": {
                            "role": "value_direct",
                            "metric": "tariff",
                            "tariff_type": "direct",
                            "currency": "RUB",
                            "value_type": "numeric",
                        },
                        "column_5": {
                            "role": "value_intraport",
                            "metric": "tariff",
                            "tariff_type": "intraport_movement",
                            "currency": "RUB",
                            "value_type": "numeric",
                        },
                    },
                }
            ],
            "normalization": {"number_format": "ru"},
            "validation": {"required_fields": ["name", "value", "unit"], "value_positive": True},
        }

    def _mock_user_profile_raw_rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_file": "tariffs.pdf",
                    "page": 2,
                    "row_id": 1,
                    "table_id": "page_2_table_1",
                    "row_index_in_table": 1,
                    "evidence_text": "Тариф в рублях РФ | прямой вариант | внутрипортовое перемещение",
                    "extraction_method": "pdfplumber_table",
                    "text_layer_quality": "ok",
                },
                {
                    "source_file": "tariffs.pdf",
                    "page": 2,
                    "row_id": 2,
                    "table_id": "page_2_table_1",
                    "row_index_in_table": 2,
                    "evidence_text": "1 Зерно, бобовые и семена",
                    "extraction_method": "pdfplumber_table",
                    "text_layer_quality": "ok",
                },
                {
                    "source_file": "tariffs.pdf",
                    "page": 2,
                    "row_id": 3,
                    "table_id": "page_2_table_1",
                    "row_index_in_table": 3,
                    "evidence_text": "1.1 | Пшеница | т | 1 729,00 | 919,00",
                    "extraction_method": "pdfplumber_table",
                    "text_layer_quality": "ok",
                },
                {
                    "source_file": "tariffs.pdf",
                    "page": 2,
                    "row_id": 4,
                    "table_id": "page_2_table_1",
                    "row_index_in_table": 4,
                    "evidence_text": "1.2 | Рожь | т | - | -",
                    "extraction_method": "pdfplumber_table",
                    "text_layer_quality": "ok",
                },
            ]
        )

    def test_user_profile_config_serializes_and_deserializes(self) -> None:
        config = self._mock_user_profile_config()
        yaml_text = dump_user_profile_yaml(config)
        self.assertIn("profile_name", yaml_text)

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_path = save_user_profile(config, Path(tmp_dir))
            loaded = load_user_profile_file(saved_path)

        self.assertEqual(loaded["profile_name"], "user_tariffs_test")
        self.assertEqual(loaded["tables"][0]["column_mapping"]["column_4"]["tariff_type"], "direct")

    def test_user_profile_column_mapping_and_wide_to_long(self) -> None:
        structured = apply_user_profile_to_sources(
            self._mock_user_profile_raw_rows(),
            pd.DataFrame(),
            self._mock_user_profile_config(),
        )

        self.assertEqual(len(structured), 2)
        self.assertEqual(list(structured["tariff_type"]), ["direct", "intraport_movement"])
        self.assertEqual(list(structured["value"]), [1729.0, 919.0])
        self.assertEqual(set(structured["unit"]), {"ton"})
        self.assertEqual(set(structured["currency"]), {"RUB"})
        self.assertEqual(set(structured["extraction_method"]), {"user_profile_parser"})
        self.assertEqual(set(structured["profile_name"]), {"user_tariffs_test"})
        self.assertTrue(structured["evidence_text"].str.contains("Пшеница").all())

    def test_user_profile_dash_values_do_not_create_rows(self) -> None:
        structured = apply_user_profile_to_sources(
            self._mock_user_profile_raw_rows().iloc[[1, 3]].copy(),
            pd.DataFrame(),
            self._mock_user_profile_config(),
        )

        self.assertTrue(structured.empty)

    def test_user_profile_normalizes_russian_numbers_and_keep_after(self) -> None:
        self.assertEqual(normalize_user_number("1 729,00"), 1729.0)
        self.assertEqual(normalize_user_number("-3,9%", value_type="percent"), -3.9)

        structured = apply_user_profile_to_sources(
            self._mock_user_profile_raw_rows(),
            pd.DataFrame(),
            self._mock_user_profile_config(),
        )

        self.assertEqual(len(structured), 2)
        self.assertNotIn("Зерно, бобовые и семена", set(structured["name"].fillna("").astype(str)))

    def test_saved_user_profile_matches_and_exports_audit_fields(self) -> None:
        config = self._mock_user_profile_config()
        raw_rows = self._mock_user_profile_raw_rows()

        self.assertTrue(profile_matches_document(config, raw_rows, pd.DataFrame()))
        structured = apply_user_profile_to_sources(raw_rows, pd.DataFrame(), config)
        export_df = select_user_profile_export_columns(structured)

        for column in [
            "source_file",
            "page",
            "table_id",
            "evidence_text",
            "extraction_method",
            "profile_name",
            "validation_status",
            "review_status",
        ]:
            self.assertIn(column, export_df.columns)
        self.assertEqual(set(export_df["validation_status"]), {"passed"})

    def test_user_profile_row_selection_config_serializes(self) -> None:
        config = self._mock_user_profile_config()
        config["tables"][0]["row_filters"] = {
            "use_manual_rows": True,
            "selected_source_rows": [3, 4],
            "skip_empty_values": True,
            "skip_dash_values": True,
            "keep_numeric_rows_only": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_path = save_user_profile(config, Path(tmp_dir))
            loaded = load_user_profile_file(saved_path)

        row_filters = loaded["tables"][0]["row_filters"]
        self.assertTrue(row_filters["use_manual_rows"])
        self.assertEqual(row_filters["selected_source_rows"], [3, 4])
        self.assertTrue(row_filters["skip_dash_values"])

    def test_user_profile_manual_selected_rows_parse_only_selected_rows(self) -> None:
        config = self._mock_user_profile_config()
        config["tables"][0]["row_filters"] = {
            "use_manual_rows": True,
            "selected_source_rows": [3],
            "skip_empty_values": True,
            "skip_dash_values": True,
        }
        raw_rows = pd.concat(
            [
                self._mock_user_profile_raw_rows(),
                pd.DataFrame(
                    [
                        {
                            "source_file": "tariffs.pdf",
                            "page": 2,
                            "row_id": 5,
                            "table_id": "page_2_table_1",
                            "row_index_in_table": 5,
                            "evidence_text": "1.3 | Масло растительное | т | 100,00 | 200,00",
                            "extraction_method": "pdfplumber_table",
                            "text_layer_quality": "ok",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        structured = apply_user_profile_to_sources(raw_rows, pd.DataFrame(), config)

        self.assertEqual(len(structured), 2)
        self.assertEqual(set(structured["name"]), {"Пшеница"})

    def test_user_profile_table_reconstruction_split_by_regex(self) -> None:
        source_rows = [
            {
                "source_row_id": 1,
                "row_index_in_table": 1,
                "evidence_text": "1.1  Пшеница  т  1 729,00  919,00",
                "cells": ["1.1  Пшеница  т  1 729,00  919,00"],
            }
        ]

        rebuilt_rows = apply_table_reconstruction(
            source_rows,
            {"method": "split_by_regex", "pattern": r"\s{2,}"},
        )

        self.assertEqual(rebuilt_rows[0]["cells"], ["1.1", "Пшеница", "т", "1 729,00", "919,00"])

    def _mock_user_profile_blocks_config(self, *, source: str = "pdf_text_layer") -> dict:
        extraction = {
            "source": source,
            "pdf_engine": "pdfplumber",
            "ocr": {"required": False, "engine": None, "lang": None},
        }
        source_kind = "pdf_table"
        if source == "ocr":
            extraction = {
                "source": "ocr",
                "ocr": {"required": True, "engine": "tesseract", "lang": "rus+eng", "pages": "auto", "dpi": 300},
            }
            source_kind = "ocr_candidate"
        elif source == "mixed":
            extraction = {
                "source": "mixed",
                "primary": "pdfplumber",
                "fallback": "tesseract",
                "ocr": {"required": False, "engine": "tesseract", "lang": "rus+eng", "pages": "auto", "dpi": 300},
            }
            source_kind = "pdf_table"
        return {
            "profile_name": f"user_tariffs_{source}",
            "display_name": f"User tariffs {source}",
            "document_match": {"keywords": ["Тариф"]},
            "extraction": extraction,
            "blocks": [
                {
                    "source_kind": source_kind,
                    "selector": {"table_ids": ["page_2_table_1" if source != "ocr" else "ocr_1"]},
                    "row_selection": {
                        "mode": "manual",
                        "use_manual_rows": True,
                        "selected_source_rows": ["raw_table:page_2_table_1:3" if source != "ocr" else "ocr_candidate:ocr_1:1"],
                        "skip_empty_values": True,
                        "skip_dash_values": True,
                    },
                    "column_mapping": {
                        "column_1": {"role": "code"},
                        "column_2": {"role": "name"},
                        "column_3": {"role": "unit"},
                        "column_4": {"role": "value", "metric": "tariff", "scenario": "direct", "currency": "RUB"},
                        "column_5": {"role": "value", "metric": "tariff", "scenario": "intraport_movement", "currency": "RUB"},
                    },
                }
            ],
            "validation": {"required_fields": ["name", "value"], "value_positive": True},
        }

    def test_user_profile_stable_row_uid_survives_rebuild(self) -> None:
        raw_rows = self._mock_user_profile_raw_rows()
        row_a = {
            "source_kind": "raw_table",
            "table_id": raw_rows.iloc[2]["table_id"],
            "row_index_in_table": raw_rows.iloc[2]["row_index_in_table"],
            "source_row_id": raw_rows.iloc[2]["row_id"],
            "evidence_text": raw_rows.iloc[2]["evidence_text"],
        }
        row_b = dict(row_a)
        row_b["source_row_id"] = 999

        self.assertEqual(source_row_uid(row_a), source_row_uid(row_b))

    def test_user_profile_source_and_output_row_counts_are_separate(self) -> None:
        config = self._mock_user_profile_blocks_config()
        result = apply_user_profile({"raw_rows": self._mock_user_profile_raw_rows()}, config)

        structured = result["structured_rows"]
        self.assertEqual(config["blocks"][0]["row_selection"]["selected_source_rows"], ["raw_table:page_2_table_1:3"])
        self.assertEqual(len(config["blocks"][0]["row_selection"]["selected_source_rows"]), 1)
        self.assertEqual(len(structured), 2)
        self.assertEqual(set(structured["scenario"]), {"direct", "intraport_movement"})

    def test_user_profile_ocr_extraction_branch_uses_runner_and_settings(self) -> None:
        config = self._mock_user_profile_blocks_config(source="ocr")
        calls: list[dict] = []

        def mock_ocr_runner(_document, ocr_config):
            calls.append(dict(ocr_config))
            return pd.DataFrame(
                [
                    {
                        "source_file": "tariffs.pdf",
                        "page": 2,
                        "ocr_block_id": "ocr_1",
                        "block_text": "1.1 | Пшеница | т | 1 729,00 | 919,00",
                        "extraction_method": "tesseract_ocr",
                    }
                ]
            )

        result = apply_user_profile({}, config, ocr_runner=mock_ocr_runner)

        self.assertTrue(result["ocr_ran"])
        self.assertEqual(calls[0]["engine"], "tesseract")
        self.assertEqual(calls[0]["lang"], "rus+eng")
        self.assertEqual(len(result["structured_rows"]), 2)
        self.assertEqual(set(result["structured_rows"]["source_kind"]), {"ocr_candidate"})

    def test_user_profile_ocr_block_uid_is_stable_after_rebuild(self) -> None:
        candidate = pd.DataFrame(
            [
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 17,
                    "ocr_block_id": "",
                    "block_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                    "block_text": "Пшеница 1 2 3 4\nЯчмень 5 6 7 8",
                    "preview": "Пшеница 1 2 3 4\nЯчмень 5 6 7 8",
                    "extraction_method": "tesseract_ocr_candidate",
                }
            ]
        )

        first_catalog = build_profile_table_catalog(pd.DataFrame(), None, candidate)
        second_catalog = build_profile_table_catalog(pd.DataFrame(), None, candidate.copy())

        self.assertEqual(first_catalog.loc[0, "block_uid"], second_catalog.loc[0, "block_uid"])
        self.assertTrue(str(first_catalog.loc[0, "block_uid"]).startswith("ocr_candidate:17:"))

    def test_user_profile_ocr_row_uid_matches_rows_and_preview(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 17,
                    "ocr_block_id": "ocr_p17_b1",
                    "block_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                    "block_text": "Пшеница 1 2 3 4\nЯчмень 5 6 7 8",
                    "preview": "Пшеница 1 2 3 4\nЯчмень 5 6 7 8",
                    "extraction_method": "tesseract_ocr_candidate",
                }
            ]
        )
        preview = source_rows_to_preview_df([], limit=10)
        self.assertIn("row_uid", preview.columns)

        rows = select_source_rows(pd.DataFrame(), candidates, {"block_uids": ["ocr_candidate:17:ocr_p17_b1"]})
        preview = source_rows_to_preview_df(rows, limit=10)

        self.assertEqual(rows[0]["row_uid"], preview.loc[0, "row_uid"])
        self.assertEqual(rows[0]["row_uid"], "ocr_candidate:17:ocr_p17_b1:row:1")

    def test_user_profile_ocr_block_selection_changes_keep_catalog(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 17,
                    "ocr_block_id": "ocr_p17_b1",
                    "block_title": "Export A",
                    "block_text": "Wheat 1 2 3 4",
                    "preview": "Wheat 1 2 3 4",
                    "extraction_method": "tesseract_ocr_candidate",
                },
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 18,
                    "ocr_block_id": "ocr_p18_b2",
                    "block_title": "Export B",
                    "block_text": "Barley 5 6 7 8",
                    "preview": "Barley 5 6 7 8",
                    "extraction_method": "tesseract_ocr_candidate",
                },
            ]
        )
        catalog = build_profile_table_catalog(pd.DataFrame(), None, candidates)
        block_a = "ocr_candidate:17:ocr_p17_b1"
        block_b = "ocr_candidate:18:ocr_p18_b2"

        rows_a = select_source_rows_for_block_uids(pd.DataFrame(), candidates, [block_a])
        rows_after_clear = select_source_rows_for_block_uids(pd.DataFrame(), candidates, [])
        catalog_after_clear = build_profile_table_catalog(pd.DataFrame(), None, candidates)
        rows_b = select_source_rows_for_block_uids(pd.DataFrame(), candidates, [block_b])

        self.assertEqual(catalog["block_uid"].tolist(), [block_a, block_b])
        self.assertEqual(catalog_after_clear["block_uid"].tolist(), [block_a, block_b])
        self.assertEqual([row["block_uid"] for row in rows_a], [block_a])
        self.assertEqual(rows_after_clear, [])
        self.assertEqual([row["block_uid"] for row in rows_b], [block_b])
        self.assertEqual(len(candidates), 2)

    def test_profile_builder_apply_block_selection_keeps_ocr_catalog(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 1,
                    "ocr_block_id": "ocr_p1_b1",
                    "block_title": "A",
                    "block_text": "Wheat 1 2",
                    "preview": "Wheat 1 2",
                    "extraction_method": "tesseract_ocr_candidate",
                },
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 1,
                    "ocr_block_id": "ocr_p1_b2",
                    "block_title": "B",
                    "block_text": "Barley 3 4",
                    "preview": "Barley 3 4",
                    "extraction_method": "tesseract_ocr_candidate",
                },
            ]
        )
        catalog = build_profile_table_catalog(pd.DataFrame(), None, candidates)
        block_a = "ocr_candidate:1:ocr_p1_b1"
        block_b = "ocr_candidate:1:ocr_p1_b2"

        editor_a = prepare_profile_builder_catalog_editor(catalog, [block_a])
        selected_a = profile_builder_selected_block_uids_from_editor(editor_a, catalog)
        editor_b = prepare_profile_builder_catalog_editor(catalog, [block_b])
        selected_b = profile_builder_selected_block_uids_from_editor(editor_b, catalog)
        editor_none = prepare_profile_builder_catalog_editor(catalog, [])
        selected_none = profile_builder_selected_block_uids_from_editor(editor_none, catalog)
        legacy_editor_b = pd.DataFrame(
            {
                "Использовать эту таблицу": [False, True],
                "block_id": ["ocr_p1_b1", "ocr_p1_b2"],
            }
        )
        selected_legacy_b = profile_builder_selected_block_uids_from_editor(legacy_editor_b, catalog)

        self.assertEqual(selected_a, [block_a])
        self.assertEqual(catalog["block_uid"].tolist(), [block_a, block_b])
        self.assertEqual(selected_b, [block_b])
        self.assertEqual(catalog["block_uid"].tolist(), [block_a, block_b])
        self.assertEqual(selected_none, [])
        self.assertEqual(selected_legacy_b, [block_b])

    def test_profile_builder_real_session_keys_keep_ocr_source_and_catalog(self) -> None:
        document_key = "kazakhstan.pdf:hash"
        candidates = pd.DataFrame(
            [
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 1,
                    "ocr_block_id": "ocr_p1_b1",
                    "block_title": "A",
                    "block_text": "Wheat 1 2",
                    "preview": "Wheat 1 2",
                    "extraction_method": "tesseract_ocr_candidate",
                },
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 1,
                    "ocr_block_id": "ocr_p1_b2",
                    "block_title": "B",
                    "block_text": "Barley 3 4",
                    "preview": "Barley 3 4",
                    "extraction_method": "tesseract_ocr_candidate",
                },
            ]
        )
        snapshot_catalog = build_profile_table_catalog(pd.DataFrame(), None, candidates)
        session_state = {
            profile_builder_source_state_key(document_key): "ocr",
            f"profile_builder_ocr_candidates:{document_key}": candidates,
            f"profile_builder_ocr_blocks_catalog:{document_key}": snapshot_catalog,
            f"profile_builder_block_selection_applied:{document_key}": ["ocr_candidate:1:ocr_p1_b1"],
        }

        source = profile_builder_get_source(session_state, document_key, "pdf_text_layer")
        source_catalog = profile_builder_catalog_for_source(pd.DataFrame(), source, snapshot_catalog)

        self.assertEqual(source, "ocr")
        self.assertNotIn(profile_builder_source_widget_key(document_key), session_state)
        self.assertEqual(
            source_catalog["block_uid"].tolist(),
            ["ocr_candidate:1:ocr_p1_b1", "ocr_candidate:1:ocr_p1_b2"],
        )

    def test_profile_builder_streamlit_apply_block_selection_keeps_ocr_catalog(self) -> None:
        pdf_path = PROJECT_DIR / "data" / "raw" / "0f9fa8d72b63_monitopring.rinka.ribi.na.01.06.2026.pdf"
        if not pdf_path.exists():
            self.skipTest("sample PDF is not available")
        pdf_bytes = pdf_path.read_bytes()
        filename = "manual_ocr_wizard_test.pdf"
        document_key = f"{filename}:{hashlib.md5(pdf_bytes).hexdigest()}"
        candidates = pd.DataFrame(
            [
                {
                    "source_file": filename,
                    "page": 1,
                    "ocr_block_id": "ocr_p1_b1",
                    "block_title": "A",
                    "block_text": "Wheat 1 2",
                    "preview": "Wheat 1 2",
                    "extraction_method": "tesseract_ocr_candidate",
                },
                {
                    "source_file": filename,
                    "page": 1,
                    "ocr_block_id": "ocr_p1_b2",
                    "block_title": "B",
                    "block_text": "Barley 3 4",
                    "preview": "Barley 3 4",
                    "extraction_method": "tesseract_ocr_candidate",
                },
            ]
        )
        catalog = build_profile_table_catalog(pd.DataFrame(), None, candidates)
        ocr_result = pd.DataFrame(
            [{"source_file": filename, "page": 1, "text": "Wheat 1 2", "extraction_level": "raw_ocr"}]
        )

        app_test = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30)
        app_test.run(timeout=30)
        app_test.file_uploader[0].upload(filename, pdf_bytes, "application/pdf").run(timeout=60)
        app_test.session_state[f"document_mode:{document_key}"] = "profile_setup"
        app_test.session_state[profile_builder_source_state_key(document_key)] = "ocr"
        app_test.session_state[f"ocr_result:{document_key}"] = ocr_result
        app_test.session_state[f"ocr_candidates:{document_key}"] = candidates
        app_test.session_state[f"profile_builder_ocr_result:{document_key}"] = ocr_result
        app_test.session_state[f"profile_builder_ocr_candidates:{document_key}"] = candidates
        app_test.session_state[f"profile_builder_ocr_blocks_catalog:{document_key}"] = catalog
        app_test.run(timeout=60)
        app_test.radio[0].set_value(PROFILE_BUILDER_STEPS[1]).run(timeout=60)
        submit_index = next(
            index
            for index, button in enumerate(app_test.button)
            if str(getattr(button, "key", "")).startswith("FormSubmitter:profile_builder_block_selection_form")
        )
        app_test.button[submit_index].click().run(timeout=60)

        info_messages = [getattr(message, "value", "") for message in app_test.info]
        self.assertFalse(
            any("Для выбранного источника пока нет таблиц" in message for message in info_messages)
        )
        self.assertEqual(app_test.session_state[profile_builder_source_state_key(document_key)], "ocr")
        self.assertEqual(
            app_test.session_state[f"profile_builder_block_selection_applied:{document_key}"],
            ["ocr_candidate:1:ocr_p1_b1"],
        )
        self.assertGreaterEqual(len(app_test.dataframe), 1)

    def test_user_profile_selected_rows_do_not_change_applied_ocr_table(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 17,
                    "ocr_block_id": "ocr_p17_b1",
                    "block_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                    "block_text": "Мясо птицы 1 2 3 4\nМолоко сухое 5 6 7 8",
                    "preview": "Мясо птицы 1 2 3 4\nМолоко сухое 5 6 7 8",
                    "extraction_method": "tesseract_ocr_candidate",
                }
            ]
        )
        source_rows = select_source_rows(pd.DataFrame(), candidates, {"block_uids": ["ocr_candidate:17:ocr_p17_b1"]})
        corrected_rows = apply_table_reconstruction(source_rows, {"method": "split_by_regex", "pattern": r"\s{1,}|\|"})
        corrected_preview_before = source_rows_to_preview_df(corrected_rows, limit=10)

        base_config = {
            "profile_name": "kazakhstan_ocr_tokens",
            "extraction": {"source": "ocr", "ocr": {"required": True, "engine": "tesseract", "lang": "rus+eng"}},
            "blocks": [
                {
                    "source_kind": "ocr_candidate",
                    "selector": {"block_uids": ["ocr_candidate:17:ocr_p17_b1"]},
                    "row_selection": {
                        "mode": "manual",
                        "selected_row_uids": ["ocr_candidate:17:ocr_p17_b1:row:1"],
                        "skip_empty_values": True,
                    },
                    "table_reconstruction": {"method": "split_by_regex", "pattern": r"\s{1,}|\|"},
                    "token_mapping": {
                        "token_1": {"enabled": True, "metric": "volume", "unit": "thousand_tons"},
                        "token_2": {"enabled": True, "metric": "trade_value", "unit": "million_usd", "currency": "USD"},
                    },
                }
            ],
        }
        first_result = apply_user_profile({}, base_config, ocr_candidates_df=candidates)
        base_config["blocks"][0]["row_selection"]["selected_row_uids"] = [
            "ocr_candidate:17:ocr_p17_b1:row:1",
            "ocr_candidate:17:ocr_p17_b1:row:2",
        ]
        second_result = apply_user_profile({}, base_config, ocr_candidates_df=candidates)
        corrected_preview_after = source_rows_to_preview_df(corrected_rows, limit=10)

        self.assertEqual(corrected_preview_before.to_dict("records"), corrected_preview_after.to_dict("records"))
        self.assertEqual(corrected_preview_before.loc[0, "row_uid"], "ocr_candidate:17:ocr_p17_b1:row:1")
        self.assertIn("column_4", corrected_preview_before.columns)
        self.assertEqual(len(first_result["structured_rows"]), 2)
        self.assertEqual(len(second_result["structured_rows"]), 4)

    def test_user_profile_ocr_two_rows_and_four_tokens_create_eight_rows(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "source_file": "kazakhstan.pdf",
                    "page": 17,
                    "ocr_block_id": "ocr_p17_b1",
                    "block_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                    "block_text": "Пшеница 1 2 3 4\nЯчмень 5 6 7 8",
                    "preview": "Пшеница 1 2 3 4\nЯчмень 5 6 7 8",
                    "extraction_method": "tesseract_ocr_candidate",
                }
            ]
        )
        config = {
            "profile_name": "kazakhstan_ocr_tokens",
            "display_name": "Kazakhstan OCR tokens",
            "extraction": {
                "source": "ocr",
                "ocr": {"required": True, "engine": "tesseract", "lang": "rus+eng", "pages": "auto", "dpi": 300},
            },
            "blocks": [
                {
                    "source_kind": "ocr_candidate",
                    "selector": {"block_uids": ["ocr_candidate:17:ocr_p17_b1"]},
                    "row_selection": {
                        "mode": "manual",
                        "selected_row_uids": [
                            "ocr_candidate:17:ocr_p17_b1:row:1",
                            "ocr_candidate:17:ocr_p17_b1:row:2",
                        ],
                        "skip_empty_values": True,
                    },
                    "token_mapping": {
                        "token_1": {"enabled": True, "metric": "volume", "year": 2023, "unit": "thousand_tons"},
                        "token_2": {"enabled": True, "metric": "trade_value", "year": 2023, "unit": "million_usd", "currency": "USD"},
                        "token_3": {"enabled": True, "metric": "volume", "year": 2024, "unit": "thousand_tons"},
                        "token_4": {"enabled": True, "metric": "trade_value", "year": 2024, "unit": "million_usd", "currency": "USD"},
                    },
                }
            ],
            "validation": {"required_fields": ["name", "value"], "value_positive": True},
        }

        result = apply_user_profile({}, config, ocr_candidates_df=candidates)

        self.assertEqual(len(config["blocks"][0]["row_selection"]["selected_row_uids"]), 2)
        self.assertEqual(len(result["structured_rows"]), 8)
        self.assertEqual(set(result["structured_rows"]["metric"]), {"volume", "trade_value"})

    def test_user_profile_pdf_text_layer_does_not_run_ocr(self) -> None:
        config = self._mock_user_profile_blocks_config(source="pdf_text_layer")
        calls = 0

        def mock_ocr_runner(_document, _ocr_config):
            nonlocal calls
            calls += 1
            return pd.DataFrame()

        result = apply_user_profile(
            {"raw_rows": self._mock_user_profile_raw_rows()},
            config,
            ocr_runner=mock_ocr_runner,
        )

        self.assertFalse(result["ocr_ran"])
        self.assertEqual(calls, 0)
        self.assertEqual(len(result["structured_rows"]), 2)

    def test_user_profile_blocks_yaml_serializes_extraction_settings(self) -> None:
        config = self._mock_user_profile_blocks_config(source="ocr")

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_path = save_user_profile(config, Path(tmp_dir))
            loaded = load_user_profile_file(saved_path)

        self.assertEqual(loaded["extraction"]["source"], "ocr")
        self.assertEqual(loaded["extraction"]["ocr"]["engine"], "tesseract")
        self.assertEqual(loaded["extraction"]["ocr"]["lang"], "rus+eng")

    def test_profile_draft_marks_sentence_like_sections_as_not_good(self) -> None:
        draft = build_profile_draft(
            source_file="obzor_ved_kazahstan_2025.pdf",
            selected_candidates_df=pd.DataFrame(
                [
                    {
                        "page": 13,
                        "block_title": "В Ton-3 стран-экспортеров продуктов питания на казахстанский рынок по итогам года",
                        "candidate_type": "mixed",
                        "table_score": 0.62,
                        "information_score": 0.86,
                        "preview": "В топ-3 стран-экспортеров вошли Россия, Китай и Беларусь.",
                    },
                    {
                        "page": 14,
                        "block_title": "В то же время в 2024 г. снизился объем импорта из Казахстана со стороны Туркмени-",
                        "candidate_type": "paragraph",
                        "table_score": 0.31,
                        "information_score": 0.78,
                        "preview": "В то же время в 2024 г. снизился объем импорта на 12,4%.",
                    },
                    {
                        "page": 15,
                        "block_title": "Смешанный блок импорта продукции АПК",
                        "candidate_type": "mixed",
                        "table_score": 0.62,
                        "information_score": 0.82,
                        "preview": "Импорт продукции АПК\nРоссия 10,0 11,0",
                    },
                ]
            ),
        )

        qualities = [section["section_quality"] for section in draft["target_sections"]]
        warnings = " ".join(
            warning
            for section in draft["target_sections"]
            for warning in section["section_warnings"]
        )

        self.assertNotIn("good", qualities)
        self.assertEqual(draft["profile_draft_summary"]["weak_sections"], 2)
        self.assertEqual(draft["profile_draft_summary"]["needs_review_sections"], 1)
        self.assertTrue(draft["profile_draft_summary"]["has_warnings"])
        self.assertFalse(draft["profile_draft_summary"]["ready_for_parser_prototype"])
        self.assertTrue(draft["profile_draft_summary"]["requires_developer_review"])
        self.assertIn("expected_fields were inferred by fallback rule", warnings)
        self.assertIn("block title looks like a sentence", warnings)
        self.assertIn("table_score is below recommended threshold", warnings)

    def test_parse_russian_number_for_prototype_parser(self) -> None:
        self.assertEqual(parse_prototype_russian_number("1 522,2"), 1522.2)
        self.assertEqual(parse_prototype_russian_number("964,6"), 964.6)
        self.assertEqual(parse_prototype_russian_number("20724"), 20724.0)
        self.assertEqual(parse_prototype_russian_number("20,1"), 20.1)
        self.assertIsNone(parse_prototype_russian_number("в3,5 раза"))

    def test_normalize_ocr_number_divides_lost_decimal_only_in_million_usd_sections(self) -> None:
        normalized_value, warnings = normalize_ocr_number(
            "20724",
            "Основные страны-экспортеры продукции АПК в Казахстан, 2020-2024 гг., млн долл. США",
            "млн долл. США",
            "1 Россия 20724 26782 32071 30781 33218",
        )

        self.assertEqual(normalized_value, 2072.4)
        self.assertEqual(warnings, [OCR_DECIMAL_WARNING])

        production_value, production_warnings = normalize_ocr_number(
            "14256",
            "Производство продукции растениеводства, тыс. тонн",
            "тыс. тонн",
            "Пшеница 14256 11814 16404 12110 18450",
        )

        self.assertEqual(production_value, 14256.0)
        self.assertEqual(production_warnings, [])

        suspicious_value, suspicious_warnings = normalize_ocr_number(
            "142561",
            "Производство продукции растениеводства, тыс. тонн",
            "тыс. тонн",
            "Пшеница 142561 118145 164042 121108 184500",
        )

        self.assertEqual(suspicious_value, 142561.0)
        self.assertEqual(suspicious_warnings, [THOUSAND_TONS_LARGE_WARNING])

    def test_prototype_parser_extracts_country_year_rows_from_draft(self) -> None:
        draft = build_profile_draft(
            source_file="obzor_ved_kazahstan_2025.pdf",
            selected_candidates_df=pd.DataFrame(
                [
                    {
                        "page": 11,
                        "block_title": "Основные страны-экспортеры продукции АПК в Казахстан, 2020-2024 гг., млн долл. США",
                        "candidate_type": "table",
                        "table_score": 0.92,
                        "information_score": 0.95,
                        "block_text": (
                            "1 Россия 20724 26782 32071 30781 33218\n"
                            "2 Китай 1583 1652 2574 3818 3883"
                        ),
                    }
                ]
            ),
        )

        parsed = parse_sections_from_draft(draft)

        self.assertEqual(len(parsed), 10)
        self.assertEqual(set(parsed["extraction_method"]), {"draft_profile_parser"})
        self.assertEqual(set(parsed["extraction_level"]), {"prototype_structured"})
        self.assertEqual(set(parsed["section_parse_mode"]), {"year_series_2020_2024"})
        self.assertEqual(set(parsed["validation_status"]), {"needs_review"})
        self.assertEqual(set(parsed["review_status"]), {"needs_review"})
        self.assertEqual(set(parsed["normalization_method"]), {"ocr_decimal_divide_by_10"})
        self.assertTrue(parsed["warnings"].str.contains(OCR_DECIMAL_WARNING, regex=False).all())
        first = parsed.iloc[0]
        self.assertEqual(first["country"], "Россия")
        self.assertIsNone(first["commodity"])
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["year"], 2020)
        self.assertEqual(first["raw_value"], "20724")
        self.assertEqual(first["normalized_value"], 2072.4)
        self.assertEqual(first["value"], 2072.4)
        self.assertLessEqual(first["confidence"], 0.85)
        self.assertEqual(first["currency"], "USD")

    def test_prototype_parser_extracts_production_commodity_year_rows(self) -> None:
        draft = build_profile_draft(
            source_file="obzor_ved_kazahstan_2025.pdf",
            selected_candidates_df=pd.DataFrame(
                [
                    {
                        "page": 7,
                        "block_title": "Производство продукции растениеводства, тыс. тонн",
                        "candidate_type": "table",
                        "table_score": 0.9,
                        "information_score": 0.95,
                        "block_text": (
                            "Пшеница 14256,1 11814,5 16404,2 12110,8 18450,0\n"
                            "Ячмень 3659,1 2377,2 3287,1 2590,5 3800,4"
                        ),
                    }
                ]
            ),
        )

        parsed = parse_sections_from_draft(draft)

        self.assertEqual(len(parsed), 10)
        self.assertEqual(set(parsed["section_parse_mode"]), {"production_2020_2024"})
        self.assertEqual(set(parsed["validation_status"]), {"passed"})
        self.assertEqual(set(parsed["review_status"]), {"auto_approved"})
        self.assertEqual(parsed.iloc[0]["commodity"], "Пшеница")
        self.assertIsNone(parsed.iloc[0]["country"])
        self.assertEqual(parsed.iloc[0]["unit"], "тыс. тонн")
        self.assertEqual(parsed.iloc[0]["year"], 2020)
        self.assertEqual(parsed.iloc[0]["raw_value"], "14256,1")
        self.assertEqual(parsed.iloc[0]["normalized_value"], 14256.1)
        self.assertEqual(parsed.iloc[0]["normalization_method"], "russian_number_parse")
        self.assertEqual(parsed.iloc[0]["value"], 14256.1)

    def test_prototype_parser_extracts_2023_2024_complex_trade_rows_for_mapping(self) -> None:
        draft = build_profile_draft(
            source_file="obzor_ved_kazahstan_2025.pdf",
            selected_candidates_df=pd.DataFrame(
                [
                    {
                        "page": 15,
                        "block_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                        "candidate_type": "table",
                        "table_score": 0.85,
                        "information_score": 0.9,
                        "block_text": "Пшеница 1 522,2 964,6 -557,6 -36,6%",
                    }
                ]
            ),
        )

        parsed = parse_sections_from_draft(draft)

        self.assertEqual(len(parsed), 1)
        first = parsed.iloc[0]
        self.assertEqual(first["section_parse_mode"], "complex_trade_2023_2024")
        self.assertEqual(first["extraction_level"], "prototype_complex_wide")
        self.assertEqual(first["commodity"], "Пшеница")
        self.assertTrue(pd.isna(first["year"]))
        self.assertTrue(pd.isna(first["value"]))
        self.assertEqual(json.loads(first["raw_numeric_tokens"]), ["1 522,2", "964,6", "-557,6", "-36,6%"])
        self.assertEqual(json.loads(first["parsed_numeric_tokens"]), [1522.2, 964.6, -557.6, -36.6])
        self.assertEqual(set(parsed["validation_status"]), {"needs_review"})
        self.assertEqual(set(parsed["review_status"]), {"needs_review"})
        self.assertEqual(first["warnings"], COMPLEX_TRADE_WARNING)

    def test_prototype_parser_keeps_ranked_complex_trade_tokens_wide(self) -> None:
        draft = build_profile_draft(
            source_file="obzor_ved_kazahstan_2025.pdf",
            selected_candidates_df=pd.DataFrame(
                [
                    {
                        "page": 15,
                        "block_title": "РРјРїРѕСЂС‚ РљР°Р·Р°С…СЃС‚Р°РЅР° РёР· Р РѕСЃСЃРёРё, 2023-2024 РіРі.",
                        "candidate_type": "table",
                        "table_score": 0.85,
                        "information_score": 0.9,
                        "block_text": "1 РњСЏСЃРѕ РїС‚РёС†С‹ 24 358 258 448 43 20,1 СЌР» 253",
                    }
                ]
            ),
        )

        parsed = parse_sections_from_draft(draft)

        self.assertEqual(len(parsed), 1)
        first = parsed.iloc[0]
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["commodity"], "РњСЏСЃРѕ РїС‚РёС†С‹")
        self.assertEqual(first["section_parse_mode"], "complex_trade_2023_2024")
        self.assertEqual(first["extraction_level"], "prototype_complex_wide")
        self.assertEqual(json.loads(first["raw_numeric_tokens"]), ["24 358", "258 448", "43", "20,1", "253"])
        self.assertEqual(json.loads(first["parsed_numeric_tokens"]), [24358.0, 258448.0, 43.0, 20.1, 253.0])
        self.assertTrue(pd.isna(first["year"]))
        self.assertTrue(pd.isna(first["value"]))
        self.assertEqual(first["warnings"], COMPLEX_TRADE_WARNING)

    def test_build_mapping_preview_decodes_complex_tokens(self) -> None:
        complex_df = pd.DataFrame(
            [
                {
                    "section_id": "export_trade",
                    "section_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                    "row_id": 1,
                    "commodity": "Пшеница",
                    "rank": None,
                    "raw_numeric_tokens": json.dumps(["1 522,2", "964,6"], ensure_ascii=False),
                    "parsed_numeric_tokens": json.dumps([1522.2, 964.6], ensure_ascii=False),
                    "evidence_text": "Пшеница 1 522,2 964,6",
                    "extraction_level": "prototype_complex_wide",
                }
            ]
        )

        preview = build_mapping_preview(complex_df)

        self.assertEqual(preview.iloc[0]["raw_numeric_tokens"], ["1 522,2", "964,6"])
        self.assertEqual(preview.iloc[0]["parsed_numeric_tokens"], [1522.2, 964.6])
        self.assertEqual(preview.iloc[0]["commodity"], "Пшеница")

    def test_apply_complex_mapping_builds_structured_rows(self) -> None:
        complex_df = pd.DataFrame(
            [
                {
                    "source_file": "obzor_ved_kazahstan_2025.pdf",
                    "source_type": "pdf",
                    "page": 15,
                    "section_id": "export_trade",
                    "section_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                    "row_id": 1,
                    "commodity": "Мясо птицы",
                    "country": None,
                    "rank": 1,
                    "raw_numeric_tokens": json.dumps(["24358,0", "258448,0", "43,0", "20,1", "253,0"], ensure_ascii=False),
                    "parsed_numeric_tokens": json.dumps([24358.0, 258448.0, 43.0, 20.1, 253.0], ensure_ascii=False),
                    "unit": None,
                    "currency": None,
                    "evidence_text": "1 Мясо птицы 24 358 258 448 43 20,1 253",
                    "extraction_level": "prototype_complex_wide",
                }
            ]
        )

        mapped = apply_complex_mapping(
            complex_df,
            {
                "section_id": "export_trade",
                "mapping": {
                    "token_1": "volume_2023",
                    "token_2": "value_2023",
                    "token_5": "change_pct",
                    "token_6": "value_2024",
                },
                "mapping_verified": True,
            },
        )

        self.assertEqual(len(mapped), 4)
        self.assertEqual(set(mapped["extraction_method"]), {"manual_complex_mapping"})
        self.assertEqual(set(mapped["extraction_level"]), {"mapped_complex_structured"})
        first = mapped.iloc[0]
        self.assertEqual(first["metric"], "volume")
        self.assertEqual(first["year"], 2023)
        self.assertEqual(first["value"], 24358.0)
        self.assertEqual(first["raw_value"], "24358,0")
        self.assertEqual(first["mapping_token"], "token_1")
        self.assertEqual(first["validation_status"], "needs_review")
        self.assertEqual(first["review_status"], "mapped_by_user")
        missing = mapped[mapped["raw_value"].isna()].iloc[0]
        self.assertEqual(missing["metric"], "trade_value")
        self.assertEqual(missing["year"], 2024)
        self.assertEqual(missing["validation_status"], "needs_review")
        self.assertIn("mapped token is missing", missing["warnings"])

    def test_apply_complex_mapping_supports_token_attribute_schema(self) -> None:
        complex_df = pd.DataFrame(
            [
                {
                    "source_file": "obzor_ved_kazahstan_2025.pdf",
                    "source_type": "pdf",
                    "page": 15,
                    "section_id": "export_trade",
                    "section_title": "Экспорт Казахстана в Россию, 2023-2024 гг.",
                    "row_id": 1,
                    "commodity": "Мясо птицы",
                    "country": None,
                    "rank": 1,
                    "raw_numeric_tokens": json.dumps(["24 358", "258 448", "20,1"], ensure_ascii=False),
                    "parsed_numeric_tokens": json.dumps([24358.0, 258448.0, 20.1], ensure_ascii=False),
                    "unit": None,
                    "currency": None,
                    "evidence_text": "1 Мясо птицы 24 358 258 448 20,1",
                    "extraction_level": "prototype_complex_wide",
                }
            ]
        )

        mapped = apply_complex_mapping(
            complex_df,
            {
                "section_id": "export_trade",
                "mapping_schema_version": "2",
                "mapping_type": "token_attribute_mapping",
                "token_mapping": {
                    "token_1": {
                        "enabled": True,
                        "metric": "volume",
                        "year": 2023,
                        "unit": "thousand_tons",
                        "currency": None,
                        "label": "2023 volume",
                    },
                    "token_2": {
                        "enabled": True,
                        "metric": "trade_value",
                        "year": 2023,
                        "unit": "million_usd",
                        "currency": "USD",
                        "label": "2023 value",
                    },
                    "token_3": {"enabled": False, "metric": "change_pct", "year": 2024, "unit": "percent"},
                },
            },
        )

        self.assertEqual(len(mapped), 2)
        self.assertEqual(list(mapped["mapping_token"]), ["token_1", "token_2"])
        self.assertEqual(list(mapped["metric"]), ["volume", "trade_value"])
        self.assertEqual(list(mapped["year"]), [2023, 2023])
        self.assertEqual(list(mapped["unit"]), ["thousand_tons", "million_usd"])
        self.assertEqual(list(mapped["currency"]), [None, "USD"])
        self.assertEqual(list(mapped["mapping_label"]), ["2023 volume", "2023 value"])

    def test_prepare_review_editor_df_coerces_streamlit_compatible_types(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "_review_row_index": 0,
                    "section_title": None,
                    "commodity": "Meat",
                    "metric": "volume",
                    "year": 2024,
                    "unit": "thousand_tons",
                    "currency": None,
                    "original_value": None,
                    "value": "25,3",
                    "raw_value": "253",
                    "normalized_value": "25.3",
                    "mapping_token": "token_1",
                    "evidence_text": "Meat 253",
                    "reconstruction_status": "needs_review",
                    "reconstruction_warnings": RECONSTRUCTION_WARNING,
                    "warnings": RECONSTRUCTION_WARNING,
                    "edited_by_user": None,
                    "approved_by_user": None,
                    "review_comment": None,
                }
            ]
        )

        editor_df = prepare_review_editor_df(rows)

        self.assertTrue(pd.api.types.is_numeric_dtype(editor_df["year"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(editor_df["value"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(editor_df["original_value"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(editor_df["raw_value"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(editor_df["normalized_value"]))
        self.assertTrue(pd.api.types.is_bool_dtype(editor_df["approved_by_user"]))
        self.assertTrue(pd.api.types.is_bool_dtype(editor_df["edited_by_user"]))
        self.assertEqual(editor_df.loc[0, "year"], 2024)
        self.assertEqual(editor_df.loc[0, "value"], 25.3)
        self.assertFalse(editor_df.loc[0, "approved_by_user"])
        self.assertEqual(editor_df.loc[0, "section_title"], "")
        self.assertEqual(editor_df.loc[0, "review_comment"], "")
        self.assertEqual(editor_df.loc[0, "metric"], "volume")
        self.assertEqual(editor_df.loc[0, "unit"], "thousand_tons")
        self.assertTrue(editor_df["commodity"].dtype == object or pd.api.types.is_string_dtype(editor_df["commodity"]))

    def test_compact_review_editor_omits_long_audit_columns(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "_review_row_index": 0,
                    "row_uid": "row-1",
                    "section_title": "Trade",
                    "commodity": "Meat",
                    "metric": "volume",
                    "year": 2024,
                    "unit": "thousand_tons",
                    "currency": None,
                    "original_value": 25.3,
                    "value": "25,3",
                    "raw_value": "253",
                    "normalized_value": "25.3",
                    "mapping_token": "token_1",
                    "evidence_text": "Very long evidence text",
                    "reconstruction_status": "needs_review",
                    "reconstruction_warnings": RECONSTRUCTION_WARNING,
                    "warnings": RECONSTRUCTION_WARNING,
                    "edited_by_user": True,
                    "approved_by_user": None,
                    "review_comment": "",
                }
            ]
        )

        compact_df = prepare_compact_review_editor_df(rows)

        self.assertIn("row_uid", compact_df.columns)
        self.assertIn("_review_row_index", compact_df.columns)
        self.assertIn("value", compact_df.columns)
        self.assertNotIn("evidence_text", compact_df.columns)
        self.assertNotIn("reconstruction_warnings", compact_df.columns)
        self.assertNotIn("warnings", compact_df.columns)
        self.assertNotIn("edited_by_user", compact_df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(compact_df["year"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(compact_df["value"]))
        self.assertTrue(pd.api.types.is_bool_dtype(compact_df["approved_by_user"]))

    def test_audit_trail_coverage_requires_source_page_and_evidence(self) -> None:
        rows = pd.DataFrame(
            [
                {"source_file": "report.pdf", "page": 1, "evidence_text": "row text", "value": 1.0},
                {"source_file": "report.pdf", "page": 2, "evidence_text": "", "value": 2.0},
            ]
        )

        coverage = audit_trail_coverage(rows)

        self.assertEqual(coverage["audit_rows"], 1)
        self.assertEqual(coverage["total_rows"], 2)
        self.assertEqual(coverage["coverage_pct"], 50.0)

    def test_processing_dashboard_summary_reports_readiness_and_funnel(self) -> None:
        reviewed_mapped = pd.DataFrame(
            [
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 5,
                    "section_id": "trade",
                    "section_title": "Trade",
                    "row_id": 1,
                    "commodity": "Meat",
                    "metric": "volume",
                    "year": 2024,
                    "value": 25.4,
                    "unit": "thousand_tons",
                    "currency": None,
                    "raw_value": "254",
                    "normalized_value": 25.4,
                    "mapping_token": "token_1",
                    "evidence_text": "Meat 254",
                    "confidence": 0.95,
                    "validation_status": "passed_after_review",
                    "warnings": "",
                    "review_status": "user_approved",
                    "approved_by_user": True,
                    "edited_by_user": True,
                }
            ]
        )
        ocr_candidates = pd.DataFrame(
            [
                {
                    "ocr_block_id": "ocr-1",
                    "candidate_type": "table",
                    "table_score": 0.8,
                    "page": 5,
                }
            ]
        )

        summary = build_processing_dashboard_summary(
            processing_time=1.25,
            file_type="pdf",
            active_profile="generic_pdf",
            profile_metadata={"profile_confidence": 0.4},
            bad_text_layer=False,
            raw_rows=pd.DataFrame(),
            raw_table_summary_df=pd.DataFrame([{"table_id": "t1"}]),
            ocr_result_df=pd.DataFrame(),
            ocr_candidates_df=ocr_candidates,
            selected_ocr_candidates_df=ocr_candidates,
            structured_rows=pd.DataFrame(),
            prototype_structured_df=pd.DataFrame(),
            mapped_complex_df=reviewed_mapped,
            reviewed_mapped_df=reviewed_mapped,
        )
        funnel = build_processing_funnel(summary, bad_text_layer=False)

        self.assertEqual(summary["readiness_status"], "Готово к экспорту")
        self.assertEqual(summary["ocr_status"], "не требовался")
        self.assertEqual(summary["strong_ocr_candidates_count"], 1)
        self.assertEqual(summary["user_approved_rows"], 1)
        self.assertEqual(summary["rows_still_need_review"], 0)
        self.assertEqual(summary["audit_coverage"]["coverage_pct"], 100.0)
        self.assertEqual(list(funnel.columns), ["stage", "count", "status", "comment"])
        self.assertIn("9. Export", set(funnel["stage"]))

    def test_processing_dashboard_summary_flags_ocr_need_before_profile_setup(self) -> None:
        summary = build_processing_dashboard_summary(
            processing_time=0.5,
            file_type="pdf",
            active_profile="generic_pdf",
            profile_metadata={"profile_confidence": 0.2},
            bad_text_layer=True,
            raw_rows=pd.DataFrame([{"source_file": "scan.pdf", "page": 1, "evidence_text": "raw"}]),
            raw_table_summary_df=pd.DataFrame([{"table_id": "t1"}]),
            ocr_result_df=pd.DataFrame(),
            ocr_candidates_df=pd.DataFrame(),
            selected_ocr_candidates_df=pd.DataFrame(),
            structured_rows=pd.DataFrame(),
            prototype_structured_df=pd.DataFrame(),
            mapped_complex_df=pd.DataFrame(),
            reviewed_mapped_df=pd.DataFrame(),
        )

        self.assertEqual(summary["readiness_status"], "Нужен OCR")
        self.assertEqual(summary["ocr_status"], "рекомендован")

    def test_mapped_review_draft_does_not_change_applied_rows_before_apply(self) -> None:
        mapped = pd.DataFrame(
            [
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 15,
                    "section_id": "export_trade",
                    "section_title": "Trade table",
                    "row_id": 1,
                    "commodity": "Meat",
                    "metric": "volume",
                    "year": 2024,
                    "value": 25.3,
                    "unit": "thousand_tons",
                    "currency": None,
                    "raw_value": "253",
                    "normalized_value": 25.3,
                    "normalization_method": "manual_complex_mapping",
                    "mapping_token": "token_1",
                    "mapping_label": "volume",
                    "reconstruction_status": "needs_review",
                    "reconstruction_warnings": RECONSTRUCTION_WARNING,
                    "evidence_text": "Meat 253",
                    "extraction_method": "manual_complex_mapping",
                    "extraction_level": "mapped_complex_structured",
                    "confidence": 0.6,
                    "validation_status": "needs_review",
                    "warnings": RECONSTRUCTION_WARNING,
                    "review_status": "needs_review",
                }
            ]
        )
        reviewed = prepare_reviewed_mapped_export(mapped)
        draft = prepare_review_editor_df(select_mapped_rows_for_review(reviewed, "needs_review"))

        self.assertIn("row_uid", reviewed.columns)
        self.assertEqual(len(draft), 1)
        draft.loc[draft.index[0], "approved_by_user"] = True
        draft.loc[draft.index[0], "value"] = 25.4
        draft.loc[draft.index[0], "review_comment"] = "checked"

        self.assertTrue(review_editor_has_unsaved_changes(reviewed, draft))
        self.assertFalse(bool(reviewed.loc[0, "approved_by_user"]))
        self.assertEqual(reviewed.loc[0, "review_status"], "needs_review")
        self.assertEqual(reviewed.loc[0, "value"], 25.3)
        self.assertEqual(len(select_mapped_rows_for_review(reviewed, "needs_review")), 1)
        self.assertEqual(
            mapped_review_editor_summary(reviewed, draft),
            {
                "rows_in_editor": 1,
                "changed_values": 1,
                "marked_for_approval": 1,
                "comments": 1,
            },
        )

        draft.loc[draft.index[0], "approved_by_user"] = False
        self.assertTrue(review_editor_has_unsaved_changes(reviewed, draft))
        self.assertEqual(mapped_review_editor_summary(reviewed, draft)["marked_for_approval"], 0)

        updated = apply_mapped_review_edits(reviewed, draft)
        self.assertTrue(bool(updated.loc[0, "edited_by_user"]))
        self.assertFalse(bool(updated.loc[0, "approved_by_user"]))
        self.assertEqual(updated.loc[0, "original_value"], 25.3)
        self.assertEqual(updated.loc[0, "value"], 25.4)
        self.assertEqual(updated.loc[0, "review_comment"], "checked")

    def test_mapped_review_edits_preserve_audit_trail_and_approve_row(self) -> None:
        mapped = pd.DataFrame(
            [
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 15,
                    "section_id": "export_trade",
                    "section_title": "Trade table",
                    "row_id": 1,
                    "commodity": "Meat",
                    "metric": "volume",
                    "year": 2024,
                    "value": 25.3,
                    "unit": "thousand_tons",
                    "currency": None,
                    "raw_value": "253",
                    "normalized_value": 25.3,
                    "normalization_method": "manual_complex_mapping",
                    "mapping_token": "token_1",
                    "mapping_label": "volume",
                    "reconstruction_status": "needs_review",
                    "reconstruction_warnings": RECONSTRUCTION_WARNING,
                    "evidence_text": "Meat 253",
                    "extraction_method": "manual_complex_mapping",
                    "extraction_level": "mapped_complex_structured",
                    "confidence": 0.6,
                    "validation_status": "needs_review",
                    "warnings": RECONSTRUCTION_WARNING,
                    "review_status": "needs_review",
                },
                {
                    "source_file": "report.pdf",
                    "source_type": "pdf",
                    "page": 15,
                    "section_id": "export_trade",
                    "section_title": "Trade table",
                    "row_id": 2,
                    "commodity": "Fish",
                    "metric": "volume",
                    "year": 2024,
                    "value": 10.0,
                    "unit": "thousand_tons",
                    "currency": None,
                    "raw_value": "10,0",
                    "normalized_value": 10.0,
                    "normalization_method": "manual_complex_mapping",
                    "mapping_token": "token_1",
                    "mapping_label": "volume",
                    "reconstruction_status": "ok",
                    "reconstruction_warnings": "",
                    "evidence_text": "Fish 10,0",
                    "extraction_method": "manual_complex_mapping",
                    "extraction_level": "mapped_complex_structured",
                    "confidence": 0.75,
                    "validation_status": "passed",
                    "warnings": "",
                    "review_status": "auto_approved",
                },
            ]
        )

        reviewed = prepare_reviewed_mapped_export(mapped)
        rows_for_review = select_mapped_rows_for_review(reviewed)
        self.assertEqual(len(rows_for_review), 1)
        self.assertEqual(rows_for_review.iloc[0]["evidence_text"], "Meat 253")
        self.assertEqual(rows_for_review.iloc[0]["reconstruction_warnings"], RECONSTRUCTION_WARNING)

        edited = rows_for_review.copy()
        edited.loc[edited.index[0], "value"] = "25,4"
        edited.loc[edited.index[0], "approved_by_user"] = True
        edited.loc[edited.index[0], "review_comment"] = "verified against source"

        updated = apply_mapped_review_edits(reviewed, edited)
        corrected = updated.iloc[0]
        self.assertEqual(corrected["original_value"], 25.3)
        self.assertEqual(corrected["value"], 25.4)
        self.assertEqual(corrected["normalized_value"], 25.4)
        self.assertTrue(corrected["edited_by_user"])
        self.assertTrue(corrected["approved_by_user"])
        self.assertEqual(corrected["review_comment"], "verified against source")
        self.assertEqual(corrected["validation_status"], "passed_after_review")
        self.assertEqual(corrected["review_status"], "user_approved")
        self.assertEqual(corrected["confidence"], 0.95)
        self.assertIn("corrected by user", corrected["warnings"])
        self.assertEqual(corrected["evidence_text"], "Meat 253")
        self.assertEqual(corrected["reconstruction_warnings"], RECONSTRUCTION_WARNING)

        summary = mapped_review_summary(updated)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["required_review"], 1)
        self.assertEqual(summary["edited_by_user"], 1)
        self.assertEqual(summary["approved_by_user"], 1)
        self.assertEqual(summary["remaining_unreviewed"], 0)
        self.assertEqual(len(select_mapped_rows_for_review(updated, "approved_by_user")), 1)
        self.assertTrue(select_mapped_rows_for_review(updated, "remaining_unreviewed").empty)

        restored = restore_mapped_review_original_values(updated)
        self.assertEqual(restored.iloc[0]["value"], 25.3)
        self.assertEqual(restored.iloc[0]["normalized_value"], 25.3)
        self.assertFalse(restored.iloc[0]["edited_by_user"])
        self.assertFalse(restored.iloc[0]["approved_by_user"])
        self.assertNotIn("corrected by user", restored.iloc[0]["warnings"])

    def test_mapping_preset_suggestion_and_template_are_schema_v2(self) -> None:
        suggested = suggest_mapping_preset(
            "Экспорт Казахстана в Россию, 2023-2024 гг.",
            "Объем Стоимость Прирост 2023 2024",
            8,
        )
        mapping = token_mapping_from_preset(suggested, 8)

        self.assertEqual(suggested, TRADE_2023_2024_PRESET)
        self.assertTrue(mapping["token_1"]["enabled"])
        self.assertEqual(mapping["token_1"]["metric"], "volume")
        self.assertEqual(mapping["token_1"]["year"], 2023)
        self.assertEqual(mapping["token_1"]["unit"], "thousand_tons")
        self.assertEqual(mapping["token_2"]["metric"], "trade_value")
        self.assertEqual(mapping["token_2"]["currency"], "USD")

    def test_reconstruct_numeric_tokens_splits_joined_trade_values_with_review_warning(self) -> None:
        mapping_config = {
            "token_mapping": token_mapping_from_preset(TRADE_2023_2024_PRESET, 8),
        }

        reconstruction = reconstruct_numeric_tokens(
            evidence_text="1 Мясо птицы 24 358 258 448 43 20,1 эл 253",
            raw_tokens=["24 358", "258 448", "43", "20,1", "253"],
            mapping_config=mapping_config,
        )

        self.assertEqual(reconstruction["expected_count"], 8)
        self.assertEqual(len(reconstruction["reconstructed_raw_tokens"]), 8)
        self.assertEqual(reconstruction["reconstructed_raw_tokens"][:6], ["24", "358", "258", "448", "43", "20,1"])
        self.assertIsNone(reconstruction["reconstructed_raw_tokens"][6])
        self.assertEqual(reconstruction["reconstructed_raw_tokens"][7], "253")
        self.assertEqual(reconstruction["reconstructed_values"][:6], [2.4, 35.8, 25.8, 44.8, 4.3, 20.1])
        self.assertIsNone(reconstruction["reconstructed_values"][6])
        self.assertEqual(reconstruction["reconstructed_values"][7], 25.3)
        self.assertEqual(reconstruction["reconstruction_status"], "needs_review")
        self.assertEqual(reconstruction["reconstruction_method"], "partial_repair")
        self.assertIn(RECONSTRUCTION_WARNING, reconstruction["reconstruction_warnings"])
        self.assertIn(FAILED_RECONSTRUCTION_WARNING, reconstruction["reconstruction_warnings"])

    def test_reconstruct_numeric_tokens_splits_joined_percent_pair_but_keeps_uncertain_large_tokens(self) -> None:
        mapping_config = {
            "token_mapping": token_mapping_from_preset(TRADE_2023_2024_PRESET, 8),
        }

        reconstruction = reconstruct_numeric_tokens(
            evidence_text="Пшеница 2633 3550 16483 3799 -9650 369249 70",
            raw_tokens=["2633", "3550", "16483", "3799", "-9650", "369249", "70"],
            mapping_config=mapping_config,
        )

        self.assertEqual(reconstruction["expected_count"], 8)
        self.assertEqual(reconstruction["reconstructed_raw_tokens"][5:8], ["369", "249", "70"])
        self.assertEqual(reconstruction["reconstructed_values"][5:8], [36.9, 24.9, 7.0])
        self.assertEqual(reconstruction["reconstructed_values"][0], 2633.0)
        self.assertEqual(reconstruction["reconstruction_status"], "needs_review")
        self.assertIn("possible lost decimal separator", " ".join(reconstruction["reconstruction_warnings"]))

    def test_ocr_pages_reports_clear_message_when_tesseract_missing(self) -> None:
        with patch("src.extract_ocr.configure_tesseract", return_value=None):
            with self.assertRaises(OCRUnavailableError) as error:
                extract_ocr_pages(
                    str(PROJECT_DIR / "data" / "examples" / "obzor_ved_kazahstan_2025.pdf"),
                    pages=[1],
                )

        self.assertEqual(str(error.exception), TESSERACT_INSTALL_MESSAGE)

    def test_get_tesseract_cmd_prefers_environment_variable(self) -> None:
        env_cmd = r"C:\tools\tesseract.exe"
        with (
            patch("src.extract_ocr.os.getenv", return_value=env_cmd),
            patch("src.extract_ocr.Path.exists", return_value=True),
            patch("src.extract_ocr.shutil.which", return_value=r"C:\path\tesseract.exe"),
        ):
            self.assertEqual(get_tesseract_cmd(), env_cmd)

    def test_get_tesseract_cmd_falls_back_to_windows_install_path(self) -> None:
        candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        def exists(path: Path) -> bool:
            return str(path) == candidate

        with (
            patch("src.extract_ocr.os.getenv", return_value=None),
            patch("src.extract_ocr.shutil.which", return_value=None),
            patch("src.extract_ocr.Path.exists", exists),
        ):
            self.assertEqual(get_tesseract_cmd(), candidate)

    def test_get_tesseract_cmd_prefers_standard_windows_install_before_path(self) -> None:
        candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        def exists(path: Path) -> bool:
            return str(path) == candidate

        with (
            patch("src.extract_ocr.os.getenv", return_value=None),
            patch("src.extract_ocr.shutil.which", return_value=r"C:\path\tesseract.exe"),
            patch("src.extract_ocr.Path.exists", exists),
        ):
            self.assertEqual(get_tesseract_cmd(), candidate)

    def test_configure_tesseract_sets_pytesseract_command(self) -> None:
        cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        fake_pytesseract = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=None))

        with (
            patch("src.extract_ocr.get_tesseract_cmd", return_value=cmd),
            patch("src.extract_ocr._load_pytesseract", return_value=fake_pytesseract),
        ):
            self.assertEqual(configure_tesseract(), cmd)

        self.assertEqual(fake_pytesseract.pytesseract.tesseract_cmd, cmd)

    def test_get_available_tesseract_languages_uses_configured_command(self) -> None:
        cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        fake_pytesseract = SimpleNamespace(
            pytesseract=SimpleNamespace(tesseract_cmd=None),
            get_languages=lambda config="": ["eng", "osd", "rus", r"script\Cyrillic"],
        )

        with (
            patch("src.extract_ocr.get_tesseract_cmd", return_value=cmd),
            patch("src.extract_ocr._load_pytesseract", return_value=fake_pytesseract),
            patch(
                "src.extract_ocr.subprocess.run",
                return_value=SimpleNamespace(
                    stdout=(
                        'List of available languages in "C:\\Program Files\\Tesseract-OCR/tessdata/" (4):\n'
                        "eng\n"
                        "osd\n"
                        "rus\n"
                        "script\\Cyrillic\n"
                    )
                ),
            ),
        ):
            self.assertEqual(
                get_available_tesseract_languages(),
                ["eng", "osd", "rus", r"script\Cyrillic"],
            )

        self.assertEqual(fake_pytesseract.pytesseract.tesseract_cmd, cmd)

    def test_language_available_accepts_combined_russian_and_english(self) -> None:
        self.assertTrue(is_language_available("rus+eng", ["eng", "osd", "rus", r"script\Cyrillic"]))
        self.assertFalse(is_language_available("rus+deu", ["eng", "osd", "rus", r"script\Cyrillic"]))

    def test_ocr_pages_reports_language_error_when_selected_lang_missing(self) -> None:
        with (
            patch("src.extract_ocr.configure_tesseract", return_value=r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            patch("src.extract_ocr._load_pytesseract", return_value=SimpleNamespace()),
            patch("src.extract_ocr.get_available_tesseract_languages", return_value=["eng"]),
        ):
            with self.assertRaises(OCRLanguageError) as error:
                extract_ocr_pages(
                    str(PROJECT_DIR / "data" / "examples" / "obzor_ved_kazahstan_2025.pdf"),
                    pages=[1],
                    lang="rus+eng",
                )

        self.assertIn("ошибка языка OCR", str(error.exception))
        self.assertIn("Выбранный язык: rus+eng", str(error.exception))

    def test_is_tesseract_available_uses_resolved_command(self) -> None:
        cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        fake_pytesseract = SimpleNamespace(
            pytesseract=SimpleNamespace(tesseract_cmd=None),
            get_tesseract_version=lambda: "5.0.0",
        )

        with (
            patch("src.extract_ocr.get_tesseract_cmd", return_value=cmd),
            patch("src.extract_ocr._load_pytesseract", return_value=fake_pytesseract),
        ):
            self.assertTrue(is_tesseract_available())

        self.assertEqual(fake_pytesseract.pytesseract.tesseract_cmd, cmd)

    def test_source_registry_contains_fish_market_report_profile(self) -> None:
        registry = load_source_registry(str(PROJECT_DIR / "configs" / "sources.yaml"))
        fish_config = get_source_config("fish_market_report")

        self.assertIn("fish_market_report", registry)
        self.assertEqual(fish_config["document_profile"], "fish_market_report")
        self.assertEqual(fish_config["extraction_strategy"], "profile_parser")
        self.assertFalse(fish_config["requires_ocr"])
        self.assertFalse(fish_config["uses_llm"])
        self.assertEqual(
            get_expected_sections("fish_market_report"),
            [
                "catch_main_species",
                "wholesale_far_east",
                "wholesale_north_west",
                "wholesale_center",
                "export_market_prices",
                "retail_frozen_fish",
            ],
        )

    def test_ui_translation_round_trip_keeps_export_schema_internal(self) -> None:
        parsed = parse_fish_market_report(
            ["Рыбные ряды", "", "Лососевые*", "335,6 | - 44,9% | -"],
            source_file="report.pdf",
            page=1,
        )
        validated = validate_extracted_data(parsed)

        ui_df = rename_columns_for_ui(translate_status_columns(validated))
        self.assertIn("Блок документа", ui_df.columns)
        self.assertIn("Показатель", ui_df.columns)
        self.assertIn("Статус проверки", ui_df.columns)
        self.assertIn("Улов основных видов рыбы", set(ui_df["Блок документа"]))
        self.assertIn("Освоение квоты", set(ui_df["Показатель"]))
        self.assertIn("Требует проверки", set(ui_df["Статус проверки"]))

        restored = restore_status_columns(ui_df)
        self.assertIn("section_name", restored.columns)
        self.assertIn("indicator", restored.columns)
        self.assertIn("validation_status", restored.columns)
        self.assertIn("catch_main_species", set(restored["section_name"]))
        self.assertIn("quota_utilization", set(restored["indicator"]))
        self.assertIn("warning", set(restored["validation_status"]))

        csv_header = export_to_csv(restored).decode("utf-8-sig").splitlines()[0]
        self.assertIn("section_name", csv_header)
        self.assertNotIn("Блок документа", csv_header)

    def test_coverage_display_uses_russian_headers(self) -> None:
        parsed = parse_fish_market_report(
            ["Рыбные ряды", "", "Лососевые*", "335,6 | - 44,9% | -"],
            source_file="report.pdf",
            page=1,
        )
        validated = validate_extracted_data(parsed)
        coverage = build_coverage_summary(validated, profile="fish_market_report")
        display = format_coverage_for_ui(coverage)

        self.assertIn("Блок документа", display.columns)
        self.assertIn("Статус", display.columns)
        self.assertIn("Ожидалось строк", display.columns)
        self.assertIn("Извлечено строк", display.columns)
        self.assertIn("Требуют проверки", display.columns)
        self.assertIn("Ошибки", display.columns)
        self.assertIn("Техническое имя блока", display.columns)


if __name__ == "__main__":
    unittest.main()
