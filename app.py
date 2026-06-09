from __future__ import annotations

import json
import hashlib
import time
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.complex_mapping import (
    CURRENCY_OPTIONS,
    MAPPING_PRESETS,
    METRIC_OPTIONS,
    UNIT_OPTIONS,
    YEAR_OPTIONS,
    apply_complex_mapping,
    build_mapping_config,
    build_mapping_preview,
    suggest_mapping_preset,
    token_mapping_from_preset,
)
from src.coverage_summary import build_coverage_summary, coverage_counts
from src.numeric_token_reconstruction import reconstruct_numeric_tokens
from src.export import export_to_csv, export_to_excel
from src.extract_excel import extract_excel
from src.extract_ocr import (
    OCRLanguageError,
    OCRPageRenderError,
    OCRUnavailableError,
    TESSERACT_INSTALL_MESSAGE,
    extract_ocr_pages,
    get_available_tesseract_languages,
    get_pdf_page_count,
    get_tesseract_cmd,
    is_tesseract_available,
)
from src.ocr_engines import get_ocr_engine, get_available_engines
from src.ocr_engines.base import OcrSettings
from src.ocr_engines.conversion import ocr_page_results_to_dataframe
from src.extract_pdf import extract_pdf
from src.file_router import detect_file_type
from src.document_profiles import PROFILE_PARSER_CONFIDENCE_THRESHOLD
from src.normalize import normalize_dataframe
from src.ocr_table_candidates import extract_ocr_table_candidates
from src.profile_draft import (
    build_profile_draft,
    candidate_is_good_profile_section,
    dump_profile_draft_json,
    dump_profile_draft_yaml,
)
from src.profile_parser_prototype import parse_sections_from_draft
from src.llm_profile_generator import LLMProfileGenerator, validate_generated_profile
from src.raw_table_analysis import build_raw_table_summary
from src.source_registry import get_display_name, get_source_config, load_source_registry
from src.user_profile_builder import (
    apply_row_filters,
    apply_table_reconstruction,
    apply_user_profile,
    apply_user_profile_to_sources,
    build_profile_table_catalog,
    dump_user_profile_yaml,
    find_matching_user_profiles,
    load_user_profiles,
    normalize_user_number,
    save_user_profile,
    select_source_rows_for_block_uids,
    select_user_profile_export_columns,
    source_row_uid,
    source_rows_to_preview_df,
)
from src.utils import safe_file_name
from src.validate import validate_extracted_data


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
USER_PROFILES_DIR = PROJECT_DIR / "profiles" / "user_profiles"

VALIDATION_STATUS_LABELS = {
    "passed_after_review": "Принято после ручной проверки",
    "passed": "Успешно",
    "passed_with_warning": "Принято с предупреждением",
    "needs_review": "Нужна проверка",
    "warning": "Требует проверки",
    "failed": "Ошибка",
    "raw_extracted": "Извлечено как сырой фрагмент",
}

REVIEW_STATUS_LABELS = {
    "user_approved": "Подтверждено пользователем",
    "auto_approved": "Автоматически принято",
    "needs_review": "Нужна проверка",
    "mapped_by_user": "Размечено пользователем",
    "manual_required": "Требуется ручная обработка",
    "needs_profile_setup": "Нужна настройка профиля",
    "needs_ocr": "Требуется OCR",
}

VALIDATION_STATUS_VALUES = {label: value for value, label in VALIDATION_STATUS_LABELS.items()}
REVIEW_STATUS_VALUES = {label: value for value, label in REVIEW_STATUS_LABELS.items()}

EXTRACTION_LEVEL_LABELS = {
    "structured": "Структурированные данные",
    "prototype_structured": "Прототип структурированных данных",
    "prototype_complex_wide": "Прототип сложной таблицы для mapping",
    "mapped_complex_structured": "Размеченные сложные строки",
    "raw": "Сырой фрагмент",
    "raw_ocr": "OCR-сырой текст",
}
EXTRACTION_LEVEL_VALUES = {label: value for value, label in EXTRACTION_LEVEL_LABELS.items()}

TEXT_LAYER_QUALITY_LABELS = {
    "ok": "нормальное",
    "bad": "плохое",
    "ocr": "OCR",
}
TEXT_LAYER_QUALITY_VALUES = {label: value for value, label in TEXT_LAYER_QUALITY_LABELS.items()}

COLUMN_LABELS_RU = {
    "original_value": "Исходное значение",
    "edited_by_user": "Исправлено пользователем",
    "approved_by_user": "Подтверждено пользователем",
    "review_comment": "Комментарий проверки",
    "source_file": "Файл",
    "source_type": "Тип источника",
    "page": "Страница",
    "sheet": "Лист",
    "row_id": "ID строки",
    "table_id": "ID таблицы",
    "row_index_in_table": "Строка в таблице",
    "section_name": "Блок документа",
    "section_id": "ID секции",
    "section_title": "Название секции",
    "section_parse_mode": "Режим разбора секции",
    "indicator": "Показатель",
    "metric": "Метрика",
    "commodity": "Товар",
    "country": "Страна",
    "region": "Регион",
    "route": "Маршрут",
    "date": "Дата",
    "rank": "Ранг",
    "year": "Год",
    "value": "Значение",
    "raw_value": "Сырое значение",
    "normalized_value": "Нормализованное значение",
    "normalization_method": "Метод нормализации",
    "raw_numeric_tokens": "Сырые числовые токены",
    "parsed_numeric_tokens": "Распознанные числовые токены",
    "mapping_token": "Токен mapping",
    "mapping_label": "Метка mapping",
    "raw_numeric_tokens_original": "Исходные числовые токены OCR",
    "reconstructed_raw_tokens": "Восстановленные числовые токены",
    "reconstruction_method": "Метод восстановления чисел",
    "reconstruction_status": "Статус восстановления чисел",
    "reconstruction_warnings": "Предупреждения восстановления чисел",
    "unit": "Единица измерения",
    "currency": "Валюта",
    "evidence_text": "Фрагмент-источник",
    "extraction_method": "Метод извлечения",
    "extraction_level": "Уровень извлечения",
    "text_layer_quality": "Качество текстового слоя",
    "text_layer_warning": "Предупреждение текстового слоя",
    "confidence": "Уверенность",
    "validation_status": "Статус проверки",
    "warnings": "Предупреждения",
    "review_status": "Статус ручной проверки",
}

SECTION_LABELS_RU = {
    "catch_main_species": "Улов основных видов рыбы",
    "wholesale_far_east": "Оптовые цены / Дальний Восток",
    "wholesale_north_west": "Оптовые цены / Северо-Запад",
    "wholesale_center": "Оптовые цены / Центр",
    "export_market_prices": "Цены на внешних рынках сбыта",
    "retail_frozen_fish": "Розничные цены на мороженую рыбу",
    "raw_page_text": "Текст страницы",
    "raw_pdf_table": "Таблица из PDF",
}

INDICATOR_LABELS_RU = {
    "catch_volume": "Объём улова",
    "yoy_change": "Изменение год к году",
    "quota_utilization": "Освоение квоты",
    "wholesale_price": "Оптовая цена",
    "weekly_change": "Изменение за неделю",
    "ytd_change": "Изменение с начала года",
    "world_price": "Цена на мировом рынке",
    "monthly_change": "Изменение за месяц",
    "yearly_change": "Изменение за год",
    "retail_price_frozen_fish": "Розничная цена мороженой рыбы",
}

SECTION_VALUES = {label: value for value, label in SECTION_LABELS_RU.items()}
INDICATOR_VALUES = {label: value for value, label in INDICATOR_LABELS_RU.items()}
COLUMN_VALUES = {label: value for value, label in COLUMN_LABELS_RU.items()}

EXTRACTION_STRATEGY_LABELS = {
    "profile_parser": "Профильный parser",
    "pdfplumber": "pdfplumber",
    "pandas": "pandas",
}

REVIEW_FILTERS = {
    "Всё требующее проверки": "all",
    "Только предупреждения": "warning",
    "Только ошибки": "failed",
}

PROTOTYPE_ROW_FILTERS = {
    "Все": "all",
    "Автоматически принято": "auto_approved",
    "С предупреждениями": "warnings",
    "Требуют проверки": "needs_review",
}

REVIEW_COLUMNS = [
    "section_name",
    "validation_status",
    "extraction_level",
    "indicator",
    "commodity",
    "region",
    "value",
    "unit",
    "evidence_text",
    "warnings",
    "confidence",
]

PROTOTYPE_SUSPICIOUS_COLUMNS = [
    "section_id",
    "commodity/country",
    "year",
    "raw_value",
    "normalized_value",
    "unit",
    "normalization_method",
    "warnings",
    "evidence_text",
]

MAPPED_REVIEW_COLUMNS = [
    "row_uid",
    "section_title",
    "commodity",
    "metric",
    "year",
    "unit",
    "currency",
    "original_value",
    "value",
    "raw_value",
    "normalized_value",
    "mapping_token",
    "evidence_text",
    "reconstruction_status",
    "reconstruction_warnings",
    "warnings",
    "edited_by_user",
    "approved_by_user",
    "review_comment",
]

MAPPED_REVIEW_EDITABLE_COLUMNS = [
    "commodity",
    "metric",
    "year",
    "value",
    "unit",
    "currency",
    "review_comment",
    "approved_by_user",
]

MAPPED_REVIEW_CORRECTION_COLUMNS = [
    "commodity",
    "metric",
    "year",
    "value",
    "unit",
    "currency",
]

MAPPED_REVIEW_AUDIT_COLUMNS = [
    "original_value",
    "edited_by_user",
    "approved_by_user",
    "review_comment",
]

REVIEWED_MAPPED_EXPORT_COLUMNS = [
    "source_file",
    "source_type",
    "page",
    "section_id",
    "section_title",
    "row_id",
    "row_uid",
    "commodity",
    "country",
    "rank",
    "metric",
    "year",
    "original_value",
    "value",
    "edited_by_user",
    "approved_by_user",
    "review_comment",
    "unit",
    "currency",
    "raw_value",
    "normalized_value",
    "normalization_method",
    "mapping_token",
    "mapping_label",
    "raw_numeric_tokens_original",
    "reconstructed_raw_tokens",
    "reconstruction_method",
    "reconstruction_status",
    "reconstruction_warnings",
    "evidence_text",
    "extraction_method",
    "extraction_level",
    "confidence",
    "validation_status",
    "warnings",
    "review_status",
]

CORRECTED_BY_USER_WARNING = "corrected by user"
REVIEW_ROW_UID_COLUMN = "row_uid"
REVIEW_ROW_INDEX_COLUMN = "_review_row_index"

MAPPED_REVIEW_COMPACT_EDITOR_COLUMNS = [
    REVIEW_ROW_UID_COLUMN,
    REVIEW_ROW_INDEX_COLUMN,
    "section_title",
    "commodity",
    "metric",
    "year",
    "value",
    "unit",
    "currency",
    "approved_by_user",
    "review_comment",
]

MAPPED_REVIEW_EVIDENCE_COLUMNS = [
    "section_title",
    "commodity",
    "metric",
    "year",
    "unit",
    "currency",
    "value",
    "raw_value",
    "normalized_value",
    "mapping_token",
    "evidence_text",
    "reconstruction_status",
    "reconstruction_warnings",
    "warnings",
]

MAPPED_REVIEW_FILTERS = {
    "Все строки": "all",
    "Требуют проверки": "needs_review",
    "Исправленные пользователем": "edited_by_user",
    "Подтверждённые пользователем": "approved_by_user",
    "Остались непроверенными": "remaining_unreviewed",
}

RAW_DISPLAY_COLUMNS = [
    "source_file",
    "page",
    "table_id",
    "row_index_in_table",
    "section_name",
    "evidence_text",
    "extraction_method",
    "extraction_level",
    "text_layer_quality",
    "text_layer_warning",
    "confidence",
    "validation_status",
    "review_status",
]

RAW_EXPORT_COLUMNS = [
    "source_file",
    "source_type",
    "page",
    "table_id",
    "row_id",
    "row_index_in_table",
    "section_name",
    "evidence_text",
    "extraction_method",
    "extraction_level",
    "text_layer_quality",
    "text_layer_warning",
    "confidence",
    "validation_status",
    "review_status",
]

TECHNICAL_RAW_EXPORT_COLUMNS = [
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
]

TABLE_SUMMARY_COLUMNS = [
    "table_id",
    "page",
    "raw_rows_count",
    "column_count",
    "preview",
    "table_score",
    "table_reason",
    "text_layer_quality",
    "text_layer_warning",
]

PROFILE_CANDIDATE_EXPORT_COLUMNS = [
    "source_file",
    "table_id",
    "page",
    "table_score",
    "table_reason",
    "preview",
    "raw_rows_count",
]

OCR_DISPLAY_COLUMNS = [
    "page",
    "evidence_text",
    "confidence",
    "extraction_method",
    "review_status",
]

OCR_EXPORT_COLUMNS = [
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
]

OCR_CANDIDATE_DISPLAY_COLUMNS = [
    "page",
    "block_title",
    "candidate_type",
    "preview",
    "rows_count",
    "numbers_count",
    "table_score",
    "information_score",
    "reason",
]

OCR_CANDIDATE_EXPORT_COLUMNS = [
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
]

OCR_CANDIDATE_LABELS_RU = {
    "page": "Страница",
    "block_title": "Заголовок блока",
    "candidate_type": "Тип кандидата",
    "preview": "Preview",
    "rows_count": "Строк",
    "numbers_count": "Чисел",
    "table_score": "Table score",
    "information_score": "Information score",
    "reason": "Причина оценки",
}

OCR_CANDIDATE_TYPE_LABELS_RU = {
    "table": "Таблица",
    "paragraph": "Текстовый блок",
    "chart_text": "График",
    "mixed": "Смешанный",
    "unknown": "Не определён",
}

OCR_CANDIDATE_FILTERS = {
    "Все": "all",
    "Таблицы": "table",
    "Текстовые блоки": "paragraph",
    "Графики/смешанные": "chart_or_mixed",
}

OCR_CANDIDATE_TYPE_ORDER = {
    "table": 0,
    "mixed": 1,
    "chart_text": 2,
    "paragraph": 3,
    "unknown": 4,
}

OCR_BEST_TABLE_SCORE_THRESHOLD = 0.65

TABLE_SUMMARY_LABELS_RU = {
    "source_file": "Файл",
    "table_id": "table_id",
    "page": "Страница",
    "raw_rows_count": "Строк",
    "column_count": "Колонок",
    "preview": "Preview",
    "table_score": "Оценка",
    "table_reason": "Причина оценки",
    "text_layer_quality": "Качество текстового слоя",
    "text_layer_warning": "Предупреждение",
}


def translate_status_columns(df):
    """Return a UI copy with Russian status, section, and indicator labels."""
    display_df = df.copy()
    if "validation_status" in display_df.columns:
        display_df["validation_status"] = (
            display_df["validation_status"].map(VALIDATION_STATUS_LABELS).fillna(display_df["validation_status"])
        )
    if "review_status" in display_df.columns:
        display_df["review_status"] = (
            display_df["review_status"].map(REVIEW_STATUS_LABELS).fillna(display_df["review_status"])
        )
    if "extraction_level" in display_df.columns:
        display_df["extraction_level"] = (
            display_df["extraction_level"].map(EXTRACTION_LEVEL_LABELS).fillna(display_df["extraction_level"])
        )
    if "text_layer_quality" in display_df.columns:
        display_df["text_layer_quality"] = (
            display_df["text_layer_quality"].map(TEXT_LAYER_QUALITY_LABELS).fillna(display_df["text_layer_quality"])
        )
    if "section_name" in display_df.columns:
        display_df["section_name"] = (
            display_df["section_name"].map(SECTION_LABELS_RU).fillna(display_df["section_name"])
        )
    if "indicator" in display_df.columns:
        display_df["indicator"] = (
            display_df["indicator"].map(INDICATOR_LABELS_RU).fillna(display_df["indicator"])
        )
    return display_df


def rename_columns_for_ui(df):
    """Return a UI copy with Russian column names."""
    return df.rename(columns=COLUMN_LABELS_RU)


def restore_status_columns(df):
    """Convert Russian UI labels back to internal column names and codes before export."""
    internal_df = df.copy()
    internal_df = internal_df.rename(columns=COLUMN_VALUES)
    if "validation_status" in internal_df.columns:
        internal_df["validation_status"] = (
            internal_df["validation_status"].map(VALIDATION_STATUS_VALUES).fillna(internal_df["validation_status"])
        )
    if "review_status" in internal_df.columns:
        internal_df["review_status"] = (
            internal_df["review_status"].map(REVIEW_STATUS_VALUES).fillna(internal_df["review_status"])
        )
    if "extraction_level" in internal_df.columns:
        internal_df["extraction_level"] = (
            internal_df["extraction_level"].map(EXTRACTION_LEVEL_VALUES).fillna(internal_df["extraction_level"])
        )
    if "text_layer_quality" in internal_df.columns:
        internal_df["text_layer_quality"] = (
            internal_df["text_layer_quality"].map(TEXT_LAYER_QUALITY_VALUES).fillna(internal_df["text_layer_quality"])
        )
    if "section_name" in internal_df.columns:
        internal_df["section_name"] = (
            internal_df["section_name"].map(SECTION_VALUES).fillna(internal_df["section_name"])
        )
    if "indicator" in internal_df.columns:
        internal_df["indicator"] = (
            internal_df["indicator"].map(INDICATOR_VALUES).fillna(internal_df["indicator"])
        )
    return internal_df


def format_coverage_for_ui(coverage_df):
    """Return a display copy of coverage summary with Russian column names."""
    if coverage_df.empty:
        return coverage_df

    display_df = coverage_df.copy()
    display_df["status"] = display_df["found"].map({True: "найден", False: "не найден"})
    display_df = display_df[
        [
            "block_name",
            "status",
            "expected_rows",
            "actual_rows",
            "warning_rows",
            "error_rows",
            "section_name",
        ]
    ]
    return display_df.rename(
        columns={
            "block_name": "Блок документа",
            "status": "Статус",
            "expected_rows": "Ожидалось строк",
            "actual_rows": "Извлечено строк",
            "warning_rows": "Требуют проверки",
            "error_rows": "Ошибки",
            "section_name": "Техническое имя блока",
        }
    )


def select_review_columns(df):
    """Keep the review table focused on fields that explain why a row is suspicious."""
    columns = [column for column in REVIEW_COLUMNS if column in df.columns]
    return df[columns].copy()


def split_rows_by_extraction_level(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split extracted rows into structured business data and raw PDF fragments."""
    if df.empty or "extraction_level" not in df.columns:
        return df.copy(), df.iloc[0:0].copy()

    levels = df["extraction_level"].fillna("structured").astype(str)
    raw_mask = levels.isin({"raw", "raw_ocr"})
    return df.loc[~raw_mask].copy(), df.loc[raw_mask].copy()


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a stable-column view without failing when older data lacks a new field."""
    existing_columns = [column for column in columns if column in df.columns]
    return df[existing_columns].copy()


def record_performance_timing(
    timings: list[dict[str, object]],
    block: str,
    start_time: float,
    *,
    cache_status: str = "",
    rows: int | None = None,
) -> None:
    """Append a lightweight timing row for the debug diagnostics expander."""
    timings.append(
        {
            "block": block,
            "seconds": round(time.perf_counter() - start_time, 4),
            "cache_status": cache_status,
            "rows": "" if rows is None else int(rows),
        }
    )


def stable_json_hash(value) -> str:
    """Return a short deterministic hash for mapping configs and UI state keys."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def split_prototype_rows(df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prototype parser output into simple long rows and complex wide rows."""
    if df is None or df.empty or "extraction_level" not in df.columns:
        empty = pd.DataFrame()
        return empty, empty
    levels = df["extraction_level"].fillna("").astype(str)
    simple_rows = df.loc[levels.eq("prototype_structured")].copy()
    complex_rows = df.loc[levels.eq("prototype_complex_wide")].copy()
    return simple_rows, complex_rows


def prototype_warning_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows with parser warnings or warning validation status."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    warnings = (
        df["warnings"].fillna("").astype(str).str.strip().ne("")
        if "warnings" in df.columns
        else pd.Series(False, index=df.index)
    )
    validation_warnings = (
        df["validation_status"].fillna("").astype(str).eq("passed_with_warning")
        if "validation_status" in df.columns
        else pd.Series(False, index=df.index)
    )
    return warnings | validation_warnings


def filter_prototype_rows(df: pd.DataFrame, row_filter: str) -> pd.DataFrame:
    """Filter prototype parser rows by review buckets."""
    if df is None or df.empty or row_filter == "all":
        return df.copy() if df is not None else pd.DataFrame()
    if row_filter == "auto_approved" and "review_status" in df.columns:
        return df.loc[df["review_status"].eq("auto_approved")].copy()
    if row_filter == "warnings":
        return df.loc[prototype_warning_mask(df)].copy()
    if row_filter == "needs_review":
        masks: list[pd.Series] = []
        if "review_status" in df.columns:
            masks.append(df["review_status"].eq("needs_review"))
        if "validation_status" in df.columns:
            masks.append(df["validation_status"].eq("needs_review"))
        if masks:
            mask = masks[0]
            for next_mask in masks[1:]:
                mask = mask | next_mask
            return df.loc[mask].copy()
    return df.copy()


def select_prototype_suspicious_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return prototype rows that deserve manual attention."""
    if df is None or df.empty:
        return pd.DataFrame(columns=PROTOTYPE_SUSPICIOUS_COLUMNS)

    warning_mask = prototype_warning_mask(df)
    review_mask = (
        df["review_status"].fillna("").astype(str).eq("needs_review")
        if "review_status" in df.columns
        else pd.Series(False, index=df.index)
    )
    validation_review_mask = (
        df["validation_status"].fillna("").astype(str).eq("needs_review")
        if "validation_status" in df.columns
        else pd.Series(False, index=df.index)
    )
    normalization_mask = (
        df["normalization_method"].fillna("").astype(str).isin(
            ["ocr_decimal_divide_by_10", "manual_review_required"]
        )
        if "normalization_method" in df.columns
        else pd.Series(False, index=df.index)
    )
    suspicious = df.loc[warning_mask | review_mask | validation_review_mask | normalization_mask].copy()
    if suspicious.empty:
        return pd.DataFrame(columns=PROTOTYPE_SUSPICIOUS_COLUMNS)

    entity = pd.Series("", index=suspicious.index, dtype="object")
    if "country" in suspicious.columns:
        country = suspicious["country"].fillna("").astype(str)
        entity = country.where(country.ne(""), entity)
    if "commodity" in suspicious.columns:
        commodity = suspicious["commodity"].fillna("").astype(str)
        entity = entity.where(entity.ne(""), commodity)
    suspicious["commodity/country"] = entity
    return select_existing_columns(suspicious, PROTOTYPE_SUSPICIOUS_COLUMNS)


def _is_missing_review_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _review_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or _is_missing_review_value(value):
        return False
    return str(value).strip().casefold() in {"true", "1", "yes", "y", "да"}


def _coerce_review_number(value):
    if _is_missing_review_value(value):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace(" ", "").replace(",", ".")
    parsed = pd.to_numeric(pd.Series([normalized]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return value
    return float(parsed)


def _review_values_equal(left, right) -> bool:
    if _is_missing_review_value(left) and _is_missing_review_value(right):
        return True
    left_number = _coerce_review_number(left)
    right_number = _coerce_review_number(right)
    if isinstance(left_number, float) and isinstance(right_number, float):
        return abs(left_number - right_number) < 1e-9
    return str(left or "") == str(right or "")


def _append_review_warning(existing, warning: str) -> str:
    parts = [
        part.strip()
        for part in str(existing or "").split(";")
        if part.strip()
    ]
    if warning not in parts:
        parts.append(warning)
    return "; ".join(parts)


def _remove_review_warning(existing, warning: str) -> str:
    parts = [
        part.strip()
        for part in str(existing or "").split(";")
        if part.strip() and part.strip() != warning
    ]
    return "; ".join(parts)


def _review_row_uid(row: pd.Series) -> str:
    parts = [
        row.get("source_file"),
        row.get("page"),
        row.get("section_id"),
        row.get("commodity"),
        row.get("metric"),
        row.get("year"),
        row.get("mapping_token"),
        row.get("row_id"),
    ]
    raw = "|".join("" if _is_missing_review_value(part) else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def ensure_mapped_review_row_uids(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a copy with stable row_uid values for review/editor matching."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REVIEWED_MAPPED_EXPORT_COLUMNS)

    result = df.copy()
    if REVIEW_ROW_UID_COLUMN not in result.columns:
        result[REVIEW_ROW_UID_COLUMN] = ""
    missing_uids = result[REVIEW_ROW_UID_COLUMN].fillna("").astype(str).str.strip().eq("")
    if missing_uids.any():
        result.loc[missing_uids, REVIEW_ROW_UID_COLUMN] = result.loc[missing_uids].apply(_review_row_uid, axis=1)
    return result


def prepare_reviewed_mapped_export(mapped_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return mapped rows with human-review audit columns and stable export order."""
    if mapped_df is None or mapped_df.empty:
        return pd.DataFrame(columns=REVIEWED_MAPPED_EXPORT_COLUMNS)

    reviewed = ensure_mapped_review_row_uids(mapped_df)
    for column in MAPPED_REVIEW_AUDIT_COLUMNS:
        if column not in reviewed.columns:
            if column in {"edited_by_user", "approved_by_user"}:
                reviewed[column] = False
            else:
                reviewed[column] = ""

    reviewed["edited_by_user"] = reviewed["edited_by_user"].fillna(False).map(_review_bool)
    reviewed["approved_by_user"] = reviewed["approved_by_user"].fillna(False).map(_review_bool)
    if "review_comment" in reviewed.columns:
        reviewed["review_comment"] = reviewed["review_comment"].fillna("").astype(str)

    ordered_columns = [column for column in REVIEWED_MAPPED_EXPORT_COLUMNS if column in reviewed.columns]
    extra_columns = [column for column in reviewed.columns if column not in ordered_columns]
    return reviewed[ordered_columns + extra_columns].copy()


def mapped_rows_need_review_mask(
    mapped_df: pd.DataFrame | None,
    *,
    exclude_user_approved: bool = True,
) -> pd.Series:
    """Return rows that should be shown in the mapped-row human review queue."""
    if mapped_df is None or mapped_df.empty:
        return pd.Series(dtype=bool)

    review_status = (
        mapped_df["review_status"].fillna("").astype(str)
        if "review_status" in mapped_df.columns
        else pd.Series("", index=mapped_df.index)
    )
    validation_status = (
        mapped_df["validation_status"].fillna("").astype(str)
        if "validation_status" in mapped_df.columns
        else pd.Series("passed", index=mapped_df.index)
    )
    warnings = (
        mapped_df["warnings"].fillna("").astype(str).str.strip()
        if "warnings" in mapped_df.columns
        else pd.Series("", index=mapped_df.index)
    )
    values_missing = (
        mapped_df["value"].isna()
        if "value" in mapped_df.columns
        else pd.Series(True, index=mapped_df.index)
    )
    needs_review = (
        review_status.eq("needs_review")
        | validation_status.ne("passed")
        | warnings.ne("")
        | values_missing
    )

    if exclude_user_approved:
        approved = (
            mapped_df["approved_by_user"].map(_review_bool)
            if "approved_by_user" in mapped_df.columns
            else pd.Series(False, index=mapped_df.index)
        )
        needs_review = needs_review & ~approved & ~review_status.eq("user_approved")

    return needs_review


def select_mapped_rows_for_review(
    mapped_df: pd.DataFrame | None,
    review_filter: str = "needs_review",
) -> pd.DataFrame:
    """Return mapped rows for the selected human-review filter."""
    reviewed = prepare_reviewed_mapped_export(mapped_df)
    if reviewed.empty:
        return pd.DataFrame(columns=[REVIEW_ROW_INDEX_COLUMN] + MAPPED_REVIEW_COLUMNS)

    if review_filter == "all":
        review_rows = reviewed.copy()
    elif review_filter == "edited_by_user":
        review_rows = reviewed.loc[reviewed["edited_by_user"].map(_review_bool)].copy()
    elif review_filter == "approved_by_user":
        review_rows = reviewed.loc[reviewed["approved_by_user"].map(_review_bool)].copy()
    elif review_filter == "remaining_unreviewed":
        review_rows = reviewed.loc[mapped_rows_need_review_mask(reviewed, exclude_user_approved=True)].copy()
    else:
        review_rows = reviewed.loc[mapped_rows_need_review_mask(reviewed, exclude_user_approved=False)].copy()

    if review_rows.empty:
        return pd.DataFrame(columns=[REVIEW_ROW_INDEX_COLUMN] + MAPPED_REVIEW_COLUMNS)

    review_rows[REVIEW_ROW_INDEX_COLUMN] = review_rows.index
    return select_existing_columns(review_rows, [REVIEW_ROW_INDEX_COLUMN] + MAPPED_REVIEW_COLUMNS)


def prepare_review_editor_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a Streamlit data_editor-safe copy of mapped rows needing review."""
    columns = [REVIEW_ROW_INDEX_COLUMN] + MAPPED_REVIEW_COLUMNS
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    editor_df = df.copy()
    for column in columns:
        if column in editor_df.columns:
            continue
        if column in {"edited_by_user", "approved_by_user"}:
            editor_df[column] = False
        elif column in {"year", "value", "original_value", "raw_value", "normalized_value"}:
            editor_df[column] = pd.NA
        else:
            editor_df[column] = ""

    for column in [REVIEW_ROW_INDEX_COLUMN, "year", "value", "original_value", "raw_value", "normalized_value"]:
        editor_df[column] = pd.to_numeric(
            editor_df[column].map(_coerce_review_number),
            errors="coerce",
        )

    for column in ["edited_by_user", "approved_by_user"]:
        editor_df[column] = editor_df[column].fillna(False).map(_review_bool).astype(bool)

    if "metric" in editor_df.columns:
        editor_df["metric"] = editor_df["metric"].where(editor_df["metric"].isin(METRIC_OPTIONS), "other")
        editor_df["metric"] = editor_df["metric"].astype(str)

    for column, options in {"unit": UNIT_OPTIONS, "currency": CURRENCY_OPTIONS}.items():
        editor_df[column] = editor_df[column].where(editor_df[column].isin(options), None).astype(object)

    for column in [
        REVIEW_ROW_UID_COLUMN,
        "section_title",
        "commodity",
        "review_comment",
        "mapping_token",
        "evidence_text",
        "reconstruction_status",
        "reconstruction_warnings",
        "warnings",
    ]:
        editor_df[column] = editor_df[column].fillna("").astype(str)

    return editor_df[columns].copy()


def prepare_compact_review_editor_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a small Streamlit editor DataFrame without long evidence/warning columns."""
    editor_df = prepare_review_editor_df(df)
    if editor_df.empty:
        return pd.DataFrame(columns=MAPPED_REVIEW_COMPACT_EDITOR_COLUMNS)
    return select_existing_columns(editor_df, MAPPED_REVIEW_COMPACT_EDITOR_COLUMNS)


def mapped_review_evidence_label(row: pd.Series) -> str:
    """Build a concise label for selecting a mapped row evidence preview."""
    label_parts = []
    for column in ["commodity", "metric", "year", "mapping_token"]:
        value = row.get(column)
        if not _is_missing_review_value(value):
            label_parts.append(str(value))
    if not label_parts:
        row_uid = row.get(REVIEW_ROW_UID_COLUMN)
        label_parts.append(str(row_uid or "row"))
    return " | ".join(label_parts)


def apply_mapped_review_edits(
    mapped_df: pd.DataFrame | None,
    edited_review_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge editable human-review rows back into the full mapped-row export."""
    reviewed = prepare_reviewed_mapped_export(mapped_df)
    if reviewed.empty or edited_review_rows is None or edited_review_rows.empty:
        return reviewed

    row_uid_to_index = (
        reviewed[REVIEW_ROW_UID_COLUMN].fillna("").astype(str).to_dict()
        if REVIEW_ROW_UID_COLUMN in reviewed.columns
        else {}
    )
    row_uid_to_index = {row_uid: index for index, row_uid in row_uid_to_index.items() if row_uid}

    for _, edited_row in edited_review_rows.iterrows():
        row_index = None
        if REVIEW_ROW_UID_COLUMN in edited_row and not _is_missing_review_value(edited_row[REVIEW_ROW_UID_COLUMN]):
            row_index = row_uid_to_index.get(str(edited_row[REVIEW_ROW_UID_COLUMN]))
        if row_index is None:
            if REVIEW_ROW_INDEX_COLUMN not in edited_row or _is_missing_review_value(edited_row[REVIEW_ROW_INDEX_COLUMN]):
                continue
            try:
                row_index = int(edited_row[REVIEW_ROW_INDEX_COLUMN])
            except (TypeError, ValueError):
                continue
        if row_index not in reviewed.index:
            continue

        row_was_corrected = False
        value_changed = False
        for column in MAPPED_REVIEW_CORRECTION_COLUMNS:
            if column not in edited_row or column not in reviewed.columns:
                continue
            new_value = edited_row[column]
            if column == "value":
                new_value = _coerce_review_number(new_value)
            if _review_values_equal(reviewed.at[row_index, column], new_value):
                continue

            if column == "value":
                if _is_missing_review_value(reviewed.at[row_index, "original_value"]):
                    val = reviewed.at[row_index, "value"]
                    reviewed.at[row_index, "original_value"] = str(val) if not _is_missing_review_value(val) else ""
                reviewed.at[row_index, "normalized_value"] = new_value
                value_changed = True
            reviewed.at[row_index, column] = new_value
            row_was_corrected = True

        if "review_comment" in edited_row and "review_comment" in reviewed.columns:
            comment = "" if _is_missing_review_value(edited_row["review_comment"]) else str(edited_row["review_comment"])
            reviewed.at[row_index, "review_comment"] = comment

        if row_was_corrected:
            reviewed.at[row_index, "edited_by_user"] = True
            reviewed.at[row_index, "warnings"] = _append_review_warning(
                reviewed.at[row_index, "warnings"],
                CORRECTED_BY_USER_WARNING,
            )
            if value_changed and "normalization_method" in reviewed.columns:
                reviewed.at[row_index, "normalization_method"] = "manual_user_correction"

        approved_by_user = (
            _review_bool(edited_row["approved_by_user"])
            if "approved_by_user" in edited_row
            else _review_bool(reviewed.at[row_index, "approved_by_user"])
        )
        reviewed.at[row_index, "approved_by_user"] = approved_by_user
        if approved_by_user:
            reviewed.at[row_index, "validation_status"] = "passed_after_review"
            reviewed.at[row_index, "review_status"] = "user_approved"
            current_confidence = pd.to_numeric(
                pd.Series([reviewed.at[row_index, "confidence"] if "confidence" in reviewed.columns else None]),
                errors="coerce",
            ).iloc[0]
            confidence = 0.0 if pd.isna(current_confidence) else float(current_confidence)
            reviewed.at[row_index, "confidence"] = max(confidence, 0.95)
        elif reviewed.at[row_index, "review_status"] == "user_approved":
            reviewed.at[row_index, "validation_status"] = "needs_review"
            reviewed.at[row_index, "review_status"] = "needs_review"

    return prepare_reviewed_mapped_export(reviewed)


def restore_mapped_review_original_values(mapped_df: pd.DataFrame | None) -> pd.DataFrame:
    """Undo applied value corrections and clear user approval flags."""
    reviewed = prepare_reviewed_mapped_export(mapped_df)
    if reviewed.empty:
        return reviewed

    for row_index, row in reviewed.iterrows():
        original_value = row.get("original_value")
        if bool(row.get("edited_by_user")) and not _is_missing_review_value(original_value):
            restored_val = _coerce_review_number(original_value)
            reviewed.at[row_index, "value"] = restored_val
            reviewed.at[row_index, "normalized_value"] = restored_val
            reviewed.at[row_index, "original_value"] = ""
            reviewed.at[row_index, "normalization_method"] = "manual_complex_mapping"
        reviewed.at[row_index, "edited_by_user"] = False
        reviewed.at[row_index, "approved_by_user"] = False
        if reviewed.at[row_index, "review_status"] == "user_approved":
            reviewed.at[row_index, "review_status"] = "needs_review"
            reviewed.at[row_index, "validation_status"] = "needs_review"
        reviewed.at[row_index, "warnings"] = _remove_review_warning(
            reviewed.at[row_index, "warnings"],
            CORRECTED_BY_USER_WARNING,
        )

    return prepare_reviewed_mapped_export(reviewed)


def _reviewed_row_by_uid(reviewed_df: pd.DataFrame | None) -> dict[str, pd.Series]:
    reviewed = prepare_reviewed_mapped_export(reviewed_df)
    if reviewed.empty or REVIEW_ROW_UID_COLUMN not in reviewed.columns:
        return {}
    return {
        str(row[REVIEW_ROW_UID_COLUMN]): row
        for _, row in reviewed.iterrows()
        if not _is_missing_review_value(row.get(REVIEW_ROW_UID_COLUMN))
    }


def review_editor_has_unsaved_changes(
    reviewed_df: pd.DataFrame | None,
    editor_df: pd.DataFrame | None,
) -> bool:
    """Return True when the editor draft differs from the applied reviewed rows."""
    if editor_df is None or editor_df.empty:
        return False

    reviewed_by_uid = _reviewed_row_by_uid(reviewed_df)
    for _, editor_row in editor_df.iterrows():
        row_uid = str(editor_row.get(REVIEW_ROW_UID_COLUMN) or "")
        if not row_uid or row_uid not in reviewed_by_uid:
            return True
        applied_row = reviewed_by_uid[row_uid]
        for column in MAPPED_REVIEW_EDITABLE_COLUMNS:
            if column not in editor_row or column not in applied_row:
                continue
            if column == "approved_by_user":
                if _review_bool(editor_row[column]) != _review_bool(applied_row[column]):
                    return True
            elif column == "review_comment":
                if str(editor_row[column] or "") != str(applied_row[column] or ""):
                    return True
            elif not _review_values_equal(applied_row[column], editor_row[column]):
                return True
    return False


def mapped_review_editor_summary(
    reviewed_df: pd.DataFrame | None,
    editor_df: pd.DataFrame | None,
) -> dict[str, int]:
    """Return draft editor metrics without changing applied review statuses."""
    if editor_df is None or editor_df.empty:
        return {
            "rows_in_editor": 0,
            "changed_values": 0,
            "marked_for_approval": 0,
            "comments": 0,
        }

    reviewed_by_uid = _reviewed_row_by_uid(reviewed_df)
    changed_values = 0
    for _, editor_row in editor_df.iterrows():
        row_uid = str(editor_row.get(REVIEW_ROW_UID_COLUMN) or "")
        applied_row = reviewed_by_uid.get(row_uid)
        if applied_row is None:
            changed_values += 1
            continue
        for column in MAPPED_REVIEW_CORRECTION_COLUMNS:
            if column in editor_row and column in applied_row and not _review_values_equal(applied_row[column], editor_row[column]):
                changed_values += 1
                break

    comments = (
        editor_df["review_comment"].fillna("").astype(str).str.strip().ne("").sum()
        if "review_comment" in editor_df.columns
        else 0
    )
    approvals = (
        editor_df["approved_by_user"].map(_review_bool).sum()
        if "approved_by_user" in editor_df.columns
        else 0
    )
    return {
        "rows_in_editor": int(len(editor_df)),
        "changed_values": int(changed_values),
        "marked_for_approval": int(approvals),
        "comments": int(comments),
    }


def mapped_review_summary(mapped_df: pd.DataFrame | None) -> dict[str, int]:
    """Return compact human-review metrics for mapped rows."""
    reviewed = prepare_reviewed_mapped_export(mapped_df)
    if reviewed.empty:
        return {
            "total_rows": 0,
            "required_review": 0,
            "edited_by_user": 0,
            "approved_by_user": 0,
            "remaining_unreviewed": 0,
        }

    return {
        "total_rows": int(len(reviewed)),
        "required_review": int(mapped_rows_need_review_mask(reviewed, exclude_user_approved=False).sum()),
        "edited_by_user": int(reviewed["edited_by_user"].map(_review_bool).sum()),
        "approved_by_user": int(reviewed["approved_by_user"].map(_review_bool).sum()),
        "remaining_unreviewed": int(mapped_rows_need_review_mask(reviewed, exclude_user_approved=True).sum()),
    }


def count_rows_with_warnings(*dataframes: pd.DataFrame | None) -> int:
    """Count rows with non-empty warnings across already-built result frames."""
    warning_rows = 0
    for df in dataframes:
        if df is None or df.empty or "warnings" not in df.columns:
            continue
        warning_rows += int(df["warnings"].fillna("").astype(str).str.strip().ne("").sum())
    return warning_rows


def count_review_status_rows(status: str, *dataframes: pd.DataFrame | None) -> int:
    """Count rows with a given review_status across result frames."""
    total = 0
    for df in dataframes:
        if df is None or df.empty or "review_status" not in df.columns:
            continue
        total += int(df["review_status"].fillna("").astype(str).eq(status).sum())
    return total


def audit_trail_coverage(*dataframes: pd.DataFrame | None) -> dict[str, float | int]:
    """Return audit-trail coverage for output rows with source_file, page and evidence_text."""
    frames = [df for df in dataframes if df is not None and not df.empty]
    if not frames:
        return {"audit_rows": 0, "total_rows": 0, "coverage_pct": 0.0}

    audit_df = pd.concat(frames, ignore_index=True, sort=False)
    total_rows = int(len(audit_df))
    if total_rows == 0:
        return {"audit_rows": 0, "total_rows": 0, "coverage_pct": 0.0}

    required_columns = ["source_file", "page", "evidence_text"]
    audit_mask = pd.Series(True, index=audit_df.index)
    for column in required_columns:
        if column not in audit_df.columns:
            audit_mask = audit_mask & False
            continue
        audit_mask = audit_mask & audit_df[column].notna() & audit_df[column].astype(str).str.strip().ne("")

    audit_rows = int(audit_mask.sum())
    return {
        "audit_rows": audit_rows,
        "total_rows": total_rows,
        "coverage_pct": round((audit_rows / total_rows) * 100, 1),
    }


def build_processing_dashboard_summary(
    *,
    processing_time: float,
    file_type: str,
    active_profile: str,
    profile_metadata: dict | None,
    bad_text_layer: bool,
    raw_rows: pd.DataFrame | None,
    raw_table_summary_df: pd.DataFrame | None,
    ocr_result_df: pd.DataFrame | None,
    ocr_candidates_df: pd.DataFrame | None,
    selected_ocr_candidates_df: pd.DataFrame | None,
    structured_rows: pd.DataFrame | None,
    prototype_structured_df: pd.DataFrame | None,
    mapped_complex_df: pd.DataFrame | None,
    reviewed_mapped_df: pd.DataFrame | None,
) -> dict[str, object]:
    """Build document-level dashboard metrics from already available state."""
    raw_rows = raw_rows if raw_rows is not None else pd.DataFrame()
    raw_table_summary_df = raw_table_summary_df if raw_table_summary_df is not None else pd.DataFrame()
    ocr_result_df = ocr_result_df if ocr_result_df is not None else pd.DataFrame()
    ocr_candidates_df = ocr_candidates_df if ocr_candidates_df is not None else pd.DataFrame()
    selected_ocr_candidates_df = (
        selected_ocr_candidates_df if selected_ocr_candidates_df is not None else pd.DataFrame()
    )
    structured_rows = structured_rows if structured_rows is not None else pd.DataFrame()
    prototype_structured_df = prototype_structured_df if prototype_structured_df is not None else pd.DataFrame()
    mapped_complex_df = mapped_complex_df if mapped_complex_df is not None else pd.DataFrame()
    reviewed_mapped_df = prepare_reviewed_mapped_export(reviewed_mapped_df)

    prototype_simple_rows, prototype_complex_rows = split_prototype_rows(prototype_structured_df)
    review_summary = mapped_review_summary(reviewed_mapped_df)
    profile_metadata = profile_metadata or {}

    if not ocr_result_df.empty:
        ocr_status = "выполнен"
    elif bad_text_layer:
        ocr_status = "рекомендован"
    else:
        ocr_status = "не требовался"

    table_count = 0
    if raw_table_summary_df is not None and not raw_table_summary_df.empty:
        table_count = int(len(raw_table_summary_df))
    elif raw_rows is not None and not raw_rows.empty and "table_id" in raw_rows.columns:
        table_count = int(raw_rows["table_id"].dropna().astype(str).nunique())

    final_output_frames = [
        structured_rows,
        prototype_simple_rows,
        reviewed_mapped_df if not reviewed_mapped_df.empty else mapped_complex_df,
    ]
    audit_coverage = audit_trail_coverage(*final_output_frames)
    rows_with_warnings = count_rows_with_warnings(*final_output_frames)
    auto_approved_rows = count_review_status_rows("auto_approved", *final_output_frames)
    user_approved_rows = int(review_summary["approved_by_user"])
    rows_still_need_review = int(review_summary["remaining_unreviewed"])
    if not structured_rows.empty and "validation_status" in structured_rows.columns:
        rows_still_need_review += int(
            structured_rows["validation_status"].fillna("").astype(str).isin(["warning", "failed"]).sum()
        )
    if not prototype_simple_rows.empty:
        rows_still_need_review += int(prototype_warning_mask(prototype_simple_rows).sum())

    has_exportable_rows = any(frame is not None and not frame.empty for frame in final_output_frames)
    has_candidates = bool(
        (ocr_candidates_df is not None and not ocr_candidates_df.empty)
        or (raw_table_summary_df is not None and not raw_table_summary_df.empty)
    )
    no_raw_or_candidates = (
        (raw_rows is None or raw_rows.empty)
        and (ocr_candidates_df is None or ocr_candidates_df.empty)
        and (raw_table_summary_df is None or raw_table_summary_df.empty)
    )

    if no_raw_or_candidates and structured_rows.empty and not has_exportable_rows:
        readiness_status = "Не удалось извлечь данные"
        readiness_level = "error"
        readiness_comment = "Нет raw rows, таблиц или OCR-кандидатов для дальнейшей обработки."
    elif bad_text_layer and ocr_result_df.empty:
        readiness_status = "Нужен OCR"
        readiness_level = "warning"
        readiness_comment = "Текстовый слой плохой, OCR ещё не выполнялся."
    elif rows_still_need_review > 0:
        readiness_status = "Нужна ручная проверка"
        readiness_level = "warning"
        readiness_comment = "Есть строки, которые не приняты автоматически."
    elif has_exportable_rows:
        readiness_status = "Готово к экспорту"
        readiness_level = "success"
        readiness_comment = "Доступны clean/prototype/reviewed rows для выгрузки."
    elif structured_rows.empty and has_candidates:
        readiness_status = "Нужна настройка профиля"
        readiness_level = "info"
        readiness_comment = "Структурированных production rows нет, но есть кандидаты для source profile."
    else:
        readiness_status = "Нет данных для экспорта"
        readiness_level = "info"
        readiness_comment = "Сначала выполните OCR, настройку профиля или mapping."

    return {
        "processing_time": float(processing_time or 0.0),
        "file_type": file_type or "нет данных",
        "active_profile": active_profile or "нет данных",
        "profile_confidence": float(profile_metadata.get("profile_confidence") or 0.0),
        "ocr_status": ocr_status,
        "table_count": int(table_count),
        "ocr_candidates_count": int(len(ocr_candidates_df)),
        "strong_ocr_candidates_count": int(count_strong_ocr_candidates(ocr_candidates_df)),
        "selected_ocr_candidates_count": int(len(selected_ocr_candidates_df)),
        "structured_rows_count": int(len(structured_rows)),
        "prototype_simple_rows_count": int(len(prototype_simple_rows)),
        "prototype_complex_rows_count": int(len(prototype_complex_rows)),
        "mapped_rows_count": int(len(mapped_complex_df)),
        "reviewed_mapped_rows_count": int(len(reviewed_mapped_df)),
        "auto_approved_rows": int(auto_approved_rows),
        "user_approved_rows": int(user_approved_rows),
        "rows_still_need_review": int(rows_still_need_review),
        "rows_with_warnings": int(rows_with_warnings),
        "audit_coverage": audit_coverage,
        "readiness_status": readiness_status,
        "readiness_level": readiness_level,
        "readiness_comment": readiness_comment,
        "has_exportable_rows": bool(has_exportable_rows),
    }


def build_processing_funnel(summary: dict[str, object], *, bad_text_layer: bool) -> pd.DataFrame:
    """Return a readable processing funnel table for the final dashboard."""
    profile_confidence = float(summary.get("profile_confidence") or 0.0)
    rows_still_need_review = int(summary.get("rows_still_need_review") or 0)
    mapped_rows_count = int(summary.get("mapped_rows_count") or 0)
    reviewed_mapped_rows_count = int(summary.get("reviewed_mapped_rows_count") or 0)
    selected_ocr_count = int(summary.get("selected_ocr_candidates_count") or 0)
    ocr_candidates_count = int(summary.get("ocr_candidates_count") or 0)
    prototype_total = int(summary.get("prototype_simple_rows_count") or 0) + int(
        summary.get("prototype_complex_rows_count") or 0
    )

    profile_status = "ok"
    profile_comment = f"Профиль: {summary.get('active_profile')}"
    if str(summary.get("active_profile") or "") == "generic_pdf" or profile_confidence < PROFILE_PARSER_CONFIDENCE_THRESHOLD:
        profile_status = "warning"
        profile_comment = "Профиль низкой уверенности или универсальный generic_pdf."

    ocr_status_value = str(summary.get("ocr_status") or "нет данных")
    if ocr_status_value == "выполнен":
        ocr_stage_status = "ok"
        ocr_comment = f"Найдено OCR-кандидатов: {ocr_candidates_count}."
    elif bad_text_layer:
        ocr_stage_status = "warning"
        ocr_comment = "Текстовый слой плохой, OCR рекомендован."
    else:
        ocr_stage_status = "info"
        ocr_comment = "OCR не требовался для текущего документа."

    selected_status = "ok" if selected_ocr_count else ("warning" if ocr_candidates_count else "info")
    selected_comment = (
        "Выбранные блоки доступны для черновика профиля."
        if selected_ocr_count
        else "Блоки ещё не выбраны или OCR-кандидатов нет."
    )

    prototype_status = "ok" if prototype_total else "info"
    prototype_comment = (
        "Prototype parser построил строки."
        if prototype_total
        else "Prototype parser ещё не запускался или не дал строк."
    )

    mapping_status = "info"
    mapping_comment = "Complex mapping ещё не применялся."
    if mapped_rows_count:
        mapping_status = "warning" if rows_still_need_review else "ok"
        mapping_comment = (
            "Часть mapped rows требует проверки."
            if rows_still_need_review
            else "Mapped rows готовы."
        )

    human_review_status = "info"
    human_review_comment = "Human review ещё не применялся."
    if reviewed_mapped_rows_count:
        human_review_status = "warning" if rows_still_need_review else "ok"
        human_review_comment = (
            f"Подтверждено пользователем: {int(summary.get('user_approved_rows') or 0)}; "
            f"осталось проверить: {rows_still_need_review}."
        )

    export_status = "ok" if summary.get("has_exportable_rows") else "warning"
    export_comment = (
        "CSV/XLSX доступны для найденных result rows."
        if summary.get("has_exportable_rows")
        else "Нет result rows для финальной выгрузки."
    )

    return pd.DataFrame(
        [
            {
                "stage": "1. Загружен документ",
                "count": 1,
                "status": "ok",
                "comment": f"{str(summary.get('file_type') or 'нет данных').upper()} принят.",
            },
            {
                "stage": "2. Определён профиль",
                "count": summary.get("active_profile") or "нет данных",
                "status": profile_status,
                "comment": profile_comment,
            },
            {
                "stage": "3. Извлечён текстовый слой",
                "count": int(summary.get("table_count") or 0),
                "status": "warning" if bad_text_layer else "ok",
                "comment": "Качество плохое, нужен OCR." if bad_text_layer else "Текстовый слой пригоден.",
            },
            {
                "stage": "4. OCR выполнен",
                "count": summary.get("ocr_status") or "нет данных",
                "status": ocr_stage_status,
                "comment": ocr_comment,
            },
            {
                "stage": "5. Выбраны блоки профиля",
                "count": selected_ocr_count,
                "status": selected_status,
                "comment": selected_comment,
            },
            {
                "stage": "6. Prototype parser",
                "count": prototype_total,
                "status": prototype_status,
                "comment": prototype_comment,
            },
            {
                "stage": "7. Complex mapping",
                "count": mapped_rows_count,
                "status": mapping_status,
                "comment": mapping_comment,
            },
            {
                "stage": "8. Human review",
                "count": int(summary.get("user_approved_rows") or 0),
                "status": human_review_status,
                "comment": human_review_comment,
            },
            {
                "stage": "9. Export",
                "count": "available" if summary.get("has_exportable_rows") else "нет данных",
                "status": export_status,
                "comment": export_comment,
            },
        ],
        columns=["stage", "count", "status", "comment"],
    )


def _stable_export_value(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False, sort_keys=True, default=str)
    if _is_missing_review_value(value):
        return ""
    return str(value)


def dataframe_export_cache_key(df: pd.DataFrame | None) -> str:
    """Return a stable key for session-state export byte caching."""
    if df is None:
        return "none"
    if df.empty:
        return hashlib.md5("|".join(df.columns).encode("utf-8")).hexdigest()
    stable_df = df.copy()
    for column in stable_df.columns:
        stable_df[column] = stable_df[column].map(_stable_export_value)
    payload = stable_df.to_csv(index=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def export_to_csv_cached(df: pd.DataFrame) -> bytes:
    """Return CSV bytes, cached by DataFrame content for the current Streamlit session."""
    cache_key = f"export_bytes:csv:{dataframe_export_cache_key(df)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = export_to_csv(df)
    return st.session_state[cache_key]


def export_to_excel_cached(df: pd.DataFrame) -> bytes:
    """Return XLSX bytes, cached by DataFrame content for the current Streamlit session."""
    cache_key = f"export_bytes:xlsx:{dataframe_export_cache_key(df)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = export_to_excel(df)
    return st.session_state[cache_key]


def select_raw_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a user-facing raw extraction export with Russian labels."""
    raw_export = select_existing_columns(df, RAW_EXPORT_COLUMNS)
    return rename_columns_for_ui(translate_status_columns(raw_export))


def select_technical_raw_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the technical raw extraction schema used for profile setup."""
    return select_existing_columns(df, TECHNICAL_RAW_EXPORT_COLUMNS)


def format_table_summary_for_ui(table_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return table summary with concise Russian headers."""
    if table_summary_df.empty:
        return table_summary_df
    
    df = table_summary_df.copy()
    # Ensure numeric columns are consistently typed
    for col in ["page", "raw_rows_count", "column_count", "rows_count", "columns_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            
    return select_existing_columns(df, TABLE_SUMMARY_COLUMNS).rename(columns=TABLE_SUMMARY_LABELS_RU)


def select_profile_candidate_export_columns(table_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return candidate table export columns for profile creation."""
    return select_existing_columns(table_summary_df, PROFILE_CANDIDATE_EXPORT_COLUMNS)


def select_ocr_export_columns(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """Return raw OCR export columns, separate from clean structured data."""
    return select_existing_columns(ocr_df, OCR_EXPORT_COLUMNS)


def prepare_ocr_candidates_for_ui(ocr_candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Return OCR candidates with current score/type fields and table-first ordering."""
    if ocr_candidates_df is None or ocr_candidates_df.empty:
        return pd.DataFrame(columns=OCR_CANDIDATE_EXPORT_COLUMNS)

    candidates = ocr_candidates_df.copy()
    fallback_score = (
        pd.to_numeric(candidates["score"], errors="coerce").fillna(0)
        if "score" in candidates.columns
        else pd.Series([0.0] * len(candidates), index=candidates.index)
    )
    if "table_score" not in candidates.columns:
        candidates["table_score"] = fallback_score
    if "information_score" not in candidates.columns:
        candidates["information_score"] = fallback_score
    candidates["table_score"] = pd.to_numeric(candidates["table_score"], errors="coerce").fillna(0.0)
    candidates["information_score"] = pd.to_numeric(candidates["information_score"], errors="coerce").fillna(0.0)
    
    # Ensure numeric columns are consistently typed to avoid Arrow serialization errors
    for col in ["rows_count", "numbers_count", "page"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(0).astype(int)

    if "ocr_block_id" not in candidates.columns:
        candidates["ocr_block_id"] = [f"ocr_candidate_{index + 1}" for index in range(len(candidates))]
    if "candidate_type" not in candidates.columns:
        candidates["candidate_type"] = candidates["table_score"].apply(lambda value: "table" if value >= 0.6 else "unknown")
    candidates["candidate_type"] = candidates["candidate_type"].fillna("unknown").astype(str)
    type_rank = candidates["candidate_type"].map(OCR_CANDIDATE_TYPE_ORDER).fillna(99)
    candidates = (
        candidates.assign(_type_rank=type_rank)
        .sort_values(["_type_rank", "table_score", "information_score"], ascending=[True, False, False], kind="mergesort")
        .drop(columns=["_type_rank"])
    )
    return candidates.reset_index(drop=True)


def get_best_ocr_table_candidates(ocr_candidates_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return selected-quality OCR candidates that look like real tables."""
    candidates = prepare_ocr_candidates_for_ui(ocr_candidates_df)
    if candidates.empty:
        return candidates
    return candidates.loc[
        candidates["candidate_type"].eq("table") & candidates["table_score"].ge(OCR_BEST_TABLE_SCORE_THRESHOLD)
    ].copy()


def filter_ocr_candidates_by_type(ocr_candidates_df: pd.DataFrame, candidate_filter: str) -> pd.DataFrame:
    """Filter OCR candidates by the UI type buckets."""
    candidates = prepare_ocr_candidates_for_ui(ocr_candidates_df)
    if candidate_filter == "table":
        return candidates.loc[candidates["candidate_type"].eq("table")].copy()
    if candidate_filter == "paragraph":
        return candidates.loc[candidates["candidate_type"].eq("paragraph")].copy()
    if candidate_filter == "chart_or_mixed":
        return candidates.loc[candidates["candidate_type"].isin(["chart_text", "mixed"])].copy()
    return candidates


def format_ocr_candidates_for_ui(ocr_candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Return OCR table candidates with concise Russian headers."""
    candidates = prepare_ocr_candidates_for_ui(ocr_candidates_df)
    if candidates.empty:
        return candidates
    display_df = select_existing_columns(candidates, OCR_CANDIDATE_DISPLAY_COLUMNS)
    if "candidate_type" in display_df.columns:
        display_df["candidate_type"] = (
            display_df["candidate_type"].map(OCR_CANDIDATE_TYPE_LABELS_RU).fillna(display_df["candidate_type"])
        )
    return display_df.rename(columns=OCR_CANDIDATE_LABELS_RU)


def select_ocr_candidate_export_columns(ocr_candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Return OCR table candidates for profile setup, separate from clean data."""
    return select_existing_columns(prepare_ocr_candidates_for_ui(ocr_candidates_df), OCR_CANDIDATE_EXPORT_COLUMNS)


PROFILE_BUILDER_STEPS = [
    "1. Источник данных",
    "2. Таблицы/блоки",
    "3. Строки",
    "4. Колонки",
    "5. LLM генератор",
    "6. Preview",
    "7. Сохранение",
]

PROFILE_BUILDER_ROLE_OPTIONS = [
    ("Не использовать", "ignore"),
    ("Код / номер строки", "code"),
    ("Наименование", "name"),
    ("Единица измерения", "unit"),
    ("Значение", "value"),
    ("Дата", "date"),
    ("Год", "year"),
    ("Страна", "country"),
    ("Регион", "region"),
    ("Категория", "category"),
    ("Комментарий", "comment"),
]

PROFILE_BUILDER_ROLE_LABELS = {value: label for label, value in PROFILE_BUILDER_ROLE_OPTIONS}
PROFILE_BUILDER_ROLE_VALUES = [value for _, value in PROFILE_BUILDER_ROLE_OPTIONS]

PROFILE_BUILDER_PREVIEW_COLUMNS = [
    "source_kind",
    "page",
    "source_row_id",
    "code",
    "name",
    "commodity",
    "metric",
    "scenario",
    "year",
    "value",
    "unit",
    "currency",
    "evidence_text",
    "extraction_method",
    "validation_status",
    "warnings",
]


def profile_builder_source_row_key(source_row: dict) -> str:
    """Return a stable row id used by manual row selection in user profiles."""
    return source_row_uid(source_row)


def profile_builder_has_numeric_value(source_row: dict) -> bool:
    return any(normalize_user_number(cell) is not None for cell in source_row.get("cells") or [])


def profile_builder_quality_label(row: pd.Series) -> str:
    score = row.get("table_score")
    quality = str(row.get("text_layer_quality") or "").strip()
    if pd.notna(score):
        try:
            return f"{float(score):.2f}" + (f" / {quality}" if quality else "")
        except (TypeError, ValueError):
            pass
    return quality or "неизвестно"


PROFILE_BUILDER_USE_BLOCK_COLUMN = "Использовать эту таблицу"


def profile_builder_table_options(table_catalog_df: pd.DataFrame) -> list[str]:
    if table_catalog_df is None or table_catalog_df.empty:
        return []
    if "block_uid" in table_catalog_df.columns:
        return table_catalog_df["block_uid"].fillna(table_catalog_df.get("table_key", "")).astype(str).tolist()
    return table_catalog_df.get("table_key", pd.Series(dtype="object")).fillna("").astype(str).tolist()


def profile_builder_selected_block_uids_from_editor(
    edited_tables_df: pd.DataFrame,
    table_catalog_df: pd.DataFrame,
) -> list[str]:
    if edited_tables_df is None or edited_tables_df.empty or table_catalog_df is None or table_catalog_df.empty:
        return []
    if PROFILE_BUILDER_USE_BLOCK_COLUMN not in edited_tables_df.columns:
        return []

    block_uid_by_alias: dict[str, str] = {}
    for _, catalog_row in table_catalog_df.iterrows():
        block_uid = str(catalog_row.get("block_uid") or catalog_row.get("table_key") or "").strip()
        if not block_uid:
            continue
        for alias_column in ("block_uid", "table_key", "block_id", "table_id"):
            alias_value = str(catalog_row.get(alias_column) or "").strip()
            if alias_value:
                block_uid_by_alias[alias_value] = block_uid
        block_uid_by_alias[block_uid] = block_uid

    selected_block_uids: list[str] = []
    selected_rows_df = edited_tables_df.loc[
        edited_tables_df[PROFILE_BUILDER_USE_BLOCK_COLUMN].fillna(False).astype(bool)
    ]
    for _, selected_row in selected_rows_df.iterrows():
        selected_block_uid = ""
        for id_column in ("block_uid", "table_key", "block_id", "table_id"):
            candidate = str(selected_row.get(id_column) or "").strip()
            if candidate and candidate in block_uid_by_alias:
                selected_block_uid = block_uid_by_alias[candidate]
                break
        if selected_block_uid and selected_block_uid not in selected_block_uids:
            selected_block_uids.append(selected_block_uid)
    return selected_block_uids


def prepare_profile_builder_catalog_editor(
    table_catalog_df: pd.DataFrame,
    selected_table_keys: list[str],
) -> pd.DataFrame:
    editor_df = table_catalog_df.copy()
    if "block_uid" not in editor_df.columns:
        editor_df["block_uid"] = editor_df.get(
            "table_key",
            pd.Series([""] * len(editor_df), index=editor_df.index),
        ).fillna("").astype(str)
    if "table_key" not in editor_df.columns:
        editor_df["table_key"] = editor_df["block_uid"]
    editor_df[PROFILE_BUILDER_USE_BLOCK_COLUMN] = editor_df["block_uid"].fillna("").astype(str).isin(selected_table_keys)
    editor_df["source_kind"] = editor_df["source_kind"].fillna("").astype(str).replace({"raw_table": "pdf_table"})
    editor_df["Страница"] = editor_df["page"]
    editor_df["block_title"] = editor_df.get("block_title", editor_df["table_id"]).fillna(editor_df["table_id"]).astype(str)
    editor_df["Таблица"] = editor_df["table_id"]
    editor_df["Найдено строк"] = pd.to_numeric(editor_df["rows_count"], errors="coerce")
    editor_df["Найдено колонок"] = pd.to_numeric(editor_df["columns_count"], errors="coerce")
    editor_df["Краткий preview"] = editor_df["preview"].fillna("").astype(str).str.slice(0, 500)
    editor_df["Качество"] = editor_df.apply(profile_builder_quality_label, axis=1)
    return editor_df[
        [
            PROFILE_BUILDER_USE_BLOCK_COLUMN,
            "block_uid",
            "table_key",
            "source_kind",
            "Страница",
            "block_title",
            "Таблица",
            "Найдено строк",
            "Найдено колонок",
            "Краткий preview",
            "Качество",
            "extraction_method",
        ]
    ].copy()


def profile_builder_reconstruction_state_matches(
    applied_state: dict,
    *,
    builder_source: str | None = None,
    selected_block_uids: list[str] | None = None,
) -> bool:
    if builder_source is not None and applied_state.get("source") != builder_source:
        return False
    if selected_block_uids is not None and list(applied_state.get("block_uids") or []) != list(selected_block_uids):
        return False
    return True


def profile_builder_reconstruction_config(
    document_key: str,
    *,
    builder_source: str | None = None,
    selected_block_uids: list[str] | None = None,
) -> dict[str, Any]:
    applied_state = st.session_state.get(f"profile_builder_table_reconstruction_applied:{document_key}") or {}
    if not profile_builder_reconstruction_state_matches(
        applied_state,
        builder_source=builder_source,
        selected_block_uids=selected_block_uids,
    ):
        return {"method": "none"}
    
    method = str(applied_state.get("method") or "none").strip()
    if method == "pair_name_row_with_following_value_row":
        return {"method": "pair_name_row_with_following_value_row"}
        
    pattern = str(applied_state.get("pattern") or "").strip()
    if method == "split_by_regex" and pattern:
        return {"method": "split_by_regex", "pattern": pattern}
        
    return {"method": "none"}


def profile_builder_applied_reconstruction_pattern(
    document_key: str,
    *,
    builder_source: str | None = None,
    selected_block_uids: list[str] | None = None,
) -> str:
    applied_state = st.session_state.get(f"profile_builder_table_reconstruction_applied:{document_key}") or {}
    if not profile_builder_reconstruction_state_matches(
        applied_state,
        builder_source=builder_source,
        selected_block_uids=selected_block_uids,
    ):
        return ""
    return str(applied_state.get("pattern") or "")


def profile_builder_corrected_rows_from_state(
    document_key: str,
    *,
    builder_source: str,
    selected_block_uids: list[str],
    fallback_rows: list[dict],
) -> list[dict]:
    applied_state = st.session_state.get(f"profile_builder_table_reconstruction_applied:{document_key}") or {}
    if profile_builder_reconstruction_state_matches(
        applied_state,
        builder_source=builder_source,
        selected_block_uids=selected_block_uids,
    ) and isinstance(applied_state.get("rows"), list):
        return list(applied_state.get("rows") or [])
    return list(fallback_rows)


def profile_builder_default_source(
    *,
    bad_text_layer: bool,
    has_pdf_tables: bool,
    has_ocr_candidates: bool,
) -> str:
    if has_pdf_tables and has_ocr_candidates:
        return "mixed"
    if has_ocr_candidates or bad_text_layer:
        return "ocr"
    return "pdf_text_layer"


def profile_builder_source_state_key(document_key: str) -> str:
    return f"profile_builder_source:{document_key}"


def profile_builder_source_widget_key(document_key: str) -> str:
    return f"profile_builder_source_widget:{document_key}"


def profile_builder_get_source(session_state, document_key: str, default_source: str) -> str:
    return str(session_state.get(profile_builder_source_state_key(document_key)) or default_source)


def profile_builder_catalog_for_source(
    all_table_catalog_df: pd.DataFrame,
    builder_source: str,
    stored_ocr_catalog_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    all_table_catalog_df = all_table_catalog_df if all_table_catalog_df is not None else pd.DataFrame()
    stored_ocr_catalog_df = stored_ocr_catalog_df if stored_ocr_catalog_df is not None else pd.DataFrame()
    if builder_source == "pdf_text_layer":
        return all_table_catalog_df.loc[
            all_table_catalog_df.get("source_kind", pd.Series(dtype="object")).fillna("").astype(str).eq("raw_table")
        ].copy()
    if builder_source == "ocr":
        catalog_df = all_table_catalog_df.loc[
            all_table_catalog_df.get("source_kind", pd.Series(dtype="object")).fillna("").astype(str).eq("ocr_candidate")
        ].copy()
        if catalog_df.empty and not stored_ocr_catalog_df.empty:
            return stored_ocr_catalog_df.loc[
                stored_ocr_catalog_df.get("source_kind", pd.Series(dtype="object")).fillna("").astype(str).eq("ocr_candidate")
            ].copy()
        return catalog_df
    if not stored_ocr_catalog_df.empty:
        known_keys = set(profile_builder_table_options(all_table_catalog_df))
        extra_ocr_catalog_df = stored_ocr_catalog_df.loc[
            ~stored_ocr_catalog_df.get("block_uid", pd.Series(dtype="object")).fillna("").astype(str).isin(known_keys)
        ].copy()
        if not extra_ocr_catalog_df.empty:
            return pd.concat([all_table_catalog_df, extra_ocr_catalog_df], ignore_index=True)
    return all_table_catalog_df.copy()


def profile_builder_source_label(source: str) -> str:
    engine_name = "Tesseract"
    try:
        # Try to find which document is active to get its engine
        # This is a bit of a hack but works for the current Streamlit structure
        for key in st.session_state.keys():
            if key.startswith("profile_builder_ocr_engine:"):
                engine_name = str(st.session_state.get(key, "Tesseract")).capitalize()
                break
    except Exception:
        pass

    return {
        "pdf_text_layer": "Текстовый слой PDF / pdfplumber",
        "ocr": f"OCR / {engine_name}",
        "mixed": "Смешанный режим",
    }.get(source, source)


def profile_builder_extraction_config(document_key: str) -> dict[str, object]:
    source = str(st.session_state.get(f"profile_builder_source:{document_key}") or "pdf_text_layer")
    ocr_lang = str(st.session_state.get(f"profile_builder_ocr_lang:{document_key}") or "rus+eng")
    ocr_pages = str(st.session_state.get(f"profile_builder_ocr_pages:{document_key}") or "auto")
    ocr_dpi = int(st.session_state.get(f"profile_builder_ocr_dpi:{document_key}") or 300)
    ocr_engine = str(st.session_state.get(f"profile_builder_ocr_engine:{document_key}", "tesseract")).lower()
    
    if source == "ocr":
        return {
            "source": "ocr",
            "ocr": {
                "required": True,
                "engine": ocr_engine,
                "lang": ocr_lang,
                "pages": ocr_pages or "auto",
                "dpi": ocr_dpi,
            },
        }
    if source == "mixed":
        return {
            "source": "mixed",
            "primary": "pdfplumber",
            "fallback": ocr_engine,
            "pdf_engine": "pdfplumber",
            "ocr": {
                "required": False,
                "engine": ocr_engine,
                "lang": ocr_lang,
                "pages": ocr_pages or "auto",
                "dpi": ocr_dpi,
            },
        }
    return {
        "source": "pdf_text_layer",
        "pdf_engine": "pdfplumber",
        "ocr": {"required": False, "engine": None, "lang": None},
    }


def profile_builder_max_columns(source_rows: list[dict]) -> int:
    return min(max(max((len(row.get("cells") or []) for row in source_rows), default=5), 1), 12)


def profile_builder_column_samples(source_rows: list[dict], column_index: int, limit: int = 8) -> list[str]:
    samples: list[str] = []
    for source_row in source_rows:
        cells = source_row.get("cells") or []
        if column_index - 1 >= len(cells):
            continue
        value = str(cells[column_index - 1]).strip()
        if value and value not in samples:
            samples.append(value)
        if len(samples) >= limit:
            break
    return samples


def profile_builder_default_role(column_index: int) -> str:
    defaults = {1: "code", 2: "name", 3: "unit", 4: "value", 5: "value"}
    return defaults.get(column_index, "ignore")


def profile_builder_default_scenario(column_index: int) -> str:
    if column_index == 4:
        return "direct"
    if column_index == 5:
        return "intraport_movement"
    return ""


def build_profile_builder_column_mapping(document_key: str, max_columns: int) -> dict[str, dict[str, object]]:
    column_mapping: dict[str, dict[str, object]] = {}
    for column_index in range(1, max_columns + 1):
        role = str(
            st.session_state.get(
                f"profile_builder_wizard_role:{document_key}:{column_index}",
                profile_builder_default_role(column_index),
            )
            or "ignore"
        )
        if role not in PROFILE_BUILDER_ROLE_VALUES:
            role = "ignore"
        column_key = f"column_{column_index}"
        config: dict[str, object] = {
            "role": role,
            "output_field": role if role != "ignore" else "",
        }
        if role == "value":
            year_text = str(st.session_state.get(f"profile_builder_wizard_year:{document_key}:{column_index}") or "").strip()
            scenario = str(
                st.session_state.get(
                    f"profile_builder_wizard_scenario:{document_key}:{column_index}",
                    profile_builder_default_scenario(column_index),
                )
                or ""
            ).strip()
            config.update(
                {
                    "metric": str(
                        st.session_state.get(f"profile_builder_wizard_metric:{document_key}:{column_index}") or "tariff"
                    ).strip()
                    or "tariff",
                    "scenario": scenario or None,
                    "tariff_type": scenario or None,
                    "value_type": str(
                        st.session_state.get(f"profile_builder_wizard_value_type:{document_key}:{column_index}") or "numeric"
                    ),
                    "year": int(year_text) if year_text.isdigit() else None,
                    "unit_override": str(
                        st.session_state.get(f"profile_builder_wizard_unit:{document_key}:{column_index}") or ""
                    ).strip()
                    or None,
                    "currency_override": str(
                        st.session_state.get(f"profile_builder_wizard_currency:{document_key}:{column_index}") or ""
                    ).strip()
                    or None,
                }
            )
        column_mapping[column_key] = config
    return column_mapping


def profile_builder_uses_token_mapping(source_rows: list[dict]) -> bool:
    if not source_rows:
        return False
    source_kinds = {str(row.get("source_kind") or "") for row in source_rows}
    has_ocr = "ocr_candidate" in source_kinds
    max_columns = max((len(row.get("cells") or []) for row in source_rows), default=0)
    max_tokens = max((len(row.get("numeric_tokens") or []) for row in source_rows), default=0)
    
    # If it's OCR and poorly structured (few columns), always allow token mapping.
    # Also allow if it's OCR and we see some numbers.
    return bool(has_ocr and (max_columns <= 2 or max_tokens >= 1))


def profile_builder_max_tokens(source_rows: list[dict]) -> int:
    return min(max((len(row.get("numeric_tokens") or []) for row in source_rows), default=0), 12)


def profile_builder_token_samples(source_rows: list[dict], token_index: int, limit: int = 8) -> list[str]:
    samples: list[str] = []
    for source_row in source_rows:
        tokens = source_row.get("numeric_tokens") or []
        if token_index - 1 >= len(tokens):
            continue
        value = str(tokens[token_index - 1]).strip()
        if value and value not in samples:
            samples.append(value)
        if len(samples) >= limit:
            break
    return samples


def build_profile_builder_token_mapping(document_key: str, max_tokens: int) -> dict[str, dict[str, object]]:
    token_mapping: dict[str, dict[str, object]] = {}
    for token_index in range(1, max_tokens + 1):
        role = str(st.session_state.get(f"profile_builder_token_role:{document_key}:{token_index}") or "ignore")
        if role == "ignore":
            continue
        year_text = str(st.session_state.get(f"profile_builder_token_year:{document_key}:{token_index}") or "").strip()
        token_mapping[f"token_{token_index}"] = {
            "enabled": True,
            "role": role,
            "metric": str(st.session_state.get(f"profile_builder_token_metric:{document_key}:{token_index}") or role).strip() or role,
            "scenario": str(st.session_state.get(f"profile_builder_token_scenario:{document_key}:{token_index}") or "").strip() or None,
            "year": int(year_text) if year_text.isdigit() else None,
            "unit": str(st.session_state.get(f"profile_builder_token_unit:{document_key}:{token_index}") or "").strip() or None,
            "currency": str(st.session_state.get(f"profile_builder_token_currency:{document_key}:{token_index}") or "").strip() or None,
        }
    return token_mapping


def build_profile_builder_row_filters(
    document_key: str,
    selected_source_rows: list[str],
) -> dict[str, object]:
    mode = str(st.session_state.get(f"profile_builder_row_mode:{document_key}") or "Ручной выбор строк")
    return {
        "use_manual_rows": mode == "Ручной выбор строк",
        "selected_source_rows": selected_source_rows if mode == "Ручной выбор строк" else [],
        "keep_after": str(st.session_state.get(f"profile_builder_rule_keep_after:{document_key}") or "").strip(),
        "keep_until": str(st.session_state.get(f"profile_builder_rule_keep_until:{document_key}") or "").strip(),
        "keep_numeric_rows_only": bool(st.session_state.get(f"profile_builder_rule_numeric_only:{document_key}", False)),
        "skip_empty_values": bool(st.session_state.get(f"profile_builder_rule_skip_empty_values:{document_key}", True)),
        "skip_dash_values": bool(st.session_state.get(f"profile_builder_rule_skip_dash_values:{document_key}", True)),
    }


def build_profile_builder_config(
    document_key: str,
    selected_block_uids: list[str],
    selected_source_rows: list[str],
    max_columns: int,
    *,
    use_token_mapping: bool = False,
    max_tokens: int = 0,
) -> dict[str, object]:
    profile_name = str(st.session_state.get(f"profile_builder_wizard_name:{document_key}") or "user_source_profile").strip()
    display_name = str(st.session_state.get(f"profile_builder_wizard_display:{document_key}") or "Пользовательский профиль").strip()
    keywords_text = str(st.session_state.get(f"profile_builder_wizard_keywords:{document_key}") or "")
    selector_text = str(st.session_state.get(f"profile_builder_selector_text:{document_key}") or "").strip()
    section_name = str(st.session_state.get(f"profile_builder_wizard_section:{document_key}") or "user_profile_section").strip()
    keywords = [line.strip() for line in keywords_text.splitlines() if line.strip()]
    table_selector: dict[str, object] = {"block_uids": selected_block_uids}
    if selector_text:
        table_selector["text_contains"] = [text.strip() for text in selector_text.splitlines() if text.strip()] or selector_text

    extraction_config = profile_builder_extraction_config(document_key)
    source = str(extraction_config.get("source") or "pdf_text_layer")
    row_selection = build_profile_builder_row_filters(document_key, selected_source_rows)
    row_selection["selected_row_uids"] = selected_source_rows
    column_mapping = build_profile_builder_column_mapping(document_key, max_columns)
    token_mapping = build_profile_builder_token_mapping(document_key, max_tokens) if use_token_mapping else {}
    table_reconstruction = profile_builder_reconstruction_config(
        document_key,
        builder_source=source,
        selected_block_uids=selected_block_uids,
    )
    block_config: dict[str, object] = {
        "selector": table_selector,
        "row_selection": row_selection,
        "table_reconstruction": table_reconstruction,
        "column_mapping": column_mapping,
    }
    if token_mapping:
        block_config["token_mapping"] = token_mapping
    if source == "pdf_text_layer":
        block_config["source_kind"] = "pdf_table"
    elif source == "ocr":
        block_config["source_kind"] = "ocr_candidate"

    # Dynamically determine required fields based on mapped roles
    required_fields = []
    
    # Check column mappings
    for col_config in column_mapping.values():
        role = col_config.get("role")
        if role in {"name", "code", "custom_text"}:
            if "name" not in required_fields:
                required_fields.append("name")
        elif role in {"value", "value_direct", "value_intraport", "percent", "custom_numeric"}:
            if "value" not in required_fields:
                required_fields.append("value")
                
    # Check token mappings
    for token_config in token_mapping.values():
        if token_config.get("enabled"):
            # Token mapping implies value and name (from text part)
            if "name" not in required_fields:
                required_fields.append("name")
            if "value" not in required_fields:
                required_fields.append("value")
                
    # Default fallback if nothing was detected (prevents breaking validation logic if user mapped nothing)
    if not required_fields:
        required_fields = ["name", "value"]

    return {
        "profile_name": profile_name or "user_source_profile",
        "display_name": display_name or profile_name or "Пользовательский профиль",
        "document_match": {"keywords": keywords},
        "extraction": extraction_config,
        "blocks": [block_config],
        "tables": [
            {
                "section_name": section_name or "user_profile_section",
                "table_selector": table_selector,
                "row_filters": row_selection,
                "table_reconstruction": table_reconstruction,
                "column_mapping": column_mapping,
                "token_mapping": token_mapping,
            }
        ],
        "normalization": {"number_format": "ru", "dash_as_null": True},
        "validation": {"required_fields": required_fields, "value_positive": True},
    }


def profile_builder_rows_matching_rules(source_rows: list[dict], document_key: str) -> list[str]:
    keep_after = str(st.session_state.get(f"profile_builder_rule_keep_after:{document_key}") or "").strip()
    keep_until = str(st.session_state.get(f"profile_builder_rule_keep_until:{document_key}") or "").strip()
    numeric_only = bool(st.session_state.get(f"profile_builder_rule_numeric_only:{document_key}", False))
    skip_empty_values = bool(st.session_state.get(f"profile_builder_rule_skip_empty_values:{document_key}", True))
    skip_dash_values = bool(st.session_state.get(f"profile_builder_rule_skip_dash_values:{document_key}", True))

    rows = list(source_rows)
    if keep_after:
        kept: list[dict] = []
        active = False
        for row in rows:
            if not active and keep_after.casefold() in str(row.get("evidence_text") or "").casefold():
                active = True
                continue
            if active:
                kept.append(row)
        rows = kept
    if keep_until:
        kept = []
        for row in rows:
            if keep_until.casefold() in str(row.get("evidence_text") or "").casefold():
                break
            kept.append(row)
        rows = kept
    if numeric_only:
        rows = [row for row in rows if profile_builder_has_numeric_value(row)]
    if skip_empty_values:
        rows = [row for row in rows if any(str(cell).strip() for cell in row.get("cells") or [])]
    if skip_dash_values:
        rows = [
            row
            for row in rows
            if any(str(cell).strip() not in {"", "-", "—", "–"} for cell in row.get("cells") or [])
        ]
    return [profile_builder_source_row_key(row) for row in rows]


def profile_builder_preview_metrics(preview_df: pd.DataFrame, selected_rows_count: int) -> dict[str, int]:
    if preview_df is None or preview_df.empty:
        return {
            "selected_rows": selected_rows_count,
            "output_rows": 0,
            "parsed_values": 0,
            "needs_review": 0,
            "errors": 0,
        }
    validation_status = preview_df.get("validation_status", pd.Series("", index=preview_df.index)).fillna("").astype(str)
    review_status = preview_df.get("review_status", pd.Series("", index=preview_df.index)).fillna("").astype(str)
    values = preview_df.get("value", pd.Series(pd.NA, index=preview_df.index))
    return {
        "selected_rows": selected_rows_count,
        "output_rows": len(preview_df),
        "parsed_values": int(values.notna().sum()),
        "needs_review": int((validation_status.ne("passed") | review_status.eq("needs_review")).sum()),
        "errors": int(validation_status.eq("failed").sum()),
    }


def select_profile_builder_preview_columns(preview_df: pd.DataFrame) -> pd.DataFrame:
    if preview_df is None or preview_df.empty:
        return pd.DataFrame(columns=PROFILE_BUILDER_PREVIEW_COLUMNS)
    return select_existing_columns(preview_df, PROFILE_BUILDER_PREVIEW_COLUMNS)


def count_strong_ocr_candidates(ocr_candidates_df: pd.DataFrame | None) -> int:
    """Count OCR candidates that are strong enough to highlight in the UI."""
    if ocr_candidates_df is None or ocr_candidates_df.empty:
        return 0
    candidates = prepare_ocr_candidates_for_ui(ocr_candidates_df)
    if "candidate_type" in candidates.columns and "table_score" in candidates.columns:
        strong_mask = candidates["candidate_type"].eq("table") & candidates["table_score"].ge(OCR_BEST_TABLE_SCORE_THRESHOLD)
        return int(strong_mask.sum())
    if "score" not in candidates.columns:
        return 0
    return int((pd.to_numeric(candidates["score"], errors="coerce").fillna(0) >= 0.6).sum())


def ocr_candidate_diagnostics(ocr_result_df: pd.DataFrame | None, ocr_candidates_df: pd.DataFrame | None) -> dict[str, object]:
    """Build a compact diagnostic payload for OCR candidate visibility issues."""
    candidates = prepare_ocr_candidates_for_ui(ocr_candidates_df)
    candidate_scores = candidates["table_score"] if not candidates.empty and "table_score" in candidates.columns else pd.Series(dtype="float64")
    titles = (
        candidates["block_title"].dropna().astype(str).head(5).tolist()
        if not candidates.empty and "block_title" in candidates.columns
        else []
    )
    return {
        "ocr_rows_count": 0 if ocr_result_df is None else len(ocr_result_df),
        "ocr_candidates_count": len(candidates),
        "ocr_candidates_columns": list(candidates.columns),
        "max_table_score": None if candidate_scores.empty else float(candidate_scores.max()),
        "strong_table_candidates": count_strong_ocr_candidates(candidates),
        "first_5_block_titles": titles,
    }


def profile_draft_quality_table(profile_draft: dict) -> pd.DataFrame:
    """Return a compact table explaining draft section quality."""
    sections = profile_draft.get("target_sections") or []
    rows = []
    for section in sections:
        warnings = section.get("section_warnings") or []
        rows.append(
            {
                "section_id": section.get("section_id"),
                "title": section.get("title"),
                "page": section.get("page"),
                "section_quality": section.get("section_quality"),
                "section_confidence": section.get("section_confidence"),
                "section_warnings": "; ".join(str(warning) for warning in warnings),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "section_id",
            "title",
            "page",
            "section_quality",
            "section_confidence",
            "section_warnings",
        ],
    )


def infer_profile_from_rows(df, file_type: str) -> str:
    """Infer source profile from extraction result when user selected auto mode."""
    if file_type in {"csv", "xlsx"}:
        return "generic_table"
    if "extraction_method" in df.columns and (df["extraction_method"] == "fish_market_report_parser").any():
        return "fish_market_report"
    if file_type == "pdf":
        return "generic_pdf"
    return "generic_pdf"


def process_document(file_path: Path, file_type: str, profile_override: str | None = None):
    """Run extraction, normalization, and validation for one uploaded document."""
    profile_metadata = {
        "profile_name": "generic_table" if file_type in {"csv", "xlsx"} else "generic_pdf",
        "profile_confidence": 1.0 if file_type in {"csv", "xlsx"} else 0.0,
        "profile_reason": "Для табличных файлов используется универсальная обработка CSV/XLSX.",
        "profile_selection": "auto",
        "selected_extraction_strategy": "pandas" if file_type in {"csv", "xlsx"} else "pdfplumber",
    }

    if file_type in {"csv", "xlsx"}:
        extracted_df = extract_excel(str(file_path))
    elif file_type == "pdf":
        extracted_df = extract_pdf(str(file_path), profile_override=profile_override)
        profile_metadata = extracted_df.attrs.get("profile_detection", profile_metadata)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    raw_extraction_summary = extracted_df.attrs.get("raw_extraction_summary", {})
    raw_table_summary = extracted_df.attrs.get("raw_table_summary", [])
    normalized_df = normalize_dataframe(extracted_df)
    validated_df = validate_extracted_data(normalized_df)
    validated_df.attrs["profile_detection"] = profile_metadata
    validated_df.attrs["raw_extraction_summary"] = raw_extraction_summary
    validated_df.attrs["raw_table_summary"] = raw_table_summary
    return validated_df, profile_metadata


def render_source_profile_card(
    profile_name: str,
    auto_mode: bool,
    profile_metadata: dict | None = None,
    ocr_has_run: bool = False,
    profile_config: dict | None = None,
    document_key: str | None = None,
) -> None:
    """Render source registry metadata for the active source profile."""
    profile_metadata = profile_metadata or {}
    config = profile_config or get_source_config(profile_name)
    display_name = config.get("display_name") or get_display_name(profile_name)
    extraction_config = config.get("extraction") or {}
    extraction_source = extraction_config.get("source")
    strategy = extraction_source or profile_metadata.get("selected_extraction_strategy") or config.get("extraction_strategy") or "-"
    strategy_label = EXTRACTION_STRATEGY_LABELS.get(strategy, str(strategy))
    if extraction_source:
        strategy_label = profile_builder_source_label(str(extraction_source))
        if str(extraction_source) == "ocr" or str(extraction_source) == "mixed":
            ocr_engine = extraction_config.get("ocr", {}).get("engine")
            if ocr_engine:
                strategy_label = f"OCR / {str(ocr_engine).title().replace('Ocr', 'OCR')}"
                if str(extraction_source) == "mixed":
                    strategy_label = f"Mixed (PDF + {strategy_label})"
    update_frequency = config.get("update_frequency") or "-"
    text_layer_info = profile_metadata.get("text_layer_quality") or {}
    bad_text_layer = bool(text_layer_info.get("bad_text_layer"))
    requires_ocr = bool(config.get("requires_ocr"))
    if extraction_config:
        requires_ocr = str(extraction_source) == "ocr" or bool((extraction_config.get("ocr") or {}).get("required"))
    if ocr_has_run:
        ocr_status = "выполнен для выбранных страниц"
    elif bad_text_layer:
        ocr_status = "рекомендуется"
    else:
        ocr_status = "требуется" if requires_ocr else "не требуется"
    uses_llm = "используется" if config.get("uses_llm") else "не используется"
    profile_confidence = float(profile_metadata.get("profile_confidence") or 0.0)
    profile_reason = str(profile_metadata.get("profile_reason") or "Причина определения профиля не передана.")

    st.subheader("Определённый профиль источника")
    st.caption("Профиль определён автоматически" if auto_mode else "Профиль выбран пользователем вручную")

    profile_cols = st.columns(4)
    profile_cols[0].metric("Профиль источника", str(display_name))
    profile_cols[1].metric("Стратегия извлечения", strategy_label)
    profile_cols[2].metric("Уверенность", f"{profile_confidence:.2f}")
    profile_cols[3].metric("Частота обновления", str(update_frequency))

    detail_cols = st.columns(3)
    detail_cols[0].metric("document_profile", str(config.get("document_profile") or profile_name))
    detail_cols[1].metric("OCR", ocr_status)
    detail_cols[2].metric("LLM", uses_llm)

    st.caption("Причина: " + profile_reason)
    action_cols = st.columns(2)
    if action_cols[0].button("Изменить профиль", key=f"change_profile:{document_key or profile_name}"):
        if document_key:
            st.session_state[f"document_mode:{document_key}"] = "choose"
            st.rerun()
    if action_cols[1].button("Редактировать профиль", key=f"edit_profile:{document_key or profile_name}"):
        if document_key:
            st.session_state[f"document_mode:{document_key}"] = "profile_setup"
            st.session_state[f"profile_builder_step:{document_key}"] = "1. Источник данных"
            st.rerun()

    if auto_mode and profile_confidence < PROFILE_PARSER_CONFIDENCE_THRESHOLD:
        st.warning(
            "Профиль документа определён с низкой уверенностью. Используется универсальный extractor. "
            "При необходимости выберите профиль вручную."
        )

    if auto_mode and profile_name == "generic_pdf":
        st.info("Профиль не распознан как профильный источник. Используется универсальный PDF extractor.")


def main() -> None:
    """Render the Streamlit MVP application."""
    st.set_page_config(
        page_title="NAMEX DataFlow",
        layout="wide",
    )

    st.title("NAMEX DataFlow: автоматическая обработка внешних документов")
    st.caption(
        "Загрузите PDF, CSV или XLSX — система извлечёт данные, проверит ошибки "
        "и подготовит таблицу для аналитического продукта."
    )

    uploaded_file = st.file_uploader(
        "Загрузите PDF, CSV или XLSX",
        type=["pdf", "csv", "xlsx"],
    )

    if uploaded_file is None:
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    upload_start_time = time.perf_counter()
    file_type = detect_file_type(uploaded_file.name)
    uploaded_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(uploaded_bytes).hexdigest()
    document_key = f"{uploaded_file.name}:{file_hash}"
    performance_timings: list[dict[str, object]] = []
    performance_timings_key = f"performance_timings:{document_key}"
    saved_path_key = f"saved_upload_path:{document_key}"
    saved_path_value = st.session_state.get(saved_path_key)
    if saved_path_value and Path(saved_path_value).exists():
        saved_path = Path(saved_path_value)
        file_cache_status = "hit"
    else:
        saved_path = RAW_DIR / f"{file_hash[:12]}_{safe_file_name(uploaded_file.name)}"
        saved_path.write_bytes(uploaded_bytes)
        st.session_state[saved_path_key] = str(saved_path)
        file_cache_status = "miss"
    record_performance_timing(
        performance_timings,
        "file upload / file read",
        upload_start_time,
        cache_status=file_cache_status,
        rows=len(uploaded_bytes),
    )
    ocr_result_key = f"ocr_result:{document_key}"
    ocr_candidates_key = f"ocr_candidates:{document_key}"
    profile_builder_ocr_result_key = f"profile_builder_ocr_result:{document_key}"
    profile_builder_ocr_candidates_key = f"profile_builder_ocr_candidates:{document_key}"
    profile_builder_ocr_blocks_catalog_key = f"profile_builder_ocr_blocks_catalog:{document_key}"
    selected_ocr_candidate_ids_key = f"selected_ocr_candidate_ids:{document_key}"
    selected_ocr_candidates_key = f"selected_ocr_candidates:{document_key}"
    profile_draft_key = f"profile_draft:{document_key}"
    profile_draft_good_only_key = f"profile_draft_good_only:{document_key}"
    prototype_structured_key = f"prototype_structured:{document_key}"
    mapped_complex_key = f"mapped_complex:{document_key}"
    reviewed_mapped_complex_key = f"reviewed_mapped_complex:{document_key}"
    mapped_review_unsaved_key = f"mapped_review_unsaved:{document_key}"
    user_profile_structured_key = f"user_profile_structured:{document_key}"
    applied_user_profile_key = f"applied_user_profile:{document_key}"
    user_profile_auto_decision_key = f"user_profile_auto_decision:{document_key}"

    info_col, type_col = st.columns(2)
    info_col.metric("Имя файла", uploaded_file.name)
    type_col.metric("Тип файла", file_type)

    source_registry = load_source_registry()
    user_profiles = load_user_profiles(USER_PROFILES_DIR)
    user_profile_option_prefix = "user::"
    user_profile_options = [f"{user_profile_option_prefix}{profile_name}" for profile_name in user_profiles]
    profile_options = ["__auto__"] + list(source_registry.keys()) + user_profile_options
    selected_profile_key = st.selectbox(
        "Профиль источника",
        profile_options,
        format_func=lambda value: "Определить автоматически"
        if value == "__auto__"
        else (
            "Пользовательский: "
            + str(
                user_profiles.get(value.removeprefix(user_profile_option_prefix), {}).get("display_name")
                or value.removeprefix(user_profile_option_prefix)
            )
            if str(value).startswith(user_profile_option_prefix)
            else str(source_registry.get(value, {}).get("display_name") or value)
        ),
    )
    auto_profile = selected_profile_key == "__auto__"
    selected_is_user_profile = str(selected_profile_key).startswith(user_profile_option_prefix)
    selected_user_profile_name = (
        str(selected_profile_key).removeprefix(user_profile_option_prefix)
        if selected_is_user_profile
        else ""
    )
    selected_user_profile_config = user_profiles.get(selected_user_profile_name)
    profile_override = (
        "generic_pdf"
        if selected_is_user_profile
        else None
        if auto_profile
        else str(source_registry.get(selected_profile_key, {}).get("document_profile") or selected_profile_key)
    )

    if file_type == "unknown":
        st.error("Формат файла не поддерживается.")
        return

    processing_cache_key = f"extraction_result:{document_key}:{profile_override or '__auto__'}"
    cached_processing_result = st.session_state.get(processing_cache_key)
    start_time = time.perf_counter()
    try:
        with st.spinner("Обработка документа..."):
            if cached_processing_result is None:
                validated_df, profile_metadata = process_document(
                    saved_path,
                    file_type,
                    profile_override=profile_override,
                )
            else:
                validated_df = cached_processing_result["validated_df"]
                profile_metadata = cached_processing_result["profile_metadata"]
    except Exception as exc:
        st.error(f"Не удалось обработать документ: {exc}")
        return

    if cached_processing_result is None:
        processing_time = time.perf_counter() - start_time
        st.session_state[processing_cache_key] = {
            "validated_df": validated_df,
            "profile_metadata": profile_metadata,
            "processing_time": processing_time,
        }
        processing_cache_status = "miss"
    else:
        processing_time = float(cached_processing_result.get("processing_time") or 0.0)
        processing_cache_status = "hit"
    record_performance_timing(
        performance_timings,
        "pdf extraction / normalization / validation",
        start_time,
        cache_status=processing_cache_status,
        rows=len(validated_df),
    )
    active_profile = selected_profile_key if not auto_profile else str(
        profile_metadata.get("profile_name") or infer_profile_from_rows(validated_df, file_type)
    )
    record_performance_timing(
        performance_timings,
        "profile detection",
        time.perf_counter(),
        cache_status=processing_cache_status,
    )

    derived_rows_key = f"derived_rows:{document_key}:{profile_override or '__auto__'}:{active_profile}"
    derived_start_time = time.perf_counter()
    cached_derived_rows = st.session_state.get(derived_rows_key)
    if cached_derived_rows is None:
        structured_rows, raw_rows = split_rows_by_extraction_level(validated_df)
        raw_table_rows = (
            raw_rows[raw_rows["section_name"] == "raw_pdf_table"].copy()
            if "section_name" in raw_rows.columns
            else raw_rows.iloc[0:0].copy()
        )
        raw_summary = validated_df.attrs.get("raw_extraction_summary", {})
        is_generic_pdf = active_profile == "generic_pdf"
        text_layer_info = profile_metadata.get("text_layer_quality") or {}
        raw_bad_text_layer = (
            "text_layer_quality" in raw_rows.columns
            and raw_rows["text_layer_quality"].fillna("").astype(str).eq("bad").any()
        )
        bad_text_layer = bool(text_layer_info.get("bad_text_layer") or raw_bad_text_layer)
        raw_table_summary_df = pd.DataFrame(validated_df.attrs.get("raw_table_summary", []))
        if raw_table_summary_df.empty and not raw_table_rows.empty:
            raw_table_summary_df = build_raw_table_summary(raw_table_rows)
        if raw_table_summary_df.empty or "table_score" not in raw_table_summary_df.columns:
            profile_candidates_df = raw_table_summary_df.iloc[0:0].copy()
            ocr_tables_df = raw_table_summary_df.iloc[0:0].copy()
        else:
            table_scores = pd.to_numeric(raw_table_summary_df["table_score"], errors="coerce").fillna(0)
            table_quality = (
                raw_table_summary_df["text_layer_quality"].fillna("ok").astype(str)
                if "text_layer_quality" in raw_table_summary_df.columns
                else pd.Series(["ok"] * len(raw_table_summary_df), index=raw_table_summary_df.index)
            )
            ocr_tables_df = raw_table_summary_df.loc[table_quality == "bad"].copy()
            profile_candidates_df = raw_table_summary_df.loc[(table_scores >= 0.5) & (table_quality != "bad")].copy()
        st.session_state[derived_rows_key] = {
            "structured_rows": structured_rows,
            "raw_rows": raw_rows,
            "raw_table_rows": raw_table_rows,
            "raw_summary": raw_summary,
            "is_generic_pdf": is_generic_pdf,
            "text_layer_info": text_layer_info,
            "bad_text_layer": bad_text_layer,
            "raw_table_summary_df": raw_table_summary_df,
            "profile_candidates_df": profile_candidates_df,
            "ocr_tables_df": ocr_tables_df,
        }
        derived_cache_status = "miss"
    else:
        structured_rows = cached_derived_rows["structured_rows"]
        raw_rows = cached_derived_rows["raw_rows"]
        raw_table_rows = cached_derived_rows["raw_table_rows"]
        raw_summary = cached_derived_rows["raw_summary"]
        is_generic_pdf = cached_derived_rows["is_generic_pdf"]
        text_layer_info = cached_derived_rows["text_layer_info"]
        bad_text_layer = cached_derived_rows["bad_text_layer"]
        raw_table_summary_df = cached_derived_rows["raw_table_summary_df"]
        profile_candidates_df = cached_derived_rows["profile_candidates_df"]
        ocr_tables_df = cached_derived_rows["ocr_tables_df"]
        derived_cache_status = "hit"
    record_performance_timing(
        performance_timings,
        "raw rows / profile candidates preparation",
        derived_start_time,
        cache_status=derived_cache_status,
        rows=len(raw_rows) + len(structured_rows),
    )
    # Engine-specific OCR result keys
    ocr_engine_state_key = f"profile_builder_ocr_engine:{document_key}"
    current_builder_engine = str(st.session_state.get(ocr_engine_state_key, "tesseract")).lower()
    
    ocr_result_key = f"ocr_result:{document_key}:{current_builder_engine}"
    ocr_candidates_key = f"ocr_candidates:{document_key}:{current_builder_engine}"
    
    ocr_document_matches = st.session_state.get("ocr_document_key") == document_key
    ocr_result_df = st.session_state.get(ocr_result_key)
    if ocr_result_df is None:
        ocr_result_df = pd.DataFrame()

    ocr_candidates_df = st.session_state.get(ocr_candidates_key)
    if ocr_candidates_df is None and not ocr_result_df.empty:
        ocr_candidates_start_time = time.perf_counter()
        ocr_candidates_df = extract_ocr_table_candidates(ocr_result_df)
        st.session_state[ocr_candidates_key] = ocr_candidates_df
        st.session_state["ocr_document_key"] = document_key
        record_performance_timing(
            performance_timings,
            f"OCR candidates ({current_builder_engine})",
            ocr_candidates_start_time,
            cache_status="miss",
            rows=len(ocr_candidates_df),
        )
    elif ocr_candidates_df is not None:
        record_performance_timing(
            performance_timings,
            f"OCR candidates ({current_builder_engine})",
            time.perf_counter(),
            cache_status="hit",
            rows=len(ocr_candidates_df),
        )
    if ocr_candidates_df is None:
        ocr_candidates_df = pd.DataFrame()

    # Synchronization with profile builder state
    profile_builder_ocr_result_key = f"profile_builder_ocr_result:{document_key}:{current_builder_engine}"
    profile_builder_ocr_candidates_key = f"profile_builder_ocr_candidates:{document_key}:{current_builder_engine}"
    profile_builder_ocr_blocks_catalog_key = f"profile_builder_ocr_blocks_catalog:{document_key}:{current_builder_engine}"

    if not ocr_candidates_df.empty:
        st.session_state[profile_builder_ocr_candidates_key] = ocr_candidates_df.copy()
        st.session_state[profile_builder_ocr_blocks_catalog_key] = build_profile_table_catalog(
            pd.DataFrame(),
            None,
            ocr_candidates_df,
        )
    if not ocr_result_df.empty:
        st.session_state[profile_builder_ocr_result_key] = ocr_result_df.copy()

    def run_user_profile_ocr(_document: object, ocr_config: dict) -> dict[str, pd.DataFrame]:
        pages_setting = ocr_config.get("pages") or "auto"
        if isinstance(pages_setting, list):
            profile_ocr_pages = [int(page) for page in pages_setting if str(page).isdigit()]
        elif str(pages_setting).strip().lower() == "auto":
            page_values = pd.to_numeric(raw_rows.get("page", pd.Series(dtype="float64")), errors="coerce").dropna()
            if page_values.empty:
                try:
                    total_pages = get_pdf_page_count(str(saved_path))
                    profile_ocr_pages = list(range(1, total_pages + 1))
                except Exception:
                    profile_ocr_pages = [1]
            else:
                profile_ocr_pages = sorted({int(page) for page in page_values.tolist()})
        else:
            profile_ocr_pages = [
                int(page.strip())
                for page in str(pages_setting).split(",")
                if page.strip().isdigit()
            ] or [1]
        
        profile_ocr_lang = str(ocr_config.get("lang") or "rus+eng")
        profile_ocr_engine = str(ocr_config.get("engine") or "tesseract").lower()
        profile_ocr_dpi = int(ocr_config.get("dpi") or 300)
        
        # Engine-specific keys for this run
        run_ocr_result_key = f"ocr_result:{document_key}:{profile_ocr_engine}"
        run_ocr_candidates_key = f"ocr_candidates:{document_key}:{profile_ocr_engine}"
        run_pb_result_key = f"profile_builder_ocr_result:{document_key}:{profile_ocr_engine}"
        run_pb_candidates_key = f"profile_builder_ocr_candidates:{document_key}:{profile_ocr_engine}"
        run_pb_catalog_key = f"profile_builder_ocr_blocks_catalog:{document_key}:{profile_ocr_engine}"

        try:
            engine = get_ocr_engine(profile_ocr_engine)
            settings = OcrSettings(
                lang=profile_ocr_lang,
                dpi=profile_ocr_dpi,
                pages=profile_ocr_pages,
            )
            ocr_results = engine.recognize_pdf(str(saved_path), settings)
            extracted = ocr_page_results_to_dataframe(ocr_results, str(saved_path.name))
        except Exception as e:
            error_str = str(e)
            if "ConvertPirAttribute2RuntimeAttribute" in error_str or "PaddleOCR worker failed" in error_str and "pir::ArrayAttribute" in error_str:
                st.error("PaddleOCR failed due to a known PaddlePaddle CPU inference issue. Try:\n\n`python -m pip uninstall -y paddlepaddle`\n`python -m pip install paddlepaddle==3.2.2`")
            else:
                st.error(f"OCR processing failed ({profile_ocr_engine}): {e}")
            # Clear stale results if OCR failed
            st.session_state.pop(run_ocr_result_key, None)
            st.session_state.pop(run_ocr_candidates_key, None)
            raise

        validated = validate_extracted_data(extracted)

        # Check for empty text in OCR results
        empty_pages = []
        if not validated.empty:
            for _, row in validated.iterrows():
                if not str(row.get("evidence_text") or "").strip():
                    empty_pages.append(int(row.get("page")))

        if empty_pages:
            st.warning(f"{profile_ocr_engine.upper()} OCR вернул пустой текст для страниц: {sorted(empty_pages)}")

        candidates = extract_ocr_table_candidates(validated)
        
        # Store in engine-specific keys
        st.session_state[run_ocr_result_key] = validated
        st.session_state[run_ocr_candidates_key] = candidates
        st.session_state[run_pb_result_key] = validated.copy()
        st.session_state[run_pb_candidates_key] = candidates.copy()
        st.session_state[run_pb_catalog_key] = build_profile_table_catalog(
            pd.DataFrame(),
            None,
            candidates,
        )
        st.session_state["ocr_document_key"] = document_key
        
        # Clean up wizard state
        st.session_state.pop(f"profile_builder_block_selection_applied:{document_key}", None)
        st.session_state.pop(f"profile_builder_block_selection_draft:{document_key}", None)
        st.session_state.pop(f"profile_builder_table_reconstruction_applied:{document_key}", None)
        st.session_state.pop(f"profile_builder_split_pattern_draft:{document_key}", None)
        st.session_state.pop(f"profile_builder_manual_rows:{document_key}", None)
        
        return {"ocr_result_df": validated, "ocr_candidates_df": candidates}

    prototype_structured_df = st.session_state.get(prototype_structured_key)
    if prototype_structured_df is None:
        prototype_structured_df = pd.DataFrame()
    mapped_complex_df = st.session_state.get(mapped_complex_key)
    if mapped_complex_df is None:
        mapped_complex_df = st.session_state.get("mapped_complex_df")
    if mapped_complex_df is None:
        mapped_complex_df = pd.DataFrame()

    applied_user_profile_config = None
    user_profile_structured_df = st.session_state.get(user_profile_structured_key)
    if user_profile_structured_df is None:
        user_profile_structured_df = pd.DataFrame()

    user_profile_candidates = []
    if selected_is_user_profile and selected_user_profile_config:
        user_profile_candidates = [selected_user_profile_config]
    elif active_profile == "generic_pdf" and user_profiles:
        matched_user_profiles = find_matching_user_profiles(
            user_profiles,
            raw_rows=raw_rows,
            ocr_candidates_df=ocr_candidates_df,
        )
        if matched_user_profiles:
            matched_profile = matched_user_profiles[0]
            matched_profile_name = str(matched_profile.get("profile_name") or "")
            matched_display_name = str(matched_profile.get("display_name") or matched_profile_name)
            decision = st.session_state.get(user_profile_auto_decision_key)
            if decision == f"apply:{matched_profile_name}":
                user_profile_candidates = [matched_profile]
            elif decision != "generic":
                st.info(f"Найден пользовательский профиль: {matched_display_name}. Применить?")
                auto_cols = st.columns(3)
                if auto_cols[0].button(
                    "Применить профиль",
                    key=f"user_profile_auto_apply:{document_key}:{matched_profile_name}",
                ):
                    st.session_state[user_profile_auto_decision_key] = f"apply:{matched_profile_name}"
                    st.rerun()
                if auto_cols[1].button(
                    "Создать новый профиль",
                    key=f"user_profile_auto_create:{document_key}:{matched_profile_name}",
                ):
                    st.session_state[user_profile_auto_decision_key] = "create"
                    st.session_state[f"document_mode:{document_key}"] = "profile_setup"
                    st.session_state[f"profile_builder_step:{document_key}"] = "1. Источник данных"
                    st.rerun()
                if auto_cols[2].button(
                    "Использовать универсальное извлечение",
                    key=f"user_profile_auto_generic:{document_key}:{matched_profile_name}",
                ):
                    st.session_state[user_profile_auto_decision_key] = "generic"
                    st.rerun()

    if user_profile_candidates:
        applied_user_profile_config = user_profile_candidates[0]
        user_profile_apply_start_time = time.perf_counter()
        
        # Apply the profile (including running OCR if needed)
        user_profile_result = apply_user_profile(
            {"raw_rows": raw_rows, "ocr_candidates_df": ocr_candidates_df},
            applied_user_profile_config,
            ocr_runner=run_user_profile_ocr
        )
        user_profile_structured_df = user_profile_result["structured_rows"]
        
        # Update session state
        st.session_state[user_profile_structured_key] = user_profile_structured_df
        st.session_state[applied_user_profile_key] = applied_user_profile_config
        
        # Determine actual extraction strategy for display
        saved_profile_source = applied_user_profile_config.get("extraction", {}).get("source", "pdf_text_layer")
        saved_profile_ocr_engine = applied_user_profile_config.get("extraction", {}).get("ocr", {}).get("engine", "tesseract")
        
        display_strategy = "user-defined profile"
        if saved_profile_source == "ocr":
            display_strategy = f"OCR / {str(saved_profile_ocr_engine).title().replace('Ocr', 'OCR')}"
        elif saved_profile_source == "mixed":
            display_strategy = f"Mixed (PDF + OCR / {str(saved_profile_ocr_engine).title().replace('Ocr', 'OCR')})"

        if not user_profile_structured_df.empty or selected_is_user_profile:
            structured_rows = user_profile_structured_df
            active_profile = str(applied_user_profile_config.get("profile_name") or active_profile)
            is_generic_pdf = False
            
            profile_metadata = {
                **profile_metadata,
                "profile_name": active_profile,
                "profile_confidence": 1.0,
                "profile_reason": "Применён пользовательский source profile",
                "profile_selection": "manual" if selected_is_user_profile else "auto_user_profile",
                "selected_extraction_strategy": display_strategy,
            }
            
        with st.expander("Техническая отладка профиля"):
            # Gather profile diagnostics from attributes
            profile_diags = user_profile_structured_df.attrs.get("profile_diagnostics", {})
            table_diag = profile_diags.get("table_0", {})
            
            # Gather engine diagnostics
            ocr_result_methods = []
            if not ocr_result_df.empty and "extraction_method" in ocr_result_df.columns:
                ocr_result_methods = ocr_result_df["extraction_method"].unique().tolist()
            
            ocr_candidate_methods = []
            if not ocr_candidates_df.empty and "extraction_method" in ocr_candidates_df.columns:
                ocr_candidate_methods = ocr_candidates_df["extraction_method"].unique().tolist()

            available_ocr_block_uids = []
            if not ocr_candidates_df.empty and "ocr_block_id" in ocr_candidates_df.columns:
                available_ocr_block_uids = ocr_candidates_df["ocr_block_id"].dropna().unique().tolist()

            saved_blocks = applied_user_profile_config.get("blocks") or []
            saved_profile_selected_block_uids = []
            saved_profile_selected_row_uids = []
            for b in saved_blocks:
                uids = (b.get("selector") or {}).get("block_uids") or []
                saved_profile_selected_block_uids.extend(uids)
                row_filters = b.get("row_filters") or b.get("row_selection")
                if isinstance(row_filters, dict):
                    selected_rows = row_filters.get("selected_row_uids") or row_filters.get("selected_source_rows") or []
                    saved_profile_selected_row_uids.extend(selected_rows)
                elif isinstance(row_filters, list):
                    for rf in row_filters:
                        if isinstance(rf, dict) and rf.get("type") == "manual_selected_rows":
                            selected_rows = rf.get("selected_source_rows") or rf.get("source_rows") or []
                            saved_profile_selected_row_uids.extend(selected_rows)
                            
            # Compute intersection based on raw UID part
            target_ocr_uids = []
            for uid in saved_profile_selected_block_uids:
                if str(uid).startswith("ocr_candidate:"):
                    parts = str(uid).split(":", 2)
                    if len(parts) == 3:
                        target_ocr_uids.append(parts[2])
                else:
                    target_ocr_uids.append(str(uid))
                    
            matched_selected_block_uids = [uid for uid in available_ocr_block_uids if uid in target_ocr_uids]
            missing_selected_block_uids = [uid for uid in target_ocr_uids if uid not in available_ocr_block_uids]

            actual_ocr_engine_used = ocr_candidate_methods[0].replace("_ocr", "").replace("_candidate", "") if ocr_candidate_methods else None

            debug_info = {
                "metric": [
                    "selected_profile_name",
                    "saved_profile_source",
                    "saved_profile_ocr_engine",
                    "actual_ocr_engine_used",
                    "reconstruction_mode",
                    "ocr_result_rows",
                    "ocr_candidates_rows",
                    "source_block_rows_count",
                    "reconstructed_rows_count",
                    "rows_after_manual_selection",
                    "structured_rows_after_profile"
                ],
                "value": [
                    str(active_profile),
                    str(saved_profile_source),
                    str(saved_profile_ocr_engine),
                    str(actual_ocr_engine_used),
                    str(table_diag.get("reconstruction_mode", "none")),
                    str(len(ocr_result_df)),
                    str(len(ocr_candidates_df)),
                    str(table_diag.get("source_block_rows_count", 0)),
                    str(table_diag.get("reconstructed_rows_count", 0)),
                    str(table_diag.get("rows_after_manual_selection", 0)),
                    str(len(user_profile_structured_df))
                ]
            }
            
            st.dataframe(pd.DataFrame(debug_info), use_container_width=True)
            
            zero_rows_reason = ""
            if len(user_profile_structured_df) == 0:
                if not matched_selected_block_uids and saved_profile_source == "ocr":
                    zero_rows_reason = "Не найдено пересечений между сохранёнными block_uids профиля и сгенерированными OCR-кандидатами."
                elif table_diag.get("reconstructed_rows_count", 0) > 0 and table_diag.get("rows_after_manual_selection", 0) == 0:
                    zero_rows_reason = "Все реконструированные строки были отфильтрованы. Проверьте selected_row_uids в профиле."
                else:
                    # Check validation mappings against required fields
                    required_fields = applied_user_profile_config.get("validation", {}).get("required_fields", [])
                    produced_fields = []
                    for b in saved_blocks:
                        for col_map in b.get("column_mapping", {}).values():
                            role = col_map.get("role")
                            if role in {"name", "code", "custom_text"} and "name" not in produced_fields:
                                produced_fields.append("name")
                            if role in {"value", "value_direct", "value_intraport", "percent", "custom_numeric"} and "value" not in produced_fields:
                                produced_fields.append("value")
                        for t_map in b.get("token_mapping", {}).values():
                            if t_map.get("enabled"):
                                if "name" not in produced_fields: produced_fields.append("name")
                                if "value" not in produced_fields: produced_fields.append("value")
                                
                    missing_req_fields = [f for f in required_fields if f not in produced_fields]
                    if missing_req_fields:
                        zero_rows_reason = f"Validation removed rows because required fields are missing: {', '.join(missing_req_fields)}"
                    else:
                        zero_rows_reason = "Все строки были отфильтрованы row_filters или token_mapping."
                        
                st.warning(zero_rows_reason)
            
            st.json({
                "saved_profile_selected_block_uids": saved_profile_selected_block_uids,
                "saved_profile_selected_row_uids": saved_profile_selected_row_uids,
                "support_rows_included": table_diag.get("support_rows_included", []),
                "available_ocr_block_uids": available_ocr_block_uids,
                "matched_selected_block_uids": matched_selected_block_uids,
                "missing_selected_block_uids": missing_selected_block_uids,
                "ocr_ran": user_profile_result.get("ocr_ran", False),
                "status": user_profile_result.get("status"),
                "ocr_engine_diagnostics": {
                    "ocr_result_extraction_methods": ocr_result_methods,
                    "ocr_candidate_extraction_methods": ocr_candidate_methods,
                }
            })
            if not user_profile_structured_df.empty:
                st.write("Примеры итоговых строк:")
                st.dataframe(user_profile_structured_df.head(3), use_container_width=True)
            else:
                st.warning("Профиль не произвёл ни одной строки. Проверьте селекторы блоков и фильтры строк.")
                if applied_user_profile_config:
                    st.write("Конфигурация таблиц в профиле:")
                    st.json(applied_user_profile_config.get("blocks") or applied_user_profile_config.get("tables"))

        record_performance_timing(
            performance_timings,
            "user profile parser",
            user_profile_apply_start_time,
            cache_status="run",
            rows=len(user_profile_structured_df),
        )
    else:
        applied_user_profile_config = st.session_state.get(applied_user_profile_key)
        if applied_user_profile_config and user_profile_structured_df is not None and not user_profile_structured_df.empty:
            structured_rows = user_profile_structured_df
            active_profile = str(applied_user_profile_config.get("profile_name") or active_profile)
            is_generic_pdf = False
            profile_metadata = {
                **profile_metadata,
                "profile_name": active_profile,
                "profile_confidence": 1.0,
                "profile_reason": "Применён пользовательский source profile",
                "profile_selection": "manual_builder",
                "selected_extraction_strategy": "user-defined profile",
            }

    render_source_profile_card(
        active_profile,
        auto_mode=auto_profile,
        profile_metadata=profile_metadata,
        ocr_has_run=not ocr_result_df.empty,
        profile_config=applied_user_profile_config,
        document_key=document_key,
    )

    document_mode_key = f"document_mode:{document_key}"
    profile_is_known = bool(
        file_type != "pdf"
        or applied_user_profile_config
        or selected_is_user_profile
        or (active_profile and active_profile != "generic_pdf")
    )
    if document_mode_key not in st.session_state:
        st.session_state[document_mode_key] = "processing" if profile_is_known else "choose"
    document_mode = str(st.session_state.get(document_mode_key) or "processing")

    if document_mode == "choose" and not profile_is_known:
        st.warning("Система не знает этот тип документа. Что сделать?")
        unknown_cols = st.columns(3)
        if unknown_cols[0].button("Создать новый профиль", key=f"create_profile_mode:{document_key}"):
            st.session_state[document_mode_key] = "profile_setup"
            st.session_state[f"profile_builder_step:{document_key}"] = "1. Источник данных"
            st.rerun()
        if unknown_cols[1].button(
            "Использовать универсальное извлечение без профиля",
            key=f"use_generic_mode:{document_key}",
        ):
            st.session_state[document_mode_key] = "generic_processing"
            st.rerun()
        if unknown_cols[2].button(
            "Выбрать существующий профиль вручную",
            key=f"manual_profile_mode:{document_key}",
        ):
            st.info("Выберите профиль в поле `Профиль источника` выше.")
        return

    coverage_df = build_coverage_summary(structured_rows, profile=active_profile)
    found_blocks, total_blocks, missing_blocks = coverage_counts(coverage_df)

    st.subheader("Качество извлечения")
    show_legacy_generic_guidance = False
    if is_generic_pdf and show_legacy_generic_guidance:
        text_page_count = int(raw_summary.get("text_pages") or len(raw_rows[raw_rows["section_name"] == "raw_page_text"]))
        table_count = int(raw_summary.get("table_count") or 0)
        table_row_count = int(raw_summary.get("table_rows") or len(raw_table_rows))
        summary_cols = st.columns(5)
        summary_cols[0].metric("Время обработки", f"{processing_time:.2f}s")
        summary_cols[1].metric("Текстовых страниц", text_page_count)
        summary_cols[2].metric("Таблиц найдено", table_count)
        summary_cols[3].metric("Строк таблиц", table_row_count)
        summary_cols[4].metric("Структурированных показателей", len(structured_rows))
        quality_cols = st.columns(2)
        quality_cols[0].metric("Качество текстового слоя", "плохое" if bad_text_layer else "нормальное")
        quality_cols[1].metric("Рекомендуемое действие", "OCR" if bad_text_layer else "Настройка профиля")
        if bad_text_layer:
            st.warning(
                "Обнаружен некорректный текстовый слой PDF, рекомендуется OCR. "
                "Часть текста извлекается как технические токены (cid:...), поэтому raw preview может быть непригоден для разметки профиля."
            )
        else:
            st.info("Статус: требуется настройка профиля источника")
    else:
        status_counts = structured_rows["validation_status"].value_counts().to_dict()
        passed_count = int(status_counts.get("passed", 0))
        warning_count = int(status_counts.get("warning", 0))
        failed_count = int(status_counts.get("failed", 0))
        needs_review_count = warning_count + failed_count
        summary_cols = st.columns(5)
        summary_cols[0].metric("Время обработки", f"{processing_time:.2f}s")
        summary_cols[1].metric("Всего строк", len(structured_rows))
        summary_cols[2].metric("Успешно", passed_count)
        summary_cols[3].metric("Требуют проверки", needs_review_count)
        summary_cols[4].metric("Ошибки", failed_count)

    if total_blocks:
        st.caption(f"Покрытие блоков: {found_blocks} из {total_blocks} найдено")
        if missing_blocks:
            st.warning("Отсутствующие блоки: " + ", ".join(missing_blocks))

    st.subheader("Покрытие документа")
    if coverage_df.empty and active_profile == "generic_pdf":
        st.info(
            "Для универсального PDF профильные блоки не заданы. "
            "Система выполнила базовое извлечение текста и таблиц."
        )
    elif coverage_df.empty:
        st.info("Для этого типа документа пока нет настроенной диагностики покрытия.")
    else:
        st.dataframe(format_coverage_for_ui(coverage_df), use_container_width=True, hide_index=True)

    if is_generic_pdf:
        st.subheader("Найденные таблицы")
        if raw_table_summary_df.empty:
            st.info("Таблицы в PDF не найдены.")
        else:
            st.dataframe(
                format_table_summary_for_ui(raw_table_summary_df),
                use_container_width=True,
                hide_index=True,
            )

        if not ocr_tables_df.empty:
            st.subheader("Таблицы, требующие OCR")
            st.warning(
                "Текст в этих таблицах извлечён некорректно, требуется OCR перед созданием source profile."
            )
            st.dataframe(
                format_table_summary_for_ui(ocr_tables_df),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Предложение следующего действия")
        if bad_text_layer:
            st.warning(
                "Система обнаружила потенциально полезные табличные области, но текстовый слой PDF повреждён. "
                "Для структурированного извлечения сначала рекомендуется OCR, затем настройка нового профиля источника."
            )
        elif profile_candidates_df.empty:
            st.warning(
                "Система не нашла устойчивых табличных блоков. Для этого документа может потребоваться "
                "OCR/LLM-assisted extraction или ручная разметка."
            )
        else:
            st.info(
                "Система нашла потенциально полезные таблицы. Чтобы автоматически извлекать показатели "
                "из этого источника в будущем, создайте новый профиль источника на основе выбранных таблиц."
            )

        st.subheader("Что делать дальше")
        st.info(
            "Документ обработан универсальным extractor'ом. Система извлекла текст и таблицы, "
            "но не применяла профильный parser. Чтобы получать структурированные показатели автоматически, "
            "нужно создать профиль источника: определить нужные блоки, целевые поля и правила валидации."
        )
        if bad_text_layer:
            st.subheader("Доступные варианты обработки")
            st.info(
                "1. Запустить OCR-извлечение для выбранных страниц\n"
                "2. Скачать raw extraction для ручной разметки профиля\n"
                "3. Выбрать другой профиль вручную"
            )
            st.subheader("Запуск OCR")
            tesseract_cmd = get_tesseract_cmd()
            if tesseract_cmd:
                st.success(f"Tesseract OCR найден: {tesseract_cmd}")
            else:
                st.warning(TESSERACT_INSTALL_MESSAGE)

            available_ocr_pages = sorted(
                int(page)
                for page in raw_rows["page"].dropna().unique().tolist()
                if pd.notna(page)
            )
            default_ocr_pages = sorted(
                int(page)
                for page in ocr_tables_df["page"].dropna().unique().tolist()
                if pd.notna(page)
            )
            if not default_ocr_pages:
                default_ocr_pages = available_ocr_pages[:3]

            selected_ocr_pages = st.multiselect(
                "Выберите страницы для OCR",
                available_ocr_pages,
                default=default_ocr_pages,
                key=f"ocr_pages:{document_key}",
            )
            ocr_lang = st.selectbox(
                "Язык OCR",
                ["rus+eng", "rus", "eng"],
                key=f"ocr_lang:{document_key}",
            )
            available_languages = get_available_tesseract_languages()
            with st.expander("Диагностика OCR"):
                st.write(
                    {
                        "tesseract_cmd": tesseract_cmd,
                        "is_tesseract_available": is_tesseract_available(),
                        "available_languages": available_languages,
                        "selected_lang": ocr_lang,
                        "selected_pages": selected_ocr_pages,
                        "file_path": str(saved_path),
                    }
                )

            if st.button(
                "Запустить OCR для выбранных страниц",
                disabled=not selected_ocr_pages,
                key=f"run_ocr:{document_key}",
            ):
                ocr_start_time = time.perf_counter()
                try:
                    with st.spinner("OCR-извлечение выбранных страниц..."):
                        extracted_ocr_df = extract_ocr_pages(str(saved_path), selected_ocr_pages, lang=ocr_lang)
                        ocr_result_df = validate_extracted_data(extracted_ocr_df)
                        
                        # Check for empty text in OCR results
                        empty_pages = []
                        if not ocr_result_df.empty:
                            for _, row in ocr_result_df.iterrows():
                                if not str(row.get("evidence_text") or "").strip():
                                    empty_pages.append(int(row.get("page")))
                        
                        if empty_pages:
                            st.warning(f"OCR вернул пустой текст для страниц: {sorted(empty_pages)}")
                        
                        ocr_candidates_df = extract_ocr_table_candidates(ocr_result_df)
                        st.session_state[ocr_result_key] = ocr_result_df
                        st.session_state[ocr_candidates_key] = ocr_candidates_df
                        st.session_state["ocr_document_key"] = document_key
                        st.session_state["ocr_result_df"] = ocr_result_df
                        st.session_state["ocr_candidates_df"] = ocr_candidates_df
                        record_performance_timing(
                            performance_timings,
                            "OCR pages extraction",
                            ocr_start_time,
                            cache_status="run",
                            rows=len(ocr_result_df),
                        )
                    st.success(f"OCR завершён за {time.perf_counter() - ocr_start_time:.2f}s")
                except OCRUnavailableError as exc:
                    st.error(str(exc))
                except OCRLanguageError as exc:
                    st.error(str(exc))
                except OCRPageRenderError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)

    st.subheader("Извлечённые данные")
    show_profile_builder = bool(
        file_type == "pdf"
        and (
            document_mode == "profile_setup"
            or selected_is_user_profile
            or active_profile == "generic_pdf"
            or not raw_table_rows.empty
            or not ocr_candidates_df.empty
        )
    )
    tab_specs = []
    if document_mode == "profile_setup":
        if show_profile_builder:
            tab_specs.append(("profile_builder", "Конструктор профиля источника"))
    else:
        tab_specs.append(("structured", "Структурированные данные"))
        tab_specs.append(("review", "Строки, требующие проверки"))

    tabs = st.tabs([label for _, label in tab_specs])
    reviewed_df = structured_rows.copy()
    for (tab_key, _), tab in zip(tab_specs, tabs):
        with tab:
            if tab_key == "structured":
                if structured_rows.empty:
                    if prototype_structured_df.empty:
                        st.info(
                            "Структурированные данные отсутствуют. "
                            "Используйте raw export для настройки нового профиля источника."
                        )
                    else:
                        st.warning(
                            "Это prototype structured data, построенные из OCR и черновика профиля. "
                            "Требуется проверка перед production parser."
                        )
                        st.dataframe(
                            rename_columns_for_ui(translate_status_columns(prototype_structured_df)),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    edited_df = st.data_editor(
                        rename_columns_for_ui(translate_status_columns(structured_rows)),
                        use_container_width=True,
                        hide_index=True,
                        num_rows="dynamic",
                        key="structured_data_editor",
                    )
                    reviewed_df = validate_extracted_data(restore_status_columns(edited_df))
            elif tab_key == "raw":
                if raw_rows.empty:
                    st.info("Сырые фрагменты отсутствуют.")
                else:
                    st.dataframe(
                        rename_columns_for_ui(
                            translate_status_columns(select_existing_columns(raw_rows, RAW_DISPLAY_COLUMNS))
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            elif tab_key == "tables":
                st.dataframe(
                    rename_columns_for_ui(
                        translate_status_columns(select_existing_columns(raw_table_rows, RAW_DISPLAY_COLUMNS))
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            elif tab_key == "profile_candidates":
                if profile_candidates_df.empty:
                    st.info("Таблиц-кандидатов со score >= 0.5 не найдено.")
                else:
                    st.caption("Кандидаты с читаемым текстовым слоем")
                    st.dataframe(
                        format_table_summary_for_ui(profile_candidates_df),
                        use_container_width=True,
                        hide_index=True,
                    )
                if not ocr_tables_df.empty:
                    st.caption("Таблицы, требующие OCR")
                    st.dataframe(
                        format_table_summary_for_ui(ocr_tables_df),
                        use_container_width=True,
                        hide_index=True,
                    )
            elif tab_key == "ocr_result":
                if ocr_result_df.empty:
                    st.info("OCR пока не запускался. Выберите страницы и нажмите кнопку запуска OCR.")
                else:
                    st.dataframe(
                        rename_columns_for_ui(
                            translate_status_columns(select_existing_columns(ocr_result_df, OCR_DISPLAY_COLUMNS))
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            elif tab_key == "ocr_candidates":
                tab_ocr_document_matches = st.session_state.get("ocr_document_key") == document_key
                tab_ocr_result_df = st.session_state.get(ocr_result_key)
                if tab_ocr_result_df is None and tab_ocr_document_matches:
                    tab_ocr_result_df = st.session_state.get("ocr_result_df")
                tab_ocr_candidates_df = st.session_state.get(ocr_candidates_key)
                if tab_ocr_candidates_df is None and tab_ocr_document_matches:
                    tab_ocr_candidates_df = st.session_state.get("ocr_candidates_df")
                if tab_ocr_candidates_df is None and tab_ocr_result_df is not None and not tab_ocr_result_df.empty:
                    tab_ocr_candidates_df = extract_ocr_table_candidates(tab_ocr_result_df)
                    st.session_state[ocr_candidates_key] = tab_ocr_candidates_df
                    st.session_state["ocr_candidates_df"] = tab_ocr_candidates_df
                    st.session_state["ocr_document_key"] = document_key

                with st.expander("Диагностика OCR-кандидатов"):
                    st.write(ocr_candidate_diagnostics(tab_ocr_result_df, tab_ocr_candidates_df))

                if tab_ocr_candidates_df is None:
                    st.info(
                        "OCR пока не запускался. После OCR система попробует найти табличные блоки в распознанном тексте."
                    )
                elif tab_ocr_candidates_df.empty:
                    st.info("OCR выполнен, но кандидаты не найдены.")
                else:
                    tab_ocr_candidates_df = prepare_ocr_candidates_for_ui(tab_ocr_candidates_df)
                    best_table_candidates_df = get_best_ocr_table_candidates(tab_ocr_candidates_df)
                    tab_strong_ocr_candidate_count = len(best_table_candidates_df)
                    st.info(
                        "После OCR система нашла потенциальные табличные блоки. "
                        "Следующий шаг — выбрать нужные блоки и создать профиль источника: "
                        "целевые поля, правила парсинга и валидации."
                    )
                    st.caption(
                        f"Всего OCR-кандидатов: {len(tab_ocr_candidates_df)}. "
                        f"Сильных кандидатов: {tab_strong_ocr_candidate_count}."
                    )
                    if tab_strong_ocr_candidate_count:
                        st.success(f"Найдено {tab_strong_ocr_candidate_count} сильных кандидатов для нового профиля.")
                    st.caption("Это не финальные показатели, а кандидаты для настройки нового профиля источника.")

                    st.subheader("Лучшие табличные кандидаты для нового профиля")
                    if best_table_candidates_df.empty:
                        st.info("Табличных OCR-кандидатов с table_score >= 0.65 не найдено.")
                    else:
                        st.dataframe(
                            format_ocr_candidates_for_ui(best_table_candidates_df),
                            use_container_width=True,
                            hide_index=True,
                        )

                    selected_candidate_filter_label = st.radio(
                        "Фильтр",
                        list(OCR_CANDIDATE_FILTERS.keys()),
                        horizontal=True,
                        key=f"ocr_candidate_filter:{document_key}",
                    )
                    candidate_filter = OCR_CANDIDATE_FILTERS[selected_candidate_filter_label]
                    filtered_ocr_candidates_df = filter_ocr_candidates_by_type(tab_ocr_candidates_df, candidate_filter)
                    if filtered_ocr_candidates_df.empty:
                        st.info("По выбранному фильтру OCR-кандидатов нет.")
                    else:
                        st.dataframe(
                            format_ocr_candidates_for_ui(filtered_ocr_candidates_df),
                            use_container_width=True,
                            hide_index=True,
                        )

                    candidate_options_df = tab_ocr_candidates_df.drop_duplicates("ocr_block_id").copy()
                    candidate_options = candidate_options_df["ocr_block_id"].astype(str).tolist()
                    default_candidate_ids = candidate_options_df.loc[
                        candidate_options_df.apply(candidate_is_good_profile_section, axis=1),
                        "ocr_block_id",
                    ].astype(str).tolist()
                    good_only_requested = bool(st.session_state.pop(profile_draft_good_only_key, False))
                    if good_only_requested:
                        st.session_state[selected_ocr_candidate_ids_key] = default_candidate_ids
                    elif selected_ocr_candidate_ids_key not in st.session_state:
                        st.session_state[selected_ocr_candidate_ids_key] = default_candidate_ids
                    else:
                        st.session_state[selected_ocr_candidate_ids_key] = [
                            candidate_id
                            for candidate_id in st.session_state[selected_ocr_candidate_ids_key]
                            if candidate_id in candidate_options
                        ]

                    candidate_label_by_id = {
                        str(candidate["ocr_block_id"]): (
                            f"Стр. {candidate.get('page', '-')}: "
                            f"{candidate.get('block_title', '')} "
                            f"({OCR_CANDIDATE_TYPE_LABELS_RU.get(str(candidate.get('candidate_type')), candidate.get('candidate_type'))})"
                        )
                        for _, candidate in candidate_options_df.iterrows()
                    }
                    selected_candidate_ids = st.multiselect(
                        "Выберите блоки для нового профиля источника",
                        candidate_options,
                        format_func=lambda candidate_id: candidate_label_by_id.get(candidate_id, candidate_id),
                        key=selected_ocr_candidate_ids_key,
                    )
                    selected_ocr_candidates_df = tab_ocr_candidates_df.loc[
                        tab_ocr_candidates_df["ocr_block_id"].astype(str).isin(selected_candidate_ids)
                    ].copy()
                    st.session_state[selected_ocr_candidates_key] = selected_ocr_candidates_df

                    if selected_ocr_candidates_df.empty:
                        st.info("Выберите один или несколько OCR-кандидатов, чтобы увидеть preview и экспортировать их.")
                    else:
                        st.caption("Preview выбранных блоков")
                        st.dataframe(
                            format_ocr_candidates_for_ui(selected_ocr_candidates_df),
                            use_container_width=True,
                            hide_index=True,
                        )
                        with st.expander("Текст выбранных блоков"):
                            for _, candidate in selected_ocr_candidates_df.iterrows():
                                st.markdown(
                                    f"**Стр. {candidate.get('page', '-')} — {candidate.get('block_title', '')}**"
                                )
                                st.text(candidate.get("preview") or candidate.get("block_text") or "")

                    st.subheader("Черновик профиля источника")
                    st.warning(
                        "Это не готовый parser, а черновик профиля источника. "
                        "Его нужно проверить и доработать: уточнить поля, правила парсинга и валидации."
                    )
                    draft_cols = st.columns(2)
                    draft_profile_name = draft_cols[0].text_input(
                        "profile_name",
                        value="agro_kazakhstan_review",
                        key=f"profile_draft_name:{document_key}",
                    )
                    draft_display_name = draft_cols[1].text_input(
                        "display_name",
                        value="Обзор ВЭД / Казахстан / АПК",
                        key=f"profile_draft_display_name:{document_key}",
                    )
                    if good_only_requested:
                        profile_draft = build_profile_draft(
                            source_file=str(uploaded_file.name),
                            selected_candidates_df=selected_ocr_candidates_df,
                            profile_name=draft_profile_name,
                        )
                        profile_draft["display_name"] = draft_display_name
                        st.session_state[profile_draft_key] = profile_draft

                    if st.button(
                        "Сформировать черновик профиля",
                        disabled=selected_ocr_candidates_df.empty,
                        key=f"build_profile_draft:{document_key}",
                    ):
                        profile_draft = build_profile_draft(
                            source_file=str(uploaded_file.name),
                            selected_candidates_df=selected_ocr_candidates_df,
                            profile_name=draft_profile_name,
                        )
                        profile_draft["display_name"] = draft_display_name
                        st.session_state[profile_draft_key] = profile_draft

                    profile_draft = st.session_state.get(profile_draft_key)
                    if profile_draft:
                        profile_draft_yaml = dump_profile_draft_yaml(profile_draft)
                        profile_draft_json = dump_profile_draft_json(profile_draft)
                        draft_summary = profile_draft.get("profile_draft_summary") or {}
                        st.subheader("Качество черновика профиля")
                        quality_cols = st.columns(4)
                        quality_cols[0].metric("Всего секций", int(draft_summary.get("total_sections") or 0))
                        quality_cols[1].metric("Хороших секций", int(draft_summary.get("good_sections") or 0))
                        quality_cols[2].metric(
                            "Требуют проверки",
                            int(draft_summary.get("needs_review_sections") or 0),
                        )
                        quality_cols[3].metric("Слабых секций", int(draft_summary.get("weak_sections") or 0))

                        if int(draft_summary.get("needs_review_sections") or 0) or int(
                            draft_summary.get("weak_sections") or 0
                        ):
                            st.warning(
                                "В черновике есть секции, требующие ручной проверки. "
                                "Они могут быть информативными, но не выглядят как устойчивые таблицы."
                            )
                        if int(draft_summary.get("weak_sections") or 0):
                            st.warning(
                                "Слабые секции лучше не включать в первый профиль parser'а без ручной проверки."
                            )

                        quality_table = profile_draft_quality_table(profile_draft)
                        if not quality_table.empty:
                            st.dataframe(quality_table, use_container_width=True, hide_index=True)

                        if st.button(
                            "Оставить только хорошие секции",
                            disabled=not default_candidate_ids,
                            key=f"profile_draft_keep_good_only:{document_key}",
                        ):
                            st.session_state[profile_draft_good_only_key] = True
                            st.rerun()

                        st.code(profile_draft_yaml, language="yaml")
                        draft_export_cols = st.columns(2)
                        draft_export_cols[0].download_button(
                            "Скачать черновик профиля YAML",
                            data=profile_draft_yaml.encode("utf-8"),
                            file_name=f"{draft_profile_name}_profile_draft.yaml",
                            mime="application/x-yaml",
                            key=f"profile_draft_yaml_download:{document_key}",
                        )
                        draft_export_cols[1].download_button(
                            "Скачать черновик профиля JSON",
                            data=profile_draft_json.encode("utf-8"),
                            file_name=f"{draft_profile_name}_profile_draft.json",
                            mime="application/json",
                            key=f"profile_draft_json_download:{document_key}",
                        )

                        st.subheader("Прототип структурированного извлечения")
                        st.caption(
                            "Prototype parser пытается превратить выбранные OCR-блоки в long-format таблицу. "
                            "Это демонстрационный результат, не production parser."
                        )
                        if st.button(
                            "Запустить prototype parser по черновику",
                            key=f"run_prototype_parser:{document_key}",
                        ):
                            prototype_start_time = time.perf_counter()
                            prototype_structured_df = parse_sections_from_draft(profile_draft)
                            st.session_state[prototype_structured_key] = prototype_structured_df
                            st.session_state[mapped_complex_key] = pd.DataFrame()
                            st.session_state[reviewed_mapped_complex_key] = pd.DataFrame()
                            st.session_state[mapped_review_unsaved_key] = False
                            st.session_state["mapped_complex_df"] = pd.DataFrame()
                            record_performance_timing(
                                performance_timings,
                                "prototype parser",
                                prototype_start_time,
                                cache_status="run",
                                rows=len(prototype_structured_df),
                            )

                        prototype_structured_df = st.session_state.get(prototype_structured_key, pd.DataFrame())
                        if prototype_structured_df is not None and not prototype_structured_df.empty:
                            simple_prototype_rows, complex_prototype_rows = split_prototype_rows(prototype_structured_df)
                            prototype_auto_approved = (
                                int(prototype_structured_df["review_status"].eq("auto_approved").sum())
                                if "review_status" in prototype_structured_df.columns
                                else 0
                            )
                            prototype_needs_review = (
                                int(prototype_structured_df["review_status"].eq("needs_review").sum())
                                if "review_status" in prototype_structured_df.columns
                                else 0
                            )
                            prototype_ocr_normalized = (
                                int(
                                    prototype_structured_df["normalization_method"]
                                    .fillna("")
                                    .astype(str)
                                    .eq("ocr_decimal_divide_by_10")
                                    .sum()
                                )
                                if "normalization_method" in prototype_structured_df.columns
                                else 0
                            )
                            prototype_cols = st.columns(5)
                            prototype_cols[0].metric("Простые структурированные строки", len(simple_prototype_rows))
                            prototype_cols[1].metric("Сложные строки для разметки", len(complex_prototype_rows))
                            prototype_cols[2].metric("Автоматически принято", prototype_auto_approved)
                            prototype_cols[3].metric("Требуют проверки", prototype_needs_review)
                            prototype_cols[4].metric("Нормализовано OCR-эвристикой", prototype_ocr_normalized)

                            simple_tab, complex_tab, mapping_tab, suspicious_tab = st.tabs(
                                [
                                    "Простые структурированные строки",
                                    "Сложные таблицы для разметки колонок",
                                    "Разметка сложных таблиц",
                                    "Подозрительные значения",
                                ]
                            )
                            with simple_tab:
                                selected_prototype_filter = st.radio(
                                    "Фильтр prototype rows",
                                    list(PROTOTYPE_ROW_FILTERS.keys()),
                                    horizontal=True,
                                    key=f"prototype_row_filter:{document_key}",
                                )
                                filtered_prototype_rows = filter_prototype_rows(
                                    simple_prototype_rows,
                                    PROTOTYPE_ROW_FILTERS[selected_prototype_filter],
                                )
                                if filtered_prototype_rows.empty:
                                    st.info("По выбранному фильтру prototype parser строк нет.")
                                else:
                                    st.dataframe(
                                        rename_columns_for_ui(translate_status_columns(filtered_prototype_rows)),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                            with complex_tab:
                                st.caption(
                                    "Эти строки извлечены из сложных таблиц. "
                                    "Система нашла товары и числовые токены, но не присвоила им смысловые колонки. "
                                    "Для production parser нужно вручную задать mapping колонок."
                                )
                                if complex_prototype_rows.empty:
                                    st.info("Сложные prototype rows для mapping не найдены.")
                                else:
                                    st.dataframe(
                                        rename_columns_for_ui(translate_status_columns(complex_prototype_rows)),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                            with mapping_tab:
                                if complex_prototype_rows.empty:
                                    st.info("Сложные секции для разметки не найдены.")
                                else:
                                    section_options_df = (
                                        complex_prototype_rows[["section_id", "section_title"]]
                                        .drop_duplicates()
                                        .reset_index(drop=True)
                                    )
                                    section_options = section_options_df["section_id"].astype(str).tolist()
                                    section_labels = {
                                        str(row["section_id"]): f"{row['section_title']} ({row['section_id']})"
                                        for _, row in section_options_df.iterrows()
                                    }
                                    selected_section_id = st.selectbox(
                                        "Выберите сложную секцию",
                                        section_options,
                                        format_func=lambda section_id: section_labels.get(str(section_id), str(section_id)),
                                        key=f"complex_mapping_section:{document_key}",
                                    )
                                    selected_complex_rows = complex_prototype_rows.loc[
                                        complex_prototype_rows["section_id"].astype(str).eq(str(selected_section_id))
                                    ].copy()
                                    mapping_preview = build_mapping_preview(selected_complex_rows)
                                    st.dataframe(
                                        select_existing_columns(
                                            mapping_preview,
                                            ["commodity", "raw_numeric_tokens", "parsed_numeric_tokens", "evidence_text"],
                                        ),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                    st.caption(
                                        "Система нашла строки и числовые токены, но не знает смысл колонок. "
                                        "Задайте соответствие токенов полям."
                                    )

                                    max_token_count = 0
                                    if not mapping_preview.empty and "raw_numeric_tokens" in mapping_preview.columns:
                                        max_token_count = int(mapping_preview["raw_numeric_tokens"].apply(len).max())

                                    section_evidence_text = (
                                        " ".join(mapping_preview["evidence_text"].fillna("").astype(str).head(3))
                                        if "evidence_text" in mapping_preview.columns
                                        else ""
                                    )
                                    selected_section_title = (
                                        str(selected_complex_rows["section_title"].iloc[0])
                                        if "section_title" in selected_complex_rows.columns and not selected_complex_rows.empty
                                        else ""
                                    )
                                    suggested_preset = suggest_mapping_preset(
                                        selected_section_title,
                                        section_evidence_text,
                                        max_token_count,
                                    )
                                    preset_names = list(MAPPING_PRESETS.keys())
                                    preset_index = preset_names.index(suggested_preset) if suggested_preset in preset_names else 0
                                    selected_preset = st.selectbox(
                                        "Preset mapping",
                                        preset_names,
                                        index=preset_index,
                                        key=f"complex_mapping_preset:{document_key}:{selected_section_id}",
                                    )
                                    preset_token_count = len(MAPPING_PRESETS[selected_preset])
                                    editor_token_count = max(max_token_count, preset_token_count)

                                    token_mapping: dict[str, dict[str, object]] = {}
                                    if editor_token_count <= 0:
                                        st.info("В выбранной секции нет числовых токенов для mapping.")
                                    else:
                                        preset_token_mapping = token_mapping_from_preset(selected_preset, editor_token_count)
                                        st.caption(
                                            "Для каждого token задайте, использовать ли его, и атрибуты будущей structured row."
                                        )
                                        for token_index in range(1, editor_token_count + 1):
                                            token_key = f"token_{token_index}"
                                            default_attributes = preset_token_mapping.get(
                                                token_key,
                                                {
                                                    "enabled": False,
                                                    "metric": "other",
                                                    "year": None,
                                                    "unit": None,
                                                    "currency": None,
                                                    "label": "",
                                                },
                                            )
                                            token_cols = st.columns([1.0, 1.4, 1.1, 1.2, 1.0, 1.6])
                                            enabled = token_cols[0].checkbox(
                                                f"{token_key}: использовать",
                                                value=bool(default_attributes.get("enabled")),
                                                key=(
                                                    f"complex_mapping_enabled:{document_key}:"
                                                    f"{selected_section_id}:{selected_preset}:{token_key}"
                                                ),
                                            )
                                            default_metric = str(default_attributes.get("metric") or "other")
                                            if default_metric not in METRIC_OPTIONS:
                                                default_metric = "other"
                                            metric = token_cols[1].selectbox(
                                                "metric",
                                                METRIC_OPTIONS,
                                                index=METRIC_OPTIONS.index(default_metric),
                                                key=(
                                                    f"complex_mapping_metric:{document_key}:"
                                                    f"{selected_section_id}:{selected_preset}:{token_key}"
                                                ),
                                            )
                                            default_year = default_attributes.get("year")
                                            if default_year not in YEAR_OPTIONS:
                                                default_year = None
                                            year = token_cols[2].selectbox(
                                                "year",
                                                YEAR_OPTIONS,
                                                index=YEAR_OPTIONS.index(default_year),
                                                format_func=lambda value: "None" if value is None else str(value),
                                                key=(
                                                    f"complex_mapping_year:{document_key}:"
                                                    f"{selected_section_id}:{selected_preset}:{token_key}"
                                                ),
                                            )
                                            default_unit = default_attributes.get("unit")
                                            if default_unit not in UNIT_OPTIONS:
                                                default_unit = None
                                            unit = token_cols[3].selectbox(
                                                "unit",
                                                UNIT_OPTIONS,
                                                index=UNIT_OPTIONS.index(default_unit),
                                                format_func=lambda value: "None" if value is None else str(value),
                                                key=(
                                                    f"complex_mapping_unit:{document_key}:"
                                                    f"{selected_section_id}:{selected_preset}:{token_key}"
                                                ),
                                            )
                                            default_currency = default_attributes.get("currency")
                                            if default_currency not in CURRENCY_OPTIONS:
                                                default_currency = None
                                            currency = token_cols[4].selectbox(
                                                "currency",
                                                CURRENCY_OPTIONS,
                                                index=CURRENCY_OPTIONS.index(default_currency),
                                                format_func=lambda value: "None" if value is None else str(value),
                                                key=(
                                                    f"complex_mapping_currency:{document_key}:"
                                                    f"{selected_section_id}:{selected_preset}:{token_key}"
                                                ),
                                            )
                                            label = token_cols[5].text_input(
                                                "label",
                                                value=str(default_attributes.get("label") or ""),
                                                key=(
                                                    f"complex_mapping_label:{document_key}:"
                                                    f"{selected_section_id}:{selected_preset}:{token_key}"
                                                ),
                                            )
                                            token_mapping[token_key] = {
                                                "enabled": enabled,
                                                "metric": metric,
                                                "year": year,
                                                "unit": unit,
                                                "currency": currency,
                                                "label": label,
                                            }

                                    mapping_verified = st.checkbox(
                                        "Считать mapping проверенным",
                                        value=False,
                                        key=f"complex_mapping_verified:{document_key}:{selected_section_id}",
                                    )
                                    mapping_config = build_mapping_config(
                                        section_id=selected_section_id,
                                        section_title=selected_section_title,
                                        token_mapping=token_mapping,
                                        mapping_verified=mapping_verified,
                                    )
                                    mapping_config_hash = stable_json_hash(mapping_config)
                                    applied_mapping_config_hash_key = (
                                        f"applied_mapping_config_hash:{document_key}:{selected_section_id}"
                                    )
                                    mapped_complex_config_key = (
                                        f"mapped_complex:{document_key}:{selected_section_id}:{mapping_config_hash}"
                                    )
                                    reviewed_mapped_complex_config_key = (
                                        f"reviewed_mapped_complex:{document_key}:{selected_section_id}:{mapping_config_hash}"
                                    )

                                    reconstruction_rows = []
                                    if not mapping_preview.empty:
                                        for _, preview_row in mapping_preview.iterrows():
                                            raw_tokens = preview_row.get("raw_numeric_tokens") or []
                                            reconstruction = reconstruct_numeric_tokens(
                                                evidence_text=str(preview_row.get("evidence_text") or ""),
                                                raw_tokens=[str(token) for token in raw_tokens],
                                                mapping_config=mapping_config,
                                            )
                                            reconstruction_rows.append(
                                                {
                                                    "commodity": preview_row.get("commodity"),
                                                    "expected_count": reconstruction["expected_count"],
                                                    "raw_tokens_count": len(raw_tokens),
                                                    "reconstructed_count": len(
                                                        reconstruction["reconstructed_values"]
                                                    ),
                                                    "reconstruction_status": reconstruction[
                                                        "reconstruction_status"
                                                    ],
                                                    "reconstruction_method": reconstruction[
                                                        "reconstruction_method"
                                                    ],
                                                    "reconstruction_warnings": "; ".join(
                                                        reconstruction["reconstruction_warnings"]
                                                    ),
                                                    "evidence_text": preview_row.get("evidence_text"),
                                                }
                                            )
                                    if reconstruction_rows:
                                        st.caption("Диагностика восстановления чисел")
                                        st.dataframe(
                                            pd.DataFrame(reconstruction_rows),
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                                    if st.button(
                                        "Применить mapping к выбранной секции",
                                        disabled=editor_token_count <= 0,
                                        key=f"apply_complex_mapping:{document_key}:{selected_section_id}",
                                    ):
                                        mapping_start_time = time.perf_counter()
                                        mapped_complex_df = apply_complex_mapping(selected_complex_rows, mapping_config)
                                        reviewed_mapped_after_mapping = prepare_reviewed_mapped_export(mapped_complex_df)
                                        st.session_state[mapped_complex_key] = mapped_complex_df
                                        st.session_state[mapped_complex_config_key] = mapped_complex_df
                                        st.session_state[reviewed_mapped_complex_key] = reviewed_mapped_after_mapping
                                        st.session_state[
                                            reviewed_mapped_complex_config_key
                                        ] = reviewed_mapped_after_mapping
                                        st.session_state[applied_mapping_config_hash_key] = mapping_config_hash
                                        st.session_state[mapped_review_unsaved_key] = False
                                        st.session_state["mapped_complex_df"] = mapped_complex_df
                                        st.session_state[f"mapping_config:{document_key}:{selected_section_id}"] = mapping_config
                                        record_performance_timing(
                                            performance_timings,
                                            "complex mapping",
                                            mapping_start_time,
                                            cache_status="run",
                                            rows=len(mapped_complex_df),
                                        )

                                    st.download_button(
                                        "Скачать mapping config JSON",
                                        data=json.dumps(mapping_config, ensure_ascii=False, indent=2).encode("utf-8"),
                                        file_name=f"{selected_section_id}_mapping_config.json",
                                        mime="application/json",
                                        key=f"download_complex_mapping_config:{document_key}:{selected_section_id}",
                                    )

                                    applied_mapping_config_hash = st.session_state.get(applied_mapping_config_hash_key)
                                    if applied_mapping_config_hash:
                                        applied_mapped_complex_key = (
                                            f"mapped_complex:{document_key}:"
                                            f"{selected_section_id}:{applied_mapping_config_hash}"
                                        )
                                        mapped_complex_df = st.session_state.get(
                                            applied_mapped_complex_key,
                                            st.session_state.get(mapped_complex_key, pd.DataFrame()),
                                        )
                                    else:
                                        mapped_complex_df = st.session_state.get(mapped_complex_key, pd.DataFrame())
                                    if mapped_complex_df is not None and not mapped_complex_df.empty:
                                        mapped_section_rows = mapped_complex_df.loc[
                                            mapped_complex_df["section_id"].astype(str).eq(str(selected_section_id))
                                        ].copy()
                                        if not mapped_section_rows.empty:
                                            mapped_status = mapped_section_rows["validation_status"].fillna("").astype(str)
                                            mapped_warnings = mapped_section_rows["warnings"].fillna("").astype(str)
                                            mapped_cols = st.columns(4)
                                            mapped_cols[0].metric("Строк после mapping", len(mapped_section_rows))
                                            mapped_cols[1].metric(
                                                "Успешно",
                                                int(mapped_status.eq("passed_with_warning").sum()),
                                            )
                                            mapped_cols[2].metric(
                                                "Требуют проверки",
                                                int(mapped_status.eq("needs_review").sum()),
                                            )
                                            mapped_cols[3].metric(
                                                "Missing tokens",
                                                int(mapped_warnings.str.contains("mapped token is missing", regex=False).sum()),
                                            )
                                            st.dataframe(
                                                rename_columns_for_ui(translate_status_columns(mapped_section_rows)),
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                            applied_reviewed_mapped_key = None
                                            if applied_mapping_config_hash:
                                                applied_reviewed_mapped_key = (
                                                    f"reviewed_mapped_complex:{document_key}:"
                                                    f"{selected_section_id}:{applied_mapping_config_hash}"
                                                )
                                            reviewed_mapped_df = (
                                                st.session_state.get(applied_reviewed_mapped_key)
                                                if applied_reviewed_mapped_key
                                                else None
                                            )
                                            if reviewed_mapped_df is None:
                                                reviewed_mapped_df = st.session_state.get(reviewed_mapped_complex_key)
                                            if reviewed_mapped_df is None or reviewed_mapped_df.empty:
                                                reviewed_mapped_df = prepare_reviewed_mapped_export(mapped_complex_df)
                                            else:
                                                reviewed_mapped_df = prepare_reviewed_mapped_export(reviewed_mapped_df)
                                            if applied_reviewed_mapped_key:
                                                st.session_state[applied_reviewed_mapped_key] = reviewed_mapped_df
                                            st.session_state[reviewed_mapped_complex_key] = reviewed_mapped_df

                                            st.subheader("Ручная проверка извлечённых строк")
                                            st.caption(
                                                "Это результат prototype parser + human review для mapped rows; "
                                                "он не смешивается с production clean export."
                                            )
                                            selected_review_filter_label = st.radio(
                                                "Фильтр ручной проверки",
                                                list(MAPPED_REVIEW_FILTERS.keys()),
                                                index=1,
                                                horizontal=True,
                                                key=f"mapped_review_filter_radio:{document_key}:{selected_section_id}",
                                            )
                                            review_filter = MAPPED_REVIEW_FILTERS[selected_review_filter_label]
                                            review_state_scope = (
                                                f"{document_key}:{selected_section_id}:"
                                                f"{applied_mapping_config_hash or 'legacy'}"
                                            )
                                            review_editor_state_key = f"mapped_review_editor_df:{review_state_scope}"
                                            review_editor_filter_state_key = f"mapped_review_editor_filter:{review_state_scope}"
                                            review_editor_widget_version_key = (
                                                f"mapped_review_editor_widget_version:{review_state_scope}"
                                            )
                                            if review_editor_widget_version_key not in st.session_state:
                                                st.session_state[review_editor_widget_version_key] = 0
                                            review_editor_widget_key = (
                                                f"mapped_review_editor_widget:{review_state_scope}:"
                                                f"{st.session_state[review_editor_widget_version_key]}"
                                            )
                                            review_history_key = f"mapped_review_previous:{review_state_scope}"

                                            review_prepare_start_time = time.perf_counter()
                                            base_review_editor_rows = prepare_review_editor_df(
                                                select_mapped_rows_for_review(reviewed_mapped_df, review_filter)
                                            )
                                            stored_review_editor_rows = st.session_state.get(review_editor_state_key)
                                            if stored_review_editor_rows is not None:
                                                stored_review_editor_rows = prepare_review_editor_df(
                                                    stored_review_editor_rows
                                                )
                                            stored_filter = st.session_state.get(review_editor_filter_state_key)
                                            filter_changed = stored_filter != review_filter
                                            has_existing_unsaved = review_editor_has_unsaved_changes(
                                                reviewed_mapped_df,
                                                stored_review_editor_rows,
                                            )
                                            if stored_review_editor_rows is None or (
                                                filter_changed and not has_existing_unsaved
                                            ):
                                                st.session_state[review_editor_state_key] = base_review_editor_rows
                                                st.session_state[review_editor_filter_state_key] = review_filter
                                                st.session_state[review_editor_widget_version_key] += 1
                                                review_editor_widget_key = (
                                                    f"mapped_review_editor_widget:{review_state_scope}:"
                                                    f"{st.session_state[review_editor_widget_version_key]}"
                                                )
                                            elif filter_changed and has_existing_unsaved:
                                                st.warning(
                                                    "Есть несохранённые правки. Примените или сбросьте их перед сменой фильтра."
                                                )

                                            review_editor_rows = prepare_review_editor_df(
                                                st.session_state.get(review_editor_state_key)
                                            )
                                            compact_review_editor_rows = prepare_compact_review_editor_df(
                                                review_editor_rows
                                            )
                                            record_performance_timing(
                                                performance_timings,
                                                "human review preparation",
                                                review_prepare_start_time,
                                                cache_status="session",
                                                rows=len(review_editor_rows),
                                            )
                                            if review_editor_rows.empty:
                                                st.success("Нет mapped rows для выбранного фильтра.")
                                            else:
                                                disabled_review_columns = [
                                                    column
                                                    for column in compact_review_editor_rows.columns
                                                    if column not in MAPPED_REVIEW_EDITABLE_COLUMNS
                                                ]
                                                with st.form(
                                                    key=f"mapped_review_form:{review_state_scope}:"
                                                    f"{st.session_state[review_editor_widget_version_key]}"
                                                ):
                                                    edited_review_rows = st.data_editor(
                                                        compact_review_editor_rows,
                                                        use_container_width=True,
                                                        hide_index=True,
                                                        num_rows="fixed",
                                                        disabled=disabled_review_columns,
                                                        column_config={
                                                            REVIEW_ROW_UID_COLUMN: None,
                                                            REVIEW_ROW_INDEX_COLUMN: None,
                                                            "section_title": st.column_config.TextColumn(
                                                                "section_title",
                                                                disabled=True,
                                                            ),
                                                            "commodity": st.column_config.TextColumn("commodity"),
                                                            "metric": st.column_config.SelectboxColumn(
                                                                "metric",
                                                                options=METRIC_OPTIONS,
                                                            ),
                                                            "year": st.column_config.NumberColumn(
                                                                "Год",
                                                                min_value=1900,
                                                                max_value=2100,
                                                                step=1,
                                                                format="%d",
                                                            ),
                                                            "value": st.column_config.NumberColumn("value"),
                                                            "unit": st.column_config.SelectboxColumn(
                                                                "unit",
                                                                options=UNIT_OPTIONS,
                                                            ),
                                                            "currency": st.column_config.SelectboxColumn(
                                                                "currency",
                                                                options=CURRENCY_OPTIONS,
                                                            ),
                                                            "approved_by_user": st.column_config.CheckboxColumn(
                                                                "approved_by_user"
                                                            ),
                                                            "review_comment": st.column_config.TextColumn(
                                                                "review_comment"
                                                            ),
                                                        },
                                                        key=review_editor_widget_key,
                                                    )
                                                    review_submitted = st.form_submit_button(
                                                        "Применить ручную проверку",
                                                        disabled=compact_review_editor_rows.empty,
                                                    )
                                                if review_submitted:
                                                    st.session_state[review_history_key] = reviewed_mapped_df.copy()
                                                    submitted_review_rows = prepare_review_editor_df(
                                                        edited_review_rows
                                                    )
                                                    reviewed_mapped_df = apply_mapped_review_edits(
                                                        reviewed_mapped_df,
                                                        submitted_review_rows,
                                                    )
                                                    st.session_state[reviewed_mapped_complex_key] = reviewed_mapped_df
                                                    if applied_reviewed_mapped_key:
                                                        st.session_state[
                                                            applied_reviewed_mapped_key
                                                        ] = reviewed_mapped_df
                                                    st.session_state[mapped_review_unsaved_key] = False
                                                    st.session_state[review_editor_state_key] = prepare_review_editor_df(
                                                        select_mapped_rows_for_review(reviewed_mapped_df, review_filter)
                                                    )
                                                    st.session_state[review_editor_filter_state_key] = review_filter
                                                    st.session_state[review_editor_widget_version_key] += 1
                                                    st.rerun()

                                                evidence_options = review_editor_rows[
                                                    REVIEW_ROW_UID_COLUMN
                                                ].fillna("").astype(str).tolist()
                                                if evidence_options:
                                                    selected_evidence_uid = st.selectbox(
                                                        "Выберите строку для просмотра evidence",
                                                        evidence_options,
                                                        format_func=lambda row_uid: mapped_review_evidence_label(
                                                            review_editor_rows.loc[
                                                                review_editor_rows[REVIEW_ROW_UID_COLUMN]
                                                                .fillna("")
                                                                .astype(str)
                                                                .eq(str(row_uid))
                                                            ].iloc[0]
                                                        ),
                                                        key=f"mapped_review_evidence:{review_state_scope}",
                                                    )
                                                    selected_evidence_rows = review_editor_rows.loc[
                                                        review_editor_rows[REVIEW_ROW_UID_COLUMN]
                                                        .fillna("")
                                                        .astype(str)
                                                        .eq(str(selected_evidence_uid))
                                                    ]
                                                    if not selected_evidence_rows.empty:
                                                        evidence_row = selected_evidence_rows.iloc[0]
                                                        with st.expander("Evidence / warnings выбранной строки"):
                                                            st.dataframe(
                                                                select_existing_columns(
                                                                    pd.DataFrame([evidence_row]),
                                                                    MAPPED_REVIEW_EVIDENCE_COLUMNS,
                                                                ),
                                                                use_container_width=True,
                                                                hide_index=True,
                                                            )
                                                            st.text(str(evidence_row.get("evidence_text") or ""))

                                            current_review_editor_df = prepare_review_editor_df(
                                                st.session_state.get(review_editor_state_key)
                                            )
                                            editor_summary = mapped_review_editor_summary(
                                                reviewed_mapped_df,
                                                current_review_editor_df,
                                            )
                                            editor_metric_cols = st.columns(4)
                                            editor_metric_cols[0].metric(
                                                "Строк в редакторе",
                                                editor_summary["rows_in_editor"],
                                            )
                                            editor_metric_cols[1].metric(
                                                "Изменено строк",
                                                editor_summary["changed_values"],
                                            )
                                            editor_metric_cols[2].metric(
                                                "Отмечено к подтверждению",
                                                editor_summary["marked_for_approval"],
                                            )
                                            editor_metric_cols[3].metric(
                                                "Есть комментарии",
                                                editor_summary["comments"],
                                            )
                                            has_unsaved_changes = review_editor_has_unsaved_changes(
                                                reviewed_mapped_df,
                                                current_review_editor_df,
                                            )
                                            st.session_state[mapped_review_unsaved_key] = has_unsaved_changes
                                            if has_unsaved_changes:
                                                st.warning(
                                                    "Есть несохранённые правки. Нажмите 'Применить ручную проверку' перед экспортом."
                                                )

                                            action_cols = st.columns(2)
                                            if action_cols[0].button(
                                                "Сбросить несохранённые правки",
                                                disabled=not has_unsaved_changes,
                                                key=f"reset_mapped_review:{document_key}:{selected_section_id}",
                                            ):
                                                st.session_state[review_editor_state_key] = base_review_editor_rows
                                                st.session_state[review_editor_filter_state_key] = review_filter
                                                st.session_state[mapped_review_unsaved_key] = False
                                                st.session_state[review_editor_widget_version_key] += 1
                                                st.rerun()
                                            if action_cols[1].button(
                                                "Вернуть original values",
                                                disabled=reviewed_mapped_df.empty,
                                                key=f"restore_mapped_review:{document_key}:{selected_section_id}",
                                            ):
                                                st.session_state[review_history_key] = reviewed_mapped_df.copy()
                                                reviewed_mapped_df = restore_mapped_review_original_values(
                                                    reviewed_mapped_df
                                                )
                                                st.session_state[reviewed_mapped_complex_key] = reviewed_mapped_df
                                                if applied_reviewed_mapped_key:
                                                    st.session_state[
                                                        applied_reviewed_mapped_key
                                                    ] = reviewed_mapped_df
                                                st.session_state[mapped_review_unsaved_key] = False
                                                st.session_state[review_editor_state_key] = prepare_review_editor_df(
                                                    select_mapped_rows_for_review(reviewed_mapped_df, review_filter)
                                                )
                                                st.session_state[review_editor_filter_state_key] = review_filter
                                                st.session_state[review_editor_widget_version_key] += 1
                                                st.rerun()

                                            review_summary = mapped_review_summary(reviewed_mapped_df)
                                            review_metric_cols = st.columns(5)
                                            review_metric_cols[0].metric(
                                                "Всего mapped rows",
                                                review_summary["total_rows"],
                                            )
                                            review_metric_cols[1].metric(
                                                "Требовали проверки",
                                                review_summary["required_review"],
                                            )
                                            review_metric_cols[2].metric(
                                                "Исправлено пользователем",
                                                review_summary["edited_by_user"],
                                            )
                                            review_metric_cols[3].metric(
                                                "Подтверждено пользователем",
                                                review_summary["approved_by_user"],
                                            )
                                            review_metric_cols[4].metric(
                                                "Остались непроверенными",
                                                review_summary["remaining_unreviewed"],
                                            )
                            with suspicious_tab:
                                suspicious_rows = select_prototype_suspicious_rows(simple_prototype_rows)
                                if suspicious_rows.empty:
                                    st.info("Подозрительных значений в простых prototype rows нет.")
                                else:
                                    st.dataframe(suspicious_rows, use_container_width=True, hide_index=True)
            elif tab_key == "profile_builder":
                st.subheader("Конструктор профиля источника")
                st.caption(
                    "Конструктор нужен, если система не знает этот тип документа. Вы один раз размечаете таблицу, "
                    "сохраняете профиль, и затем похожие документы обрабатываются автоматически."
                )
                all_table_catalog_df = build_profile_table_catalog(raw_table_rows, raw_table_summary_df, ocr_candidates_df)
                stored_profile_builder_ocr_catalog = st.session_state.get(profile_builder_ocr_blocks_catalog_key)
                if (
                    ocr_candidates_df.empty
                    and isinstance(stored_profile_builder_ocr_catalog, pd.DataFrame)
                    and not stored_profile_builder_ocr_catalog.empty
                ):
                    non_ocr_catalog_df = all_table_catalog_df.loc[
                        ~all_table_catalog_df["source_kind"].fillna("").astype(str).eq("ocr_candidate")
                    ].copy()
                    all_table_catalog_df = pd.concat(
                        [non_ocr_catalog_df, stored_profile_builder_ocr_catalog.copy()],
                        ignore_index=True,
                    )
                source_state_key = profile_builder_source_state_key(document_key)
                source_widget_key = profile_builder_source_widget_key(document_key)
                default_builder_source = profile_builder_default_source(
                    bad_text_layer=bad_text_layer,
                    has_pdf_tables=not raw_table_rows.empty,
                    has_ocr_candidates=not ocr_candidates_df.empty,
                )
                if source_state_key not in st.session_state:
                    st.session_state[source_state_key] = default_builder_source
                builder_source = profile_builder_get_source(st.session_state, document_key, default_builder_source)
                stored_profile_builder_ocr_catalog_df = (
                    stored_profile_builder_ocr_catalog
                    if isinstance(stored_profile_builder_ocr_catalog, pd.DataFrame)
                    else pd.DataFrame()
                )
                table_catalog_df = profile_builder_catalog_for_source(
                    all_table_catalog_df,
                    builder_source,
                    stored_profile_builder_ocr_catalog_df,
                )

                wizard_step = st.radio(
                    "Шаг конструктора",
                    PROFILE_BUILDER_STEPS,
                    horizontal=True,
                    key=f"profile_builder_step:{document_key}",
                )
                if wizard_step == "2. Таблицы/блоки":
                    selected_tables_debug_key = f"profile_builder_block_selection_applied:{document_key}"
                    draft_tables_debug_key = f"profile_builder_block_selection_draft:{document_key}"
                    last_submit_debug_key = f"profile_builder_last_block_selection_debug:{document_key}"
                    stored_profile_builder_ocr_candidates = st.session_state.get(profile_builder_ocr_candidates_key)
                    stored_profile_builder_ocr_result = st.session_state.get(profile_builder_ocr_result_key)
                    debug_payload = {
                        "wizard_step": wizard_step,
                        "builder_source": builder_source,
                        "default_builder_source": default_builder_source,
                        "selected_ocr_engine_from_ui": current_builder_engine,
                        "durable_ocr_engine_state": st.session_state.get(ocr_engine_state_key),
                        "source_state_key": source_state_key,
                        "source_state_value": st.session_state.get(source_state_key),
                        "source_widget_key": source_widget_key,
                        "source_widget_value": st.session_state.get(source_widget_key),
                        "ocr_result_key": ocr_result_key,
                        "ocr_result_rows": len(ocr_result_df),
                        "ocr_candidates_key": ocr_candidates_key,
                        "ocr_candidates_rows": len(ocr_candidates_df),
                        "builder_ocr_result_key": profile_builder_ocr_result_key,
                        "builder_ocr_result_rows": len(stored_profile_builder_ocr_result)
                        if isinstance(stored_profile_builder_ocr_result, pd.DataFrame)
                        else 0,
                        "builder_ocr_candidates_key": profile_builder_ocr_candidates_key,
                        "builder_ocr_candidates_rows": len(stored_profile_builder_ocr_candidates)
                        if isinstance(stored_profile_builder_ocr_candidates, pd.DataFrame)
                        else 0,
                        "builder_ocr_blocks_catalog_key": profile_builder_ocr_blocks_catalog_key,
                        "builder_ocr_blocks_catalog_rows": len(stored_profile_builder_ocr_catalog_df),
                        "all_table_catalog_rows": len(all_table_catalog_df),
                        "source_table_catalog_rows": len(table_catalog_df),
                        "available_block_uids": profile_builder_table_options(table_catalog_df),
                        "selected_block_uids_before_submit": st.session_state.get(selected_tables_debug_key, []),
                        "draft_block_uids": st.session_state.get(draft_tables_debug_key, []),
                        "last_submit": st.session_state.get(last_submit_debug_key, {}),
                    }
                    
                    if (builder_source == "ocr" or builder_source == "mixed") and not ocr_result_df.empty:
                        ocr_pages = sorted(ocr_result_df["page"].unique().tolist())
                        candidate_pages = sorted(ocr_candidates_df["page"].unique().tolist()) if not ocr_candidates_df.empty else []
                        catalog_pages = sorted(table_catalog_df["page"].unique().tolist()) if not table_catalog_df.empty else []
                        missing_pages = [p for p in ocr_pages if p not in candidate_pages]

                        # Enhanced per-page debug info
                        raw_page_stats = []
                        for page_num in ocr_pages:
                            page_rows = ocr_result_df[ocr_result_df["page"] == page_num]
                            if not page_rows.empty:
                                first_row = page_rows.iloc[0]
                                evidence_text = str(first_row.get("evidence_text") or "")
                                raw_page_stats.append({
                                    "page": int(page_num),
                                    "text_len": len(evidence_text),
                                    "lines_count": len(evidence_text.splitlines()),
                                    "numbers_count": len(re.findall(r"\d+", evidence_text)),
                                    "extraction_method": first_row.get("extraction_method"),
                                    "ocr_engine": first_row.get("ocr_engine"),
                                    "has_candidate": page_num in candidate_pages
                                })

                        debug_payload["ocr_multipage_stats"] = {
                            "ocr_result_pages": ocr_pages,
                            "candidate_pages": candidate_pages,
                            "catalog_pages": catalog_pages,
                            "missing_candidate_pages": missing_pages,
                            "total_ocr_pages": len(ocr_pages),
                            "total_candidate_pages": len(candidate_pages),
                            "total_catalog_pages": len(catalog_pages),
                            "raw_page_stats": raw_page_stats,
                        }

                        if missing_pages:
                            debug_payload["candidate_skip_reasons"] = {}
                            from src.ocr_table_candidates import _has_fallback_criteria
                            for mp in missing_pages:
                                page_text = str(ocr_result_df[ocr_result_df["page"] == mp]["evidence_text"].iloc[0] or "")
                                if not page_text.strip():
                                    reason = "Пустой текст OCR"
                                elif len(page_text.splitlines()) < 5:
                                    reason = f"Слишком мало строк ({len(page_text.splitlines())} < 5)"
                                elif not _has_fallback_criteria(page_text):
                                    reason = "Не соответствует критериям fallback (мало чисел или ключевых слов)"
                                else:
                                    reason = "Неизвестная причина (критерии fallback пройдены, но кандидат не создан)"
                                debug_payload["candidate_skip_reasons"][f"стр. {mp}"] = reason
                    
                    with st.expander("Debug OCR wizard state", expanded=True):
                        st.json(debug_payload)

                if wizard_step == "1. Источник данных":
                    st.write(
                        "Сначала выберите, откуда брать данные. Если PDF содержит плохой текстовый слой, "
                        "лучше использовать OCR. Это сохранится в профиле и будет автоматически применяться "
                        "к похожим документам."
                    )
                    st.caption(f"Рекомендация: {profile_builder_source_label(default_builder_source)}")
                    selected_source = st.radio(
                        "Источник данных",
                        ["pdf_text_layer", "ocr", "mixed"],
                        index=["pdf_text_layer", "ocr", "mixed"].index(builder_source)
                        if builder_source in {"pdf_text_layer", "ocr", "mixed"}
                        else 0,
                        format_func=profile_builder_source_label,
                        horizontal=True,
                        key=source_widget_key,
                    )
                    if selected_source != builder_source:
                        st.session_state[source_state_key] = selected_source
                        st.session_state[f"profile_builder_tables:{document_key}"] = []
                        st.session_state[f"profile_builder_block_selection_applied:{document_key}"] = []
                        st.session_state[f"profile_builder_block_selection_draft:{document_key}"] = []
                        st.session_state.pop(f"profile_builder_table_reconstruction_applied:{document_key}", None)
                        st.session_state.pop(f"profile_builder_split_pattern_draft:{document_key}", None)
                        st.session_state.pop(f"profile_builder_manual_rows:{document_key}", None)
                        st.rerun()
                    if selected_source in {"ocr", "mixed"}:
                        available_engines_dict = get_available_engines()
                        all_engine_names = ["tesseract", "paddleocr", "yandex_vision"]
                        
                        ocr_cols = st.columns(3)
                        
                        engine_state_key = f"profile_builder_ocr_engine:{document_key}"
                        
                        if engine_state_key not in st.session_state:
                            st.session_state[engine_state_key] = "tesseract"
                            
                        current_engine = st.session_state.get(engine_state_key, "tesseract")
                        if current_engine not in all_engine_names:
                            current_engine = "tesseract"
                        
                        engine_widget_key = f"profile_builder_ocr_engine_widget:{document_key}"
                        selected_engine = ocr_cols[0].selectbox(
                            "OCR engine",
                            options=all_engine_names,
                            format_func=lambda name: available_engines_dict.get(name, (name, (False, "")))[0],
                            index=all_engine_names.index(current_engine) if current_engine in all_engine_names else 0,
                            key=engine_widget_key,
                        )
                        
                        if selected_engine != current_engine:
                            st.session_state[engine_state_key] = selected_engine
                            st.rerun()
                        
                        # Show availability message if the selected engine is not available
                        is_available, availability_msg = available_engines_dict.get(selected_engine, (selected_engine, (False, "Unknown engine")))[1]
                        if not is_available:
                            st.warning(availability_msg)
                        
                        ocr_cols[1].text_input(
                            "Язык OCR",
                            value=st.session_state.get(f"profile_builder_ocr_lang:{document_key}", "rus+eng"),
                            key=f"profile_builder_ocr_lang:{document_key}",
                        )
                        ocr_cols[2].text_input(
                            "Страницы для OCR",
                            value=st.session_state.get(f"profile_builder_ocr_pages:{document_key}", "auto"),
                            help="auto или список страниц через запятую, например 1,2,3",
                            key=f"profile_builder_ocr_pages:{document_key}",
                        )
                        st.number_input(
                            "DPI",
                            min_value=150,
                            max_value=600,
                            value=int(st.session_state.get(f"profile_builder_ocr_dpi:{document_key}", 300)),
                            step=50,
                            key=f"profile_builder_ocr_dpi:{document_key}",
                        )
                        if st.button("Запустить OCR", key=f"profile_builder_run_ocr:{document_key}", disabled=not is_available):
                            with st.spinner(f"OCR / {selected_engine} выполняется по настройкам профиля..."):
                                run_user_profile_ocr({}, profile_builder_extraction_config(document_key).get("ocr") or {})
                            st.success("OCR завершён. Перейдите к выбору OCR-кандидатов.")
                            st.rerun()

                elif table_catalog_df.empty:
                    if builder_source == "ocr":
                        ocr_has_result = (
                            not ocr_result_df.empty
                            or (
                                isinstance(st.session_state.get(profile_builder_ocr_result_key), pd.DataFrame)
                                and not st.session_state.get(profile_builder_ocr_result_key).empty
                            )
                        )
                        ocr_catalog_rows = len(stored_profile_builder_ocr_catalog_df)
                        if not ocr_has_result:
                            st.info("Сначала запустите OCR.")
                        elif ocr_catalog_rows == 0:
                            st.info("OCR не нашёл таблиц или кандидатов.")
                        else:
                            st.error(
                                "OCR catalog есть, но список блоков для выбранного source пуст. "
                                "Проверьте source_kind и block_uid в диагностике выше."
                            )
                            st.json(
                                {
                                    "builder_source": builder_source,
                                    "available_ocr_catalog_ids": profile_builder_table_options(
                                        stored_profile_builder_ocr_catalog_df
                                    ),
                                    "all_catalog_source_kinds": all_table_catalog_df.get(
                                        "source_kind",
                                        pd.Series(dtype="object"),
                                    )
                                    .fillna("")
                                    .astype(str)
                                    .drop_duplicates()
                                    .tolist(),
                                    "source_state_key": source_state_key,
                                    "source_state_value": st.session_state.get(source_state_key),
                                    "source_widget_key": source_widget_key,
                                    "source_widget_value": st.session_state.get(source_widget_key),
                                }
                            )
                    else:
                        st.info("Для выбранного источника пока нет таблиц или OCR-кандидатов.")
                else:
                    table_options = profile_builder_table_options(table_catalog_df)
                    selected_tables_state_key = f"profile_builder_block_selection_applied:{document_key}:{current_builder_engine}"
                    draft_tables_state_key = f"profile_builder_block_selection_draft:{document_key}:{current_builder_engine}"
                    selected_rows_state_key = f"profile_builder_manual_rows:{document_key}:{current_builder_engine}"
                    if selected_tables_state_key not in st.session_state:
                        st.session_state[selected_tables_state_key] = table_options[:1]
                    if draft_tables_state_key not in st.session_state:
                        st.session_state[draft_tables_state_key] = list(st.session_state[selected_tables_state_key])
                    stored_selected_builder_table_keys = [
                        str(table_key)
                        for table_key in st.session_state.get(selected_tables_state_key, [])
                        if str(table_key)
                    ]
                    selected_builder_table_keys = [
                        str(table_key)
                        for table_key in stored_selected_builder_table_keys
                        if str(table_key) and str(table_key) in set(table_options)
                    ]
                    missing_selected_block_uids = [
                        table_key for table_key in stored_selected_builder_table_keys if table_key not in set(table_options)
                    ]
                    if selected_builder_table_keys != st.session_state.get(selected_tables_state_key, []):
                        st.session_state[selected_tables_state_key] = selected_builder_table_keys

                    selected_source_rows_raw = select_source_rows_for_block_uids(
                        raw_table_rows,
                        ocr_candidates_df,
                        selected_builder_table_keys,
                    )
                    selected_source_rows = profile_builder_corrected_rows_from_state(
                        document_key,
                        builder_source=builder_source,
                        selected_block_uids=selected_builder_table_keys,
                        fallback_rows=selected_source_rows_raw,
                    )
                    if selected_rows_state_key not in st.session_state:
                        st.session_state[selected_rows_state_key] = []
                    current_source_row_keys = {profile_builder_source_row_key(row) for row in selected_source_rows}
                    selected_row_keys = [
                        str(row_key)
                        for row_key in st.session_state.get(selected_rows_state_key, [])
                        if str(row_key) and str(row_key) in current_source_row_keys
                    ]
                    if selected_row_keys != st.session_state.get(selected_rows_state_key, []):
                        st.session_state[selected_rows_state_key] = selected_row_keys
                    max_builder_columns = profile_builder_max_columns(selected_source_rows)
                    selected_source_row_set = set(selected_row_keys)
                    applied_selected_source_rows = [
                        row for row in selected_source_rows if profile_builder_source_row_key(row) in selected_source_row_set
                    ]
                    token_mapping_mode = profile_builder_uses_token_mapping(applied_selected_source_rows or selected_source_rows)
                    max_builder_tokens = profile_builder_max_tokens(applied_selected_source_rows or selected_source_rows)

                    if wizard_step == "2. Таблицы/блоки":
                        st.write(
                            "Выберите таблицу, из которой хотите извлекать данные. "
                            "Если нужная таблица разбилась на несколько частей, выберите несколько таблиц."
                        )
                        table_editor_df = prepare_profile_builder_catalog_editor(
                            table_catalog_df,
                            st.session_state.get(draft_tables_state_key, selected_builder_table_keys),
                        )
                        
                        # Fix Arrow serialization: ensure strict string/numeric typing
                        for col in ["block_uid", "table_key", "source_kind", "block_title", "Таблица", "Краткий preview", "extraction_method"]:
                            if col in table_editor_df.columns:
                                table_editor_df[col] = table_editor_df[col].fillna("").astype(str)
                        for col in ["Страница", "Найдено строк", "Найдено колонок"]:
                            if col in table_editor_df.columns:
                                table_editor_df[col] = pd.to_numeric(table_editor_df[col], errors="coerce").fillna(0).astype(int)

                        block_action_cols = st.columns(3)
                        if block_action_cols[0].button("Выбрать все блоки", key=f"profile_builder_blocks_all:{document_key}:{current_builder_engine}"):
                            st.session_state[draft_tables_state_key] = table_options
                            st.rerun()
                        if block_action_cols[1].button("Снять выбор", key=f"profile_builder_blocks_clear:{document_key}:{current_builder_engine}"):
                            st.session_state[draft_tables_state_key] = []
                            st.rerun()

                        with st.form(f"profile_builder_block_selection_form:{document_key}:{current_builder_engine}"):
                            edited_tables_df = st.data_editor(
                                table_editor_df,
                                use_container_width=True,
                                hide_index=True,
                                disabled=[
                                    "block_uid",
                                    "table_key",
                                    "source_kind",
                                    "Страница",
                                    "block_title",
                                    "Таблица",
                                    "Найдено строк",
                                    "Найдено колонок",
                                    "Краткий preview",
                                    "Качество",
                                    "extraction_method",
                                ],
                                column_config={
                                    "Использовать эту таблицу": st.column_config.CheckboxColumn(
                                        "Использовать эту таблицу"
                                    ),
                                    "block_uid": st.column_config.TextColumn("block_uid", help="Стабильный ID выбранного блока."),
                                    "source_kind": st.column_config.TextColumn("source_kind"),
                                    "block_title": st.column_config.TextColumn("block_title", width="medium"),
                                    "Краткий preview": st.column_config.TextColumn("Краткий preview", width="large"),
                                },
                                key=f"profile_builder_table_editor:{document_key}:{current_builder_engine}",
                            )
                            apply_block_selection = st.form_submit_button("Применить выбор блоков")
                        
                        if apply_block_selection:
                            # Direct reading from editor instead of relying purely on helper
                            updated_table_keys = []
                            if PROFILE_BUILDER_USE_BLOCK_COLUMN in edited_tables_df.columns:
                                checked_mask = edited_tables_df[PROFILE_BUILDER_USE_BLOCK_COLUMN].fillna(False).astype(bool)
                                checked_rows = edited_tables_df.loc[checked_mask]
                                for _, row in checked_rows.iterrows():
                                    if "block_uid" in row and row["block_uid"]:
                                        updated_table_keys.append(str(row["block_uid"]))
                                    elif "table_key" in row and row["table_key"]:
                                        updated_table_keys.append(str(row["table_key"]))

                            updated_missing_block_uids = [
                                table_key for table_key in updated_table_keys if table_key not in set(table_options)
                            ]
                            st.session_state[draft_tables_state_key] = updated_table_keys
                            st.session_state[selected_tables_state_key] = updated_table_keys
                            st.session_state[f"profile_builder_last_block_selection_debug:{document_key}"] = {
                                "submitted": True,
                                "updated_block_uids": updated_table_keys,
                                "updated_missing_block_uids": updated_missing_block_uids,
                                "available_block_uids": table_options,
                                "editor_columns": list(edited_tables_df.columns),
                                "editor_checked_rows": edited_tables_df.loc[
                                    edited_tables_df[PROFILE_BUILDER_USE_BLOCK_COLUMN].fillna(False).astype(bool)
                                ].to_dict("records")
                                if PROFILE_BUILDER_USE_BLOCK_COLUMN in edited_tables_df.columns
                                else [],
                            }
                            if updated_table_keys != selected_builder_table_keys:
                                st.session_state.pop(f"profile_builder_table_reconstruction_applied:{document_key}", None)
                                st.session_state.pop(f"profile_builder_split_pattern_draft:{document_key}", None)
                            st.session_state.pop(selected_rows_state_key, None)
                            st.rerun()
                            
                        if missing_selected_block_uids:
                            st.warning(f"Внимание: Выбранные блоки ({', '.join(missing_selected_block_uids)}) отсутствуют в каталоге текущего OCR engine.")
                        block_metric_cols = st.columns(2)
                        block_metric_cols[0].metric("Всего блоков", len(table_options))
                        block_metric_cols[1].metric("Выбрано блоков", len(selected_builder_table_keys))
                        if not selected_builder_table_keys:
                            st.warning("Выберите хотя бы один блок.")
                        if missing_selected_block_uids:
                            st.error("Выбранные block_uid не найдены в OCR catalog.")
                            st.json(
                                {
                                    "missing_selected_block_uids": missing_selected_block_uids,
                                    "available_block_uids": table_options,
                                }
                            )

                        with st.expander("Проблемы со структурой таблицы"):
                            applied_config = (
                                st.session_state.get(f"profile_builder_table_reconstruction_applied:{document_key}") or {}
                            )
                            applied_method = str(applied_config.get("method") or "none")
                            applied_pattern = str(applied_config.get("pattern") or "")

                            st.caption(
                                f"Текущий метод: `{applied_method}`"
                                + (f" (pattern: `{applied_pattern}`)" if applied_pattern else "")
                            )
                            raw_evidence_df = pd.DataFrame(
                                [
                                    {
                                        "page": row.get("page"),
                                        "source_row_id": profile_builder_source_row_key(row),
                                        "evidence_text": row.get("evidence_text"),
                                    }
                                    for row in selected_source_rows_raw[:100]
                                ]
                            )
                            if raw_evidence_df.empty:
                                st.info("Сначала выберите таблицу.")
                            else:
                                st.dataframe(raw_evidence_df, use_container_width=True, hide_index=True)

                            st.divider()
                            st.write("Настройка реконструкции:")

                            recon_method = st.radio(
                                "Метод реконструкции",
                                options=["none", "split_by_regex", "pair_name_row_with_following_value_row"],
                                format_func=lambda x: {
                                    "none": "Без изменений",
                                    "split_by_regex": "Разделить по регулярному выражению (если колонки склеились)",
                                    "pair_name_row_with_following_value_row": "Склеивать строку с названием со следующей строкой-значением (для тарифов)",
                                }.get(x, x),
                                index=(
                                    ["none", "split_by_regex", "pair_name_row_with_following_value_row"].index(
                                        applied_method
                                    )
                                    if applied_method in ["none", "split_by_regex", "pair_name_row_with_following_value_row"]
                                    else 0
                                ),
                                key=f"profile_builder_recon_method_radio:{document_key}",
                            )

                            draft_pattern = ""
                            if recon_method == "split_by_regex":
                                draft_pattern = st.text_input(
                                    "Регулярное выражение для разделения колонок",
                                    value=applied_pattern if applied_method == "split_by_regex" else r"\s{1,}|\|",
                                    key=f"profile_builder_split_pattern_input:{document_key}",
                                )

                            if st.button("Применить реконструкцию", key=f"profile_builder_apply_recon:{document_key}"):
                                recon_config = {"method": recon_method}
                                if recon_method == "split_by_regex":
                                    recon_config["pattern"] = draft_pattern

                                corrected_rows = apply_table_reconstruction(selected_source_rows_raw, recon_config)
                                st.session_state[f"profile_builder_table_reconstruction_applied:{document_key}"] = {
                                    "source": builder_source,
                                    "block_uids": list(selected_builder_table_keys),
                                    "method": recon_method,
                                    "pattern": draft_pattern,
                                    "rows": corrected_rows,
                                }
                                st.session_state.pop(selected_rows_state_key, None)
                                st.rerun()

                        st.caption("Preview выбранной таблицы")
                        if selected_source_rows:
                            st.dataframe(
                                source_rows_to_preview_df(selected_source_rows, limit=50),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("Выберите хотя бы одну таблицу.")

                    elif wizard_step == "3. Строки":
                        st.write(
                            "Отметьте строки, которые должны попасть в итоговую таблицу. "
                            "Заголовки, подзаголовки и пустые строки лучше не выбирать."
                        )
                        if not selected_source_rows:
                            st.warning("Сначала выберите блоки на шаге 2.")
                        else:
                            row_mode = st.radio(
                                "Режим выбора строк",
                                ["Ручной выбор строк", "Правила отбора строк"],
                                horizontal=True,
                                key=f"profile_builder_row_mode:{document_key}",
                            )
                            if row_mode == "Ручной выбор строк":
                                quick_cols = st.columns(3)
                                if quick_cols[0].button("Выбрать все строки", key=f"profile_builder_rows_all:{document_key}"):
                                    st.session_state[selected_rows_state_key] = [
                                        profile_builder_source_row_key(row) for row in selected_source_rows
                                    ]
                                    st.rerun()
                                if quick_cols[1].button(
                                    "Выбрать строки с числовыми значениями",
                                    key=f"profile_builder_rows_numeric:{document_key}",
                                ):
                                    st.session_state[selected_rows_state_key] = [
                                        profile_builder_source_row_key(row)
                                        for row in selected_source_rows
                                        if profile_builder_has_numeric_value(row)
                                    ]
                                    st.rerun()
                                if quick_cols[2].button("Снять выбор", key=f"profile_builder_rows_clear:{document_key}"):
                                    st.session_state[selected_rows_state_key] = []
                                    st.rerun()

                                row_preview_df = source_rows_to_preview_df(selected_source_rows, limit=1000)
                                row_keys = [profile_builder_source_row_key(row) for row in selected_source_rows[: len(row_preview_df)]]
                                selected_key_set = set(selected_row_keys)
                                row_preview_df.insert(0, "source_row_key", row_keys)
                                row_preview_df.insert(
                                    0,
                                    "use_row",
                                    [row_key in selected_key_set for row_key in row_keys],
                                )
                                with st.form(f"profile_builder_row_selection_form:{document_key}"):
                                    edited_rows_df = st.data_editor(
                                        row_preview_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        disabled=[column for column in row_preview_df.columns if column != "use_row"],
                                        column_config={
                                            "use_row": st.column_config.CheckboxColumn("Использовать строку"),
                                            "source_row_key": st.column_config.TextColumn("ID строки"),
                                            "evidence_text": st.column_config.TextColumn("Фрагмент-источник", width="large"),
                                        },
                                        key=f"profile_builder_row_editor:{document_key}",
                                    )
                                    apply_row_selection = st.form_submit_button("Применить выбор строк")
                                if apply_row_selection:
                                    st.session_state[selected_rows_state_key] = (
                                        edited_rows_df.loc[
                                            edited_rows_df["use_row"].fillna(False).astype(bool),
                                            "source_row_key",
                                        ]
                                        .fillna("")
                                        .astype(str)
                                        .tolist()
                                    )
                                    st.rerun()
                            else:
                                rule_cols = st.columns(2)
                                rule_cols[0].text_input(
                                    "Начать после строки, содержащей",
                                    key=f"profile_builder_rule_keep_after:{document_key}",
                                )
                                rule_cols[1].text_input(
                                    "Остановиться перед строкой, содержащей",
                                    key=f"profile_builder_rule_keep_until:{document_key}",
                                )
                                check_cols = st.columns(3)
                                check_cols[0].checkbox(
                                    "Оставить только строки, где есть число",
                                    value=False,
                                    key=f"profile_builder_rule_numeric_only:{document_key}",
                                )
                                check_cols[1].checkbox(
                                    "Пропустить строки, где все value-колонки пустые",
                                    value=True,
                                    key=f"profile_builder_rule_skip_empty_values:{document_key}",
                                )
                                check_cols[2].checkbox(
                                    "Пропустить строки, где значение равно '-' или '—'",
                                    value=True,
                                    key=f"profile_builder_rule_skip_dash_values:{document_key}",
                                )
                                if st.button("Применить правила к строкам", key=f"profile_builder_apply_rules:{document_key}"):
                                    st.session_state[selected_rows_state_key] = profile_builder_rows_matching_rules(
                                        selected_source_rows,
                                        document_key,
                                    )
                                    st.rerun()

                                selected_key_set = set(selected_row_keys)
                                rules_preview_df = source_rows_to_preview_df(selected_source_rows, limit=200)
                                row_keys = [profile_builder_source_row_key(row) for row in selected_source_rows[: len(rules_preview_df)]]
                                rules_preview_df.insert(0, "use_row", [row_key in selected_key_set for row_key in row_keys])
                                st.dataframe(rules_preview_df, use_container_width=True, hide_index=True)

                            selected_count = len(selected_row_keys)
                            metric_cols = st.columns(3)
                            metric_cols[0].metric("Всего строк в таблице", len(selected_source_rows))
                            metric_cols[1].metric("Выбрано строк", selected_count)
                            metric_cols[2].metric("Будет пропущено строк", max(0, len(selected_source_rows) - selected_count))

                    elif wizard_step == "4. Колонки":
                        st.write(
                            "Укажите смысл каждой колонки. Если в таблице несколько числовых колонок, "
                            "каждая из них станет отдельной строкой в итоговом long-format."
                        )
                        if not selected_source_rows:
                            st.warning("Сначала выберите блоки на шаге 2.")
                        else:
                            selected_key_set = set(selected_row_keys)
                            rows_for_columns = [
                                row for row in selected_source_rows if profile_builder_source_row_key(row) in selected_key_set
                            ] or selected_source_rows
                            if token_mapping_mode:
                                st.info("Для выбранного OCR-блока включена разметка числовых токенов.")
                                token_role_options = ["ignore", "value", "year", "volume", "trade_value", "change"]
                                for token_index in range(1, max_builder_tokens + 1):
                                    samples = profile_builder_token_samples(rows_for_columns, token_index)
                                    with st.expander(f"token_{token_index}", expanded=token_index <= 4):
                                        st.write("Примеры: " + (" | ".join(samples) if samples else "нет данных"))
                                        token_role = st.selectbox(
                                            "Роль токена",
                                            token_role_options,
                                            format_func=lambda value: {
                                                "ignore": "Не использовать",
                                                "value": "Значение",
                                                "year": "Год",
                                                "volume": "Объём",
                                                "trade_value": "Стоимость",
                                                "change": "Изменение",
                                            }.get(value, value),
                                            key=f"profile_builder_token_role:{document_key}:{token_index}",
                                        )
                                        if token_role != "ignore":
                                            token_cols = st.columns(5)
                                            token_cols[0].text_input(
                                                "metric",
                                                value="volume" if token_role == "volume" else "trade_value" if token_role == "trade_value" else token_role,
                                                key=f"profile_builder_token_metric:{document_key}:{token_index}",
                                            )
                                            token_cols[1].text_input(
                                                "scenario",
                                                value="",
                                                key=f"profile_builder_token_scenario:{document_key}:{token_index}",
                                            )
                                            token_cols[2].text_input(
                                                "year",
                                                value="",
                                                key=f"profile_builder_token_year:{document_key}:{token_index}",
                                            )
                                            token_cols[3].text_input(
                                                "unit",
                                                value="thousand_tons" if token_role == "volume" else "million_usd" if token_role == "trade_value" else "",
                                                key=f"profile_builder_token_unit:{document_key}:{token_index}",
                                            )
                                            token_cols[4].text_input(
                                                "currency",
                                                value="USD" if token_role == "trade_value" else "",
                                                key=f"profile_builder_token_currency:{document_key}:{token_index}",
                                            )
                            else:
                                for column_index in range(1, max_builder_columns + 1):
                                    samples = profile_builder_column_samples(rows_for_columns, column_index)
                                    with st.expander(f"Колонка {column_index}", expanded=column_index <= 5):
                                        st.write("Первые значения: " + (" | ".join(samples) if samples else "нет данных"))
                                        role = st.selectbox(
                                            "Что это за колонка?",
                                            PROFILE_BUILDER_ROLE_VALUES,
                                            format_func=lambda value: PROFILE_BUILDER_ROLE_LABELS.get(value, value),
                                            index=PROFILE_BUILDER_ROLE_VALUES.index(profile_builder_default_role(column_index)),
                                            key=f"profile_builder_wizard_role:{document_key}:{column_index}",
                                        )
                                        if role == "value":
                                            value_cols = st.columns(4)
                                            value_cols[0].text_input(
                                                "Название показателя",
                                                value="tariff",
                                                placeholder="tariff, price, volume, trade_value",
                                                key=f"profile_builder_wizard_metric:{document_key}:{column_index}",
                                            )
                                            value_cols[1].text_input(
                                                "Сценарий / тип значения",
                                                value=profile_builder_default_scenario(column_index),
                                                placeholder="direct, intraport_movement, 2024",
                                                key=f"profile_builder_wizard_scenario:{document_key}:{column_index}",
                                            )
                                            value_cols[2].text_input(
                                                "Единица измерения",
                                                value="",
                                                placeholder="RUB, ton, thousand_tons, percent",
                                                key=f"profile_builder_wizard_unit:{document_key}:{column_index}",
                                            )
                                            value_cols[3].text_input(
                                                "Валюта",
                                                value="RUB" if column_index in {4, 5} else "",
                                                placeholder="RUB, USD, EUR",
                                                key=f"profile_builder_wizard_currency:{document_key}:{column_index}",
                                            )
                                            st.text_input(
                                                "Год",
                                                value="",
                                                placeholder="2024",
                                                key=f"profile_builder_wizard_year:{document_key}:{column_index}",
                                            )
                                            st.selectbox(
                                                "Тип значения",
                                                ["numeric", "percent", "text", "date"],
                                                index=0,
                                                key=f"profile_builder_wizard_value_type:{document_key}:{column_index}",
                                            )

                    elif wizard_step == "5. LLM генератор":
                        st.subheader("Генерация профиля через LLM")
                        st.write("Опишите, какие данные нужно извлечь из документа на естественном языке.")
                        
                        llm_instruction = st.text_area(
                            "Инструкция для LLM",
                            value=st.session_state.get(f"llm_profile_instruction:{document_key}", ""),
                            placeholder="Мне нужны строки с зерновыми грузами на странице 2, кроме погрузки на автотранспорт. Извлеки код услуги, название услуги, единицу измерения и тариф. Валюта RUB.",
                            key=f"llm_profile_instruction_input:{document_key}",
                            height=150
                        )
                        st.session_state[f"llm_profile_instruction:{document_key}"] = llm_instruction
                        
                        gen_cols = st.columns(2)
                        target_pages = gen_cols[0].text_input("Страницы (например, 2 или 1,2,3)", value=st.session_state.get(f"llm_profile_pages:{document_key}", "2"), key=f"llm_profile_pages_input:{document_key}")
                        st.session_state[f"llm_profile_pages:{document_key}"] = target_pages
                        
                        ocr_engine_llm = gen_cols[1].selectbox(
                            "OCR Engine", 
                            ["yandex_vision", "tesseract", "paddleocr"], 
                            index=["yandex_vision", "tesseract", "paddleocr"].index(st.session_state.get(f"llm_profile_ocr_engine:{document_key}", "yandex_vision")),
                            key=f"llm_profile_ocr_engine_input:{document_key}"
                        )
                        st.session_state[f"llm_profile_ocr_engine:{document_key}"] = ocr_engine_llm
                        
                        use_context = st.checkbox(
                            "Использовать OCR контекст (рекомендуется)", 
                            value=st.session_state.get(f"llm_profile_use_context:{document_key}", True),
                            key=f"llm_profile_use_context_input:{document_key}"
                        )
                        st.session_state[f"llm_profile_use_context:{document_key}"] = use_context
                        
                        if st.button("Сгенерировать профиль через LLM", key=f"llm_profile_generate_btn:{document_key}", type="primary"):
                            if not llm_instruction:
                                st.error("Пожалуйста, введите инструкцию.")
                            else:
                                with st.spinner("LLM генерирует профиль..."):
                                    try:
                                        generator = LLMProfileGenerator()
                                        
                                        # Build document context
                                        doc_context = {
                                            "ocr_candidates_df": ocr_candidates_df,
                                            "raw_rows": raw_table_rows
                                        }
                                        
                                        # Mock existing schema for hint
                                        existing_schema = build_profile_builder_config(
                                            document_key,
                                            selected_builder_table_keys,
                                            selected_row_keys,
                                            max_builder_columns,
                                        )
                                        
                                        generated_profile = generator.generate_profile(
                                            doc_context,
                                            llm_instruction,
                                            existing_schema=existing_schema
                                        )
                                        
                                        # Validate
                                        validation_errors = validate_generated_profile(generated_profile)
                                        if validation_errors:
                                            st.warning("Профиль сгенерирован с ошибками валидации:\n" + "\n".join(validation_errors))
                                        
                                        st.session_state[f"llm_generated_profile:{document_key}"] = generated_profile
                                        st.success("Профиль успешно сгенерирован!")
                                    except Exception as e:
                                        st.error(f"Ошибка при генерации профиля: {e}")
                                        
                        generated_profile = st.session_state.get(f"llm_generated_profile:{document_key}")
                        if generated_profile:
                            st.subheader("Сгенерированный профиль")
                            st.yaml(generated_profile)
                            
                            if st.button("Применить сгенерированный профиль", key=f"llm_profile_apply_btn:{document_key}"):
                                # Mark LLM profile as active for preview/save
                                st.session_state[f"llm_profile_active:{document_key}"] = True
                                st.success("Профиль применён. Теперь вы можете проверить его на шаге Preview.")

                    elif wizard_step == "6. Preview":
                        llm_active = st.session_state.get(f"llm_profile_active:{document_key}", False)
                        llm_profile = st.session_state.get(f"llm_generated_profile:{document_key}")
                        
                        if llm_active and llm_profile:
                            builder_profile_config = llm_profile
                        else:
                            builder_profile_config = build_profile_builder_config(
                                document_key,
                                selected_builder_table_keys,
                                selected_row_keys,
                                max_builder_columns,
                                use_token_mapping=token_mapping_mode,
                                max_tokens=max_builder_tokens,
                            )
                        builder_preview_df = apply_user_profile_to_sources(
                            raw_table_rows,
                            ocr_candidates_df,
                            builder_profile_config,
                        )
                        table_config = (builder_profile_config.get("tables") or [{}])[0]
                        filtered_source_rows = apply_row_filters(
                            selected_source_rows,
                            table_config.get("row_filters"),
                            table_config.get("column_mapping"),
                        )
                        metrics = profile_builder_preview_metrics(builder_preview_df, len(filtered_source_rows))
                        
                        with st.expander("Отладка извлечения (Preview)"):
                            # Gather engine diagnostics
                            ocr_result_methods = []
                            if not ocr_result_df.empty and "extraction_method" in ocr_result_df.columns:
                                ocr_result_methods = ocr_result_df["extraction_method"].unique().tolist()

                            ocr_candidate_methods = []
                            if not ocr_candidates_df.empty and "extraction_method" in ocr_candidates_df.columns:
                                ocr_candidate_methods = ocr_candidates_df["extraction_method"].unique().tolist()

                            # Reconstruction diagnostics
                            recon_config = table_config.get("table_reconstruction") or {}
                            recon_mode = recon_config.get("method", "none")
                            paired_rows_count = sum(1 for r in filtered_source_rows if r.get("table_reconstruction_method") == "pair_name_row_with_following_value_row")

                            recon_metrics = {
                                "metric": [
                                    "reconstruction_mode",
                                    "source_rows_before_reconstruction",
                                    "output_rows_after_reconstruction",
                                    "paired_rows_count",
                                    "has_token_mapping",
                                    "has_column_mapping",
                                    "token_mapping_mode",
                                    "ocr_result_rows",
                                    "ocr_candidate_rows"
                                ],
                                "value": [
                                    str(recon_mode),
                                    str(len(selected_source_rows)),
                                    str(len(filtered_source_rows)),
                                    str(paired_rows_count),
                                    str(bool(table_config.get("token_mapping"))),
                                    str(bool(table_config.get("column_mapping"))),
                                    str(token_mapping_mode),
                                    str(len(ocr_result_df)),
                                    str(len(ocr_candidates_df))
                                ]
                            }
                            st.dataframe(pd.DataFrame(recon_metrics), use_container_width=True)

                            debug_info = {
                                "selected_block_uids": selected_builder_table_keys,
                                "selected_row_uids": selected_row_keys,
                                "num_source_rows_total": len(selected_source_rows),
                                "num_filtered_rows": len(filtered_source_rows),
                                "num_output_rows": len(builder_preview_df),
                                "builder_source": builder_source,
                                "table_config": table_config,
                                "ocr_engine_diagnostics": {
                                    "selected_ocr_engine_from_ui": current_builder_engine,
                                    "ocr_result_rows": len(ocr_result_df),
                                    "ocr_candidate_rows": len(ocr_candidates_df),
                                    "ocr_result_extraction_methods": ocr_result_methods,
                                    "ocr_candidate_extraction_methods": ocr_candidate_methods,
                                }
                            }
                            st.json(debug_info)
                            if selected_source_rows:
                                st.write("Примеры source_row_uids в таблице:")
                                st.write([profile_builder_source_row_key(r) for r in selected_source_rows[:5]])
                            if selected_row_keys:
                                st.write("Пересечение выбранных ключей с текущими:")
                                current_keys = {profile_builder_source_row_key(r) for r in selected_source_rows}
                                intersection = [k for k in selected_row_keys if k in current_keys]
                                st.write(f"Найдено {len(intersection)} из {len(selected_row_keys)} выбранных строк.")

                        metric_cols = st.columns(5)
                        metric_cols[0].metric("Строк выбрано", metrics["selected_rows"])
                        metric_cols[1].metric("Итоговых строк получится", metrics["output_rows"])
                        metric_cols[2].metric("Успешно распознано значений", metrics["parsed_values"])
                        metric_cols[3].metric("Требуют проверки", metrics["needs_review"])
                        metric_cols[4].metric("Ошибки", metrics["errors"])
                        st.write(
                            "Проверьте, что итоговые строки выглядят правильно. Если часть значений не распознана, "
                            "они попадут в ручную проверку."
                        )
                        st.caption(
                            "Одна исходная строка может дать несколько итоговых строк, если в ней несколько колонок "
                            "со значениями."
                        )
                        if metrics["output_rows"] > metrics["selected_rows"]:
                            st.info(
                                f"Выбрано исходных строк: {metrics['selected_rows']}. "
                                f"Итоговых строк получится: {metrics['output_rows']}."
                            )
                        if builder_preview_df.empty:
                            st.info("Preview пока пустой: выберите строки и назначьте хотя бы одну колонку со значением.")
                        else:
                            preview_export_df = select_profile_builder_preview_columns(builder_preview_df)
                            st.dataframe(preview_export_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Скачать preview structured CSV",
                                data=export_to_csv_cached(select_user_profile_export_columns(builder_preview_df)),
                                file_name=f"{builder_profile_config['profile_name']}_preview.csv",
                                mime="text/csv",
                                key=f"profile_builder_preview_csv:{document_key}",
                            )

                    elif wizard_step == "7. Сохранение":
                        profile_cols = st.columns(2)
                        
                        llm_active = st.session_state.get(f"llm_profile_active:{document_key}", False)
                        llm_profile = st.session_state.get(f"llm_generated_profile:{document_key}")
                        
                        default_display = "Тарифы НМТП"
                        default_name = "nmpt_tariffs"
                        if llm_active and llm_profile:
                            default_display = llm_profile.get("display_name", default_display)
                            default_name = llm_profile.get("profile_name", default_name)

                        profile_cols[0].text_input(
                            "Название профиля",
                            value=st.session_state.get(f"profile_builder_wizard_display:{document_key}", default_display),
                            key=f"profile_builder_wizard_display:{document_key}",
                        )
                        profile_cols[1].text_input(
                            "Техническое имя",
                            value=st.session_state.get(f"profile_builder_wizard_name:{document_key}", default_name),
                            key=f"profile_builder_wizard_name:{document_key}",
                        )
                        st.text_area(
                            "Ключевые слова для автоопределения документа",
                            value=st.session_state.get(f"profile_builder_wizard_keywords:{document_key}", ""),
                            placeholder="ТАРИФЫ ПАО\nНаименование груза\nТариф в рублях РФ",
                            key=f"profile_builder_wizard_keywords:{document_key}",
                        )
                        save_cols = st.columns(2)
                        save_cols[0].text_input(
                            "Фразы внутри таблицы для автопоиска",
                            value=st.session_state.get(f"profile_builder_selector_text:{document_key}", ""),
                            placeholder="Наименование груза",
                            key=f"profile_builder_selector_text:{document_key}",
                        )
                        save_cols[1].text_input(
                            "Название секции",
                            value=st.session_state.get(f"profile_builder_wizard_section:{document_key}", "user_profile_section"),
                            key=f"profile_builder_wizard_section:{document_key}",
                        )

                        if llm_active and llm_profile:
                            builder_profile_config = llm_profile
                        else:
                            builder_profile_config = build_profile_builder_config(
                                document_key,
                                selected_builder_table_keys,
                                selected_row_keys,
                                max_builder_columns,
                                use_token_mapping=token_mapping_mode,
                                max_tokens=max_builder_tokens,
                            )
                        builder_preview_df = apply_user_profile_to_sources(
                            raw_table_rows,
                            ocr_candidates_df,
                            builder_profile_config,
                        )
                        if builder_preview_df.empty:
                            st.warning("Перед сохранением проверьте, что preview на шаге 4 создаёт structured rows.")
                        else:
                            st.success(f"Preview готов: {len(builder_preview_df)} structured rows.")

                        config_cols = st.columns(3)
                        config_cols[0].download_button(
                            "Скачать профиль YAML",
                            data=dump_user_profile_yaml(builder_profile_config).encode("utf-8"),
                            file_name=f"{builder_profile_config['profile_name']}.yaml",
                            mime="application/x-yaml",
                            key=f"profile_builder_download_yaml:{document_key}",
                        )
                        if config_cols[1].button(
                            "Сохранить профиль",
                            key=f"profile_builder_save:{document_key}",
                        ):
                            saved_profile_path = save_user_profile(builder_profile_config, USER_PROFILES_DIR)
                            st.success(
                                "Профиль сохранён. Теперь при загрузке похожего документа система предложит "
                                f"применить профиль `{builder_profile_config.get('display_name')}`. Файл: {saved_profile_path}"
                            )
                        if config_cols[2].button(
                            "Применить профиль к текущему документу",
                            disabled=builder_preview_df.empty,
                            key=f"profile_builder_apply:{document_key}",
                        ):
                            st.session_state[user_profile_structured_key] = builder_preview_df
                            st.session_state[applied_user_profile_key] = builder_profile_config
                            st.rerun()
            elif tab_key == "review":
                review_rows = reviewed_df[reviewed_df["validation_status"].isin(["warning", "failed"])]
                if review_rows.empty:
                    st.success("Нет структурированных строк, требующих ручной проверки.")
                else:
                    st.caption(
                        "Здесь показаны только структурированные строки, которые система не приняла автоматически. "
                        "Сырые PDF-фрагменты не считаются ошибками и доступны на отдельной вкладке."
                    )
                    selected_filter = st.radio(
                        "Фильтр",
                        list(REVIEW_FILTERS.keys()),
                        horizontal=True,
                        key="review_filter",
                    )
                    review_filter = REVIEW_FILTERS[selected_filter]
                    if review_filter != "all":
                        review_rows = review_rows[review_rows["validation_status"] == review_filter]

                    if review_rows.empty:
                        st.info("По выбранному фильтру строк нет.")
                    else:
                        st.dataframe(
                            rename_columns_for_ui(translate_status_columns(select_review_columns(review_rows))),
                            use_container_width=True,
                            hide_index=True,
                        )

    if document_mode != "profile_setup":
        with st.expander("Техническая диагностика"):
            diag_cols = st.columns(4)
            diag_cols[0].metric("Сырых строк", len(raw_rows))
            diag_cols[1].metric("PDF-таблиц", int(raw_summary.get("table_count") or 0))
            diag_cols[2].metric("OCR-строк", len(ocr_result_df))
            diag_cols[3].metric("OCR-кандидатов", len(ocr_candidates_df))
            if not raw_rows.empty:
                st.caption("Raw fragments")
                st.dataframe(select_technical_raw_export_columns(raw_rows).head(200), use_container_width=True, hide_index=True)
            if not raw_table_summary_df.empty:
                st.caption("PDF table candidates")
                st.dataframe(format_table_summary_for_ui(raw_table_summary_df), use_container_width=True, hide_index=True)
            if not ocr_candidates_df.empty:
                st.caption("OCR candidates")
                st.dataframe(format_ocr_candidates_for_ui(ocr_candidates_df), use_container_width=True, hide_index=True)

    export_ocr_candidates_df = st.session_state.get(ocr_candidates_key)
    if export_ocr_candidates_df is None and st.session_state.get("ocr_document_key") == document_key:
        export_ocr_candidates_df = st.session_state.get("ocr_candidates_df")
    if export_ocr_candidates_df is None:
        export_ocr_candidates_df = ocr_candidates_df
    selected_export_ocr_candidates_df = st.session_state.get(selected_ocr_candidates_key)
    if selected_export_ocr_candidates_df is None:
        selected_export_ocr_candidates_df = pd.DataFrame()
    export_prototype_structured_df = st.session_state.get(prototype_structured_key)
    if export_prototype_structured_df is None:
        export_prototype_structured_df = prototype_structured_df
    export_mapped_complex_df = st.session_state.get(mapped_complex_key)
    if export_mapped_complex_df is None:
        export_mapped_complex_df = st.session_state.get("mapped_complex_df")
    if export_mapped_complex_df is None:
        export_mapped_complex_df = pd.DataFrame()
    export_reviewed_mapped_complex_df = st.session_state.get(reviewed_mapped_complex_key)
    if export_reviewed_mapped_complex_df is None or export_reviewed_mapped_complex_df.empty:
        export_reviewed_mapped_complex_df = prepare_reviewed_mapped_export(export_mapped_complex_df)
    else:
        export_reviewed_mapped_complex_df = prepare_reviewed_mapped_export(export_reviewed_mapped_complex_df)

    dashboard_summary = build_processing_dashboard_summary(
        processing_time=processing_time,
        file_type=file_type,
        active_profile=active_profile,
        profile_metadata=profile_metadata,
        bad_text_layer=bad_text_layer,
        raw_rows=raw_rows,
        raw_table_summary_df=raw_table_summary_df,
        ocr_result_df=ocr_result_df,
        ocr_candidates_df=export_ocr_candidates_df,
        selected_ocr_candidates_df=selected_export_ocr_candidates_df,
        structured_rows=reviewed_df,
        prototype_structured_df=export_prototype_structured_df,
        mapped_complex_df=export_mapped_complex_df,
        reviewed_mapped_df=export_reviewed_mapped_complex_df,
    )
    processing_funnel_df = build_processing_funnel(dashboard_summary, bad_text_layer=bad_text_layer)

    st.subheader("Итог обработки документа")
    readiness_text = (
        f"{dashboard_summary['readiness_status']}: {dashboard_summary['readiness_comment']}"
    )
    readiness_level = dashboard_summary.get("readiness_level")
    if readiness_level == "success":
        st.success(readiness_text)
    elif readiness_level == "error":
        st.error(readiness_text)
    elif readiness_level == "warning":
        st.warning(readiness_text)
    else:
        st.info(readiness_text)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Время обработки", f"{dashboard_summary['processing_time']:.2f}s")
    summary_cols[1].metric("Тип документа", str(dashboard_summary["file_type"]).upper())
    summary_cols[2].metric("Определённый профиль", str(dashboard_summary["active_profile"]))
    summary_cols[3].metric("OCR", str(dashboard_summary["ocr_status"]))

    extraction_cols = st.columns(4)
    extraction_cols[0].metric("Найдено таблиц", int(dashboard_summary["table_count"]))
    extraction_cols[1].metric("OCR-кандидатов", int(dashboard_summary["ocr_candidates_count"]))
    extraction_cols[2].metric("Сильных OCR-кандидатов", int(dashboard_summary["strong_ocr_candidates_count"]))
    extraction_cols[3].metric("Production structured rows", int(dashboard_summary["structured_rows_count"]))

    row_cols = st.columns(4)
    row_cols[0].metric("Простых structured rows", int(dashboard_summary["prototype_simple_rows_count"]))
    row_cols[1].metric("Сложных строк для mapping", int(dashboard_summary["prototype_complex_rows_count"]))
    row_cols[2].metric("Mapped rows", int(dashboard_summary["mapped_rows_count"]))
    row_cols[3].metric("Reviewed mapped rows", int(dashboard_summary["reviewed_mapped_rows_count"]))

    review_cols = st.columns(4)
    review_cols[0].metric("Auto-approved rows", int(dashboard_summary["auto_approved_rows"]))
    review_cols[1].metric("User-approved rows", int(dashboard_summary["user_approved_rows"]))
    review_cols[2].metric("Rows still need review", int(dashboard_summary["rows_still_need_review"]))
    review_cols[3].metric("Rows with warnings", int(dashboard_summary["rows_with_warnings"]))

    audit_coverage = dashboard_summary["audit_coverage"]
    audit_cols = st.columns(3)
    audit_cols[0].metric("Audit rows count", int(audit_coverage["audit_rows"]))
    audit_cols[1].metric("Audit total rows", int(audit_coverage["total_rows"]))
    audit_cols[2].metric("Покрытие audit trail", f"{float(audit_coverage['coverage_pct']):.1f}%")
    if float(audit_coverage["coverage_pct"]) < 100.0 and int(audit_coverage["total_rows"]) > 0:
        st.warning("Покрытие audit trail ниже 100%: не у всех строк заполнены source_file, page и evidence_text.")

    st.caption("Воронка обработки")
    st.dataframe(processing_funnel_df, use_container_width=True, hide_index=True)

    interpretation_messages = []
    if int(dashboard_summary["reviewed_mapped_rows_count"]) > 0:
        interpretation_messages.append(
            "Система извлекла данные из сложных таблиц, пользователь проверил часть строк, "
            "экспорт содержит audit trail: исходный файл, страницу, evidence text и статус проверки."
        )
    if int(dashboard_summary["rows_still_need_review"]) > 0:
        interpretation_messages.append(
            "Часть строк не была принята автоматически из-за OCR-ошибок или неоднозначной структуры таблицы. "
            "Они доступны в ручной проверке."
        )
    if bad_text_layer or dashboard_summary["ocr_status"] == "выполнен":
        interpretation_messages.append(
            "У документа плохой текстовый слой, поэтому применялся или рекомендуется OCR. "
            "Это типовой сценарий для сканов или PDF с повреждённым текстом."
        )
    if not interpretation_messages:
        interpretation_messages.append(
            "Документ обработан без дополнительных предупреждений на текущем этапе pipeline."
        )
    with st.expander("Что это значит", expanded=True):
        for message in interpretation_messages:
            st.write(message)

    st.subheader("Экспорт")
    export_start_time = time.perf_counter()
    output_name = f"{Path(uploaded_file.name).stem}_clean.xlsx"
    csv_output_name = f"{Path(uploaded_file.name).stem}_clean.csv"
    raw_output_name = f"{Path(uploaded_file.name).stem}_raw_extraction.csv"
    technical_raw_output_name = f"{Path(uploaded_file.name).stem}_technical_raw_export.csv"
    profile_candidates_csv_name = f"{Path(uploaded_file.name).stem}_profile_candidates.csv"
    profile_candidates_xlsx_name = f"{Path(uploaded_file.name).stem}_profile_candidates.xlsx"
    ocr_csv_name = f"{Path(uploaded_file.name).stem}_ocr_result.csv"
    ocr_xlsx_name = f"{Path(uploaded_file.name).stem}_ocr_result.xlsx"
    ocr_candidates_csv_name = f"{Path(uploaded_file.name).stem}_ocr_profile_candidates.csv"
    ocr_candidates_xlsx_name = f"{Path(uploaded_file.name).stem}_ocr_profile_candidates.xlsx"
    selected_ocr_candidates_csv_name = f"{Path(uploaded_file.name).stem}_selected_ocr_profile_candidates.csv"
    selected_ocr_candidates_xlsx_name = f"{Path(uploaded_file.name).stem}_selected_ocr_profile_candidates.xlsx"
    prototype_simple_csv_name = f"{Path(uploaded_file.name).stem}_prototype_simple_structured.csv"
    prototype_simple_xlsx_name = f"{Path(uploaded_file.name).stem}_prototype_simple_structured.xlsx"
    prototype_complex_csv_name = f"{Path(uploaded_file.name).stem}_prototype_complex_mapping.csv"
    prototype_complex_xlsx_name = f"{Path(uploaded_file.name).stem}_prototype_complex_mapping.xlsx"
    mapped_complex_csv_name = f"{Path(uploaded_file.name).stem}_mapped_complex_rows.csv"
    mapped_complex_xlsx_name = f"{Path(uploaded_file.name).stem}_mapped_complex_rows.xlsx"
    reviewed_mapped_complex_csv_name = f"{Path(uploaded_file.name).stem}_reviewed_mapped_complex_rows.csv"
    reviewed_mapped_complex_xlsx_name = f"{Path(uploaded_file.name).stem}_reviewed_mapped_complex_rows.xlsx"
    user_profile_csv_name = f"{Path(uploaded_file.name).stem}_user_profile_structured.csv"
    user_profile_xlsx_name = f"{Path(uploaded_file.name).stem}_user_profile_structured.xlsx"
    full_audit_csv_name = f"{Path(uploaded_file.name).stem}_full_audit_export.csv"
    full_audit_xlsx_name = f"{Path(uploaded_file.name).stem}_full_audit_export.xlsx"
    prototype_full_csv_name = f"{Path(uploaded_file.name).stem}_prototype_full.csv"
    prototype_full_xlsx_name = f"{Path(uploaded_file.name).stem}_prototype_full.xlsx"
    export_summary_df = pd.DataFrame(
        [
            {
                "export": "clean export",
                "description": "production structured rows из профильного parser'а",
            },
            {
                "export": "prototype simple export",
                "description": "простые строки prototype parser",
            },
            {
                "export": "mapped complex export",
                "description": "строки после ручной разметки сложных таблиц",
            },
            {
                "export": "reviewed mapped export",
                "description": "строки после human review с audit trail",
            },
            {
                "export": "user profile structured export",
                "description": "строки, построенные пользовательским source profile",
            },
            {
                "export": "full export with audit trail",
                "description": "объединение clean/user-profile/prototype/mapped rows с evidence fields",
            },
            {
                "export": "full prototype export",
                "description": "prototype rows + mapped rows; это не production clean export",
            },
        ]
    )
    st.caption("Какой файл скачивать")
    st.dataframe(export_summary_df, use_container_width=True, hide_index=True)

    user_profile_export_df = select_user_profile_export_columns(user_profile_structured_df)
    full_audit_frames = []
    for frame in [
        reviewed_df,
        user_profile_export_df,
        export_prototype_structured_df,
        export_reviewed_mapped_complex_df,
    ]:
        if frame is not None and not frame.empty:
            full_audit_frames.append(frame)
    if full_audit_frames:
        full_audit_export_df = pd.concat(full_audit_frames, ignore_index=True, sort=False)
        full_audit_export_df = full_audit_export_df.loc[
            ~full_audit_export_df.astype(str).duplicated()
        ].copy()
    else:
        full_audit_export_df = pd.DataFrame()

    if reviewed_df.empty:
        st.info(
            "Структурированные данные отсутствуют. "
            "Используйте raw export для настройки нового профиля источника."
        )
    else:
        clean_export_cols = st.columns(2)
        clean_export_cols[0].download_button(
            "Скачать чистую Excel-таблицу",
            data=export_to_excel_cached(reviewed_df),
            file_name=output_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        clean_export_cols[1].download_button(
            "Скачать чистый CSV",
            data=export_to_csv_cached(reviewed_df),
            file_name=csv_output_name,
            mime="text/csv",
        )

    if not user_profile_export_df.empty:
        st.caption("User profile structured export: строки, построенные пользовательским source profile.")
        user_profile_export_cols = st.columns(2)
        user_profile_export_cols[0].download_button(
            "Скачать user profile structured CSV",
            data=export_to_csv_cached(user_profile_export_df),
            file_name=user_profile_csv_name,
            mime="text/csv",
        )
        user_profile_export_cols[1].download_button(
            "Скачать user profile structured XLSX",
            data=export_to_excel_cached(user_profile_export_df),
            file_name=user_profile_xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not full_audit_export_df.empty:
        st.caption("Full export with audit trail: объединяет доступные result rows и сохраняет evidence fields.")
        full_audit_export_cols = st.columns(2)
        full_audit_export_cols[0].download_button(
            "Скачать full export with audit trail CSV",
            data=export_to_csv_cached(full_audit_export_df),
            file_name=full_audit_csv_name,
            mime="text/csv",
        )
        full_audit_export_cols[1].download_button(
            "Скачать full export with audit trail XLSX",
            data=export_to_excel_cached(full_audit_export_df),
            file_name=full_audit_xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not raw_rows.empty:
        st.caption("Raw extraction: пользовательская версия и техническая версия для настройки профиля")
        raw_export_cols = st.columns(2)
        raw_export_cols[0].download_button(
            "Скачать сырые фрагменты",
            data=export_to_csv_cached(select_raw_export_columns(raw_rows)),
            file_name=raw_output_name,
            mime="text/csv",
        )
        raw_export_cols[1].download_button(
            "Скачать технический raw export",
            data=export_to_csv_cached(select_technical_raw_export_columns(raw_rows)),
            file_name=technical_raw_output_name,
            mime="text/csv",
        )

    if not ocr_result_df.empty:
        st.caption("OCR-результат: raw OCR text для дальнейшей настройки профиля")
        ocr_export = select_ocr_export_columns(ocr_result_df)
        ocr_export_cols = st.columns(2)
        ocr_export_cols[0].download_button(
            "Скачать OCR-результат (CSV)",
            data=export_to_csv_cached(ocr_export),
            file_name=ocr_csv_name,
            mime="text/csv",
        )
        ocr_export_cols[1].download_button(
            "Скачать OCR-результат (XLSX)",
            data=export_to_excel_cached(ocr_export),
            file_name=ocr_xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if export_prototype_structured_df is not None and not export_prototype_structured_df.empty:
        st.caption("Prototype export: простые structured rows, complex mapping rows и полный export")
        prototype_simple_export_df, prototype_complex_export_df = split_prototype_rows(export_prototype_structured_df)
        prototype_full_export_df = export_prototype_structured_df
        if export_mapped_complex_df is not None and not export_mapped_complex_df.empty:
            prototype_full_export_df = pd.concat(
                [export_prototype_structured_df, export_mapped_complex_df],
                ignore_index=True,
                sort=False,
            )
        if not prototype_simple_export_df.empty:
            prototype_simple_export_cols = st.columns(2)
            prototype_simple_export_cols[0].download_button(
                "Скачать simple structured CSV",
                data=export_to_csv_cached(prototype_simple_export_df),
                file_name=prototype_simple_csv_name,
                mime="text/csv",
            )
            prototype_simple_export_cols[1].download_button(
                "Скачать simple structured XLSX",
                data=export_to_excel_cached(prototype_simple_export_df),
                file_name=prototype_simple_xlsx_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        if not prototype_complex_export_df.empty:
            prototype_complex_export_cols = st.columns(2)
            prototype_complex_export_cols[0].download_button(
                "Скачать complex mapping CSV",
                data=export_to_csv_cached(prototype_complex_export_df),
                file_name=prototype_complex_csv_name,
                mime="text/csv",
            )
            prototype_complex_export_cols[1].download_button(
                "Скачать complex mapping XLSX",
                data=export_to_excel_cached(prototype_complex_export_df),
                file_name=prototype_complex_xlsx_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        if export_mapped_complex_df is not None and not export_mapped_complex_df.empty:
            st.caption("Mapped rows до ручной проверки: исходный результат complex mapping без human review.")
            mapped_complex_export_cols = st.columns(2)
            mapped_complex_export_cols[0].download_button(
                "Скачать mapped rows до ручной проверки",
                data=export_to_csv_cached(export_mapped_complex_df),
                file_name=mapped_complex_csv_name,
                mime="text/csv",
            )
            mapped_complex_export_cols[1].download_button(
                "Скачать mapped rows до ручной проверки XLSX",
                data=export_to_excel_cached(export_mapped_complex_df),
                file_name=mapped_complex_xlsx_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.caption(
                "Reviewed mapped rows: результат prototype parser + human review; "
                "это отдельная demo/prototype выгрузка, не production clean rows."
            )
            if st.session_state.get(mapped_review_unsaved_key):
                st.warning(
                    "Есть несохранённые правки. Нажмите 'Применить ручную проверку' перед экспортом."
                )
            reviewed_mapped_complex_export_cols = st.columns(2)
            reviewed_mapped_complex_export_cols[0].download_button(
                "Скачать reviewed mapped rows CSV",
                data=export_to_csv_cached(export_reviewed_mapped_complex_df),
                file_name=reviewed_mapped_complex_csv_name,
                mime="text/csv",
            )
            reviewed_mapped_complex_export_cols[1].download_button(
                "Скачать reviewed mapped rows XLSX",
                data=export_to_excel_cached(export_reviewed_mapped_complex_df),
                file_name=reviewed_mapped_complex_xlsx_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        prototype_full_export_cols = st.columns(2)
        prototype_full_export_cols[0].download_button(
            "Скачать полный prototype export CSV",
            data=export_to_csv_cached(prototype_full_export_df),
            file_name=prototype_full_csv_name,
            mime="text/csv",
        )
        prototype_full_export_cols[1].download_button(
            "Скачать полный prototype export XLSX",
            data=export_to_excel_cached(prototype_full_export_df),
            file_name=prototype_full_xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not selected_export_ocr_candidates_df.empty:
        st.caption("Выбранные OCR-кандидаты для нового профиля источника")
        selected_ocr_candidate_export = select_ocr_candidate_export_columns(selected_export_ocr_candidates_df)
        selected_ocr_candidate_export_cols = st.columns(2)
        selected_ocr_candidate_export_cols[0].download_button(
            "Скачать выбранные кандидаты для профиля (CSV)",
            data=export_to_csv_cached(selected_ocr_candidate_export),
            file_name=selected_ocr_candidates_csv_name,
            mime="text/csv",
        )
        selected_ocr_candidate_export_cols[1].download_button(
            "Скачать выбранные кандидаты для профиля (XLSX)",
            data=export_to_excel_cached(selected_ocr_candidate_export),
            file_name=selected_ocr_candidates_xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not export_ocr_candidates_df.empty:
        st.caption(
            "OCR-кандидаты для профиля: это не финальные показатели, "
            "а кандидаты для настройки нового профиля источника."
        )
        ocr_candidate_export = select_ocr_candidate_export_columns(export_ocr_candidates_df)
        ocr_candidate_export_cols = st.columns(2)
        ocr_candidate_export_cols[0].download_button(
            "Скачать OCR-кандидаты для профиля (CSV)",
            data=export_to_csv_cached(ocr_candidate_export),
            file_name=ocr_candidates_csv_name,
            mime="text/csv",
        )
        ocr_candidate_export_cols[1].download_button(
            "Скачать OCR-кандидаты для профиля (XLSX)",
            data=export_to_excel_cached(ocr_candidate_export),
            file_name=ocr_candidates_xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not profile_candidates_df.empty:
        st.caption("Кандидаты для профиля: материал для аналитика и разработчика нового parser'а")
        candidate_export = select_profile_candidate_export_columns(profile_candidates_df)
        candidate_export_cols = st.columns(2)
        candidate_export_cols[0].download_button(
            "Скачать кандидаты для профиля (CSV)",
            data=export_to_csv_cached(candidate_export),
            file_name=profile_candidates_csv_name,
            mime="text/csv",
        )
        candidate_export_cols[1].download_button(
            "Скачать кандидаты для профиля (XLSX)",
            data=export_to_excel_cached(candidate_export),
            file_name=profile_candidates_xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    record_performance_timing(
        performance_timings,
        "exports preparation",
        export_start_time,
        cache_status="cached-by-content",
    )
    st.session_state[performance_timings_key] = performance_timings
    with st.expander("Debug performance timings"):
        st.dataframe(pd.DataFrame(performance_timings), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
