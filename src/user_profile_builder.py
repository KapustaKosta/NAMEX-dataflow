from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_standard_columns, is_missing, safe_file_name


USER_PROFILE_EXTRA_COLUMNS = [
    "profile_name",
    "source_row_id",
    "code",
    "name",
    "operation",
    "metric",
    "scenario",
    "tariff_type",
    "category",
    "validation_status",
    "warnings",
    "review_status",
]

USER_PROFILE_EXPORT_COLUMNS = [
    "source_file",
    "source_type",
    "source_kind",
    "page",
    "table_id",
    "row_index_in_table",
    "source_row_id",
    "section_name",
    "profile_name",
    "code",
    "name",
    "commodity",
    "operation",
    "metric",
    "value",
    "unit",
    "currency",
    "scenario",
    "tariff_type",
    "category",
    "evidence_text",
    "extraction_method",
    "extraction_level",
    "confidence",
    "validation_status",
    "warnings",
    "review_status",
]

COLUMN_ROLES = [
    "ignore",
    "row_number",
    "code",
    "name",
    "unit",
    "value",
    "value_direct",
    "value_intraport",
    "currency",
    "date",
    "year",
    "percent",
    "category",
    "region",
    "country",
    "custom_text",
    "custom_numeric",
]

VALUE_ROLES = {"value", "value_direct", "value_intraport", "percent", "custom_numeric"}
EMPTY_TOKENS = {"", "-", "—", "–", "n/a", "na", "нет", "none", "null"}

SOURCE_KIND_ALIASES = {
    "pdf_table": "raw_table",
    "pdfplumber_table": "raw_table",
    "pdf_text_layer": "raw_table",
    "raw_table": "raw_table",
    "ocr": "ocr_candidate",
    "ocr_candidate": "ocr_candidate",
    "tesseract_ocr": "ocr_candidate",
}

PUBLIC_SOURCE_KIND = {
    "raw_table": "pdf_table",
    "ocr_candidate": "ocr_candidate",
}

NUMBER_TOKEN_PATTERN = re.compile(r"(?<!\w)[+-]?\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?(?:\s*%)?(?!\w)")

UNIT_ALIASES = {
    "т": "ton",
    "тонна": "ton",
    "тонн": "ton",
    "тонны": "ton",
    "ton": "ton",
    "tons": "ton",
    "тыс. тонн": "thousand_tons",
    "тыс тонн": "thousand_tons",
    "тыс.тонн": "thousand_tons",
    "thousand_tons": "thousand_tons",
    "млн долл. сша": "million_usd",
    "млн долл сша": "million_usd",
    "млн долл. США": "million_usd",
    "million_usd": "million_usd",
    "%": "percent",
    "процент": "percent",
    "проценты": "percent",
    "руб/т": "RUB/ton",
    "руб. / т": "RUB/ton",
    "рублей/т": "RUB/ton",
}

CURRENCY_ALIASES = {
    "рублях рф": "RUB",
    "рубли": "RUB",
    "руб": "RUB",
    "руб.": "RUB",
    "rur": "RUB",
    "rub": "RUB",
    "долл. сша": "USD",
    "долл сша": "USD",
    "usd": "USD",
    "$": "USD",
    "eur": "EUR",
    "евро": "EUR",
}


def normalize_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _normalize_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("ё", "е").casefold()).strip()


def is_empty_profile_value(value: Any) -> bool:
    if is_missing(value):
        return True
    return _normalize_key(value) in EMPTY_TOKENS


def normalize_user_number(value: Any, *, value_type: str = "numeric") -> float | None:
    """Normalize Russian-style numbers, percentages and common empty tokens."""
    if is_empty_profile_value(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip().replace("\u00a0", " ")
    if value_type == "percent":
        text = text.replace("%", "")
    text = text.replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    text = re.sub(r"[^0-9.+-]", "", text)
    if text in {"", "+", "-", ".", "+.", "-."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_user_unit(value: Any) -> str | None:
    if is_empty_profile_value(value):
        return None
    text = str(value).strip()
    return UNIT_ALIASES.get(_normalize_key(text)) or text


def normalize_user_currency(value: Any) -> str | None:
    if is_empty_profile_value(value):
        return None
    text = str(value).strip()
    return CURRENCY_ALIASES.get(_normalize_key(text)) or text.upper()


def split_profile_row_cells(evidence_text: Any) -> list[str]:
    """Split a raw table row into cells using pipe/tab/multiple-space separators."""
    text = str(evidence_text or "").strip()
    if not text:
        return []
    if "|" in text:
        return [cell.strip() for cell in text.split("|")]
    if "\t" in text:
        return [cell.strip() for cell in text.split("\t")]
    return [cell.strip() for cell in re.split(r"\s{2,}", text) if cell.strip()]


def normalize_source_kind(value: Any) -> str | None:
    source_kind = normalize_text(value)
    if not source_kind:
        return None
    return SOURCE_KIND_ALIASES.get(source_kind, source_kind)


def public_source_kind(value: Any) -> str:
    return PUBLIC_SOURCE_KIND.get(normalize_source_kind(value) or "", str(value or "source"))


def stable_text_id(*parts: Any, length: int = 12) -> str:
    payload = "|".join(str(part or "") for part in parts)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:length]


def source_block_uid_from_parts(source_kind: Any, page: Any, block_id: Any, title: Any = "", preview: Any = "") -> str:
    public_kind = public_source_kind(source_kind)
    page_text = "na" if is_missing(page) else str(page)
    block_text = normalize_text(block_id)
    if not block_text:
        block_text = stable_text_id(public_kind, page_text, normalize_text(title), str(preview or "")[:160])
    return f"{public_kind}:{page_text}:{block_text}"


def source_block_uid(source_row: dict[str, Any]) -> str:
    return source_block_uid_from_parts(
        source_row.get("source_kind"),
        source_row.get("page"),
        source_row.get("table_id"),
        source_row.get("block_title"),
        source_row.get("evidence_text"),
    )


def source_row_uid(source_row: dict[str, Any]) -> str:
    """Return a stable row id independent from UI display order."""
    block_uid = normalize_text(source_row.get("block_uid")) or source_block_uid(source_row)
    row_index = source_row.get("row_index_in_table")
    if not is_missing(row_index):
        try:
            row_index_text = str(int(row_index))
        except (TypeError, ValueError):
            row_index_text = str(row_index)
        return f"{block_uid}:row:{row_index_text}"
    source_row_id = normalize_text(source_row.get("source_row_id"))
    if source_row_id:
        return f"{block_uid}:row:{source_row_id}"
    evidence_hash = hashlib.md5(str(source_row.get("evidence_text") or "").encode("utf-8")).hexdigest()[:12]
    return f"{block_uid}:row:{evidence_hash}"


def extract_numeric_tokens(text: Any) -> list[str]:
    return [match.group(0).strip() for match in NUMBER_TOKEN_PATTERN.finditer(str(text or ""))]


def text_without_numeric_tokens(text: Any) -> str:
    return re.sub(r"\s+", " ", NUMBER_TOKEN_PATTERN.sub(" ", str(text or ""))).strip()


def _source_rows_from_raw(raw_rows: pd.DataFrame | None) -> list[dict[str, Any]]:
    if raw_rows is None or raw_rows.empty:
        return []
    rows = raw_rows.copy()
    sort_columns = [column for column in ["page", "table_id", "row_index_in_table", "row_id"] if column in rows.columns]
    if sort_columns:
        rows = rows.sort_values(sort_columns)
    source_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        evidence_text = str(row.get("evidence_text") or "")
        cells = split_profile_row_cells(evidence_text)
        if not cells:
            continue
        table_id = row.get("table_id")
        source_row = {
            "source_kind": "raw_table",
            "source_file": row.get("source_file"),
            "page": row.get("page"),
            "table_id": "" if is_missing(table_id) else str(table_id),
            "block_title": row.get("block_title") or row.get("section_title") or row.get("section_name"),
            "row_index_in_table": row.get("row_index_in_table"),
            "source_row_id": row.get("row_id"),
            "extraction_method": row.get("extraction_method") or "pdfplumber_table",
            "text_layer_quality": row.get("text_layer_quality"),
            "evidence_text": evidence_text,
            "cells": cells,
            "numeric_tokens": extract_numeric_tokens(evidence_text),
            "text_part": text_without_numeric_tokens(evidence_text),
        }
        source_row["block_uid"] = source_block_uid(source_row)
        source_row["row_uid"] = source_row_uid(source_row)
        source_rows.append(source_row)
    return source_rows


def _source_rows_from_ocr_candidates(ocr_candidates_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if ocr_candidates_df is None or ocr_candidates_df.empty:
        return []
    source_rows: list[dict[str, Any]] = []
    for _, candidate in ocr_candidates_df.iterrows():
        block_text = str(candidate.get("block_text") or candidate.get("preview") or "")
        block_id = str(
            candidate.get("ocr_block_id")
            or stable_text_id("ocr_candidate", candidate.get("page"), candidate.get("block_title"), block_text[:160])
        )
        for row_index, line in enumerate(block_text.splitlines(), start=1):
            cells = split_profile_row_cells(line)
            if not cells:
                continue
            source_row = {
                "source_kind": "ocr_candidate",
                "source_file": candidate.get("source_file"),
                "page": candidate.get("page"),
                "table_id": block_id,
                "block_title": candidate.get("block_title"),
                "row_index_in_table": row_index,
                "source_row_id": f"{block_id}:{row_index}",
                "extraction_method": candidate.get("extraction_method") or "ocr_candidate",
                "text_layer_quality": "ocr",
                "evidence_text": line,
                "cells": cells,
                "numeric_tokens": extract_numeric_tokens(line),
                "text_part": text_without_numeric_tokens(line),
            }
            source_row["block_uid"] = source_block_uid(source_row)
            source_row["row_uid"] = source_row_uid(source_row)
            source_rows.append(source_row)
    return source_rows


def source_rows_from_frames(
    raw_rows: pd.DataFrame | None,
    ocr_candidates_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    return _source_rows_from_raw(raw_rows) + _source_rows_from_ocr_candidates(ocr_candidates_df)


def _selector_matches(source_row: dict[str, Any], selector: dict[str, Any]) -> bool:
    table_ids = {str(value) for value in selector.get("table_ids") or [] if str(value)}
    if table_ids and str(source_row.get("table_id") or "") not in table_ids:
        return False
    block_uids = {str(value) for value in selector.get("block_uids") or [] if str(value)}
    if block_uids and str(source_row.get("block_uid") or source_block_uid(source_row)) not in block_uids:
        return False
    page_contains = normalize_text(selector.get("page_contains"))
    if page_contains and page_contains not in str(source_row.get("page") or ""):
        return False
    text_contains_values = selector.get("text_contains")
    if isinstance(text_contains_values, str) or not isinstance(text_contains_values, list):
        text_contains_values = [text_contains_values]
    text_contains_values = [normalize_text(value) for value in text_contains_values]
    text_contains_values = [value for value in text_contains_values if value]
    if text_contains_values:
        evidence_text = str(source_row.get("evidence_text") or "").casefold()
        if not all(str(value).casefold() in evidence_text for value in text_contains_values):
            return False
    source_kind = normalize_source_kind(selector.get("source_kind"))
    if source_kind and source_kind != normalize_source_kind(source_row.get("source_kind")):
        return False
    return True


def select_source_rows(
    raw_rows: pd.DataFrame | None,
    ocr_candidates_df: pd.DataFrame | None,
    table_selector: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selector = table_selector or {}
    source_rows = source_rows_from_frames(raw_rows, ocr_candidates_df)
    table_ids = {str(value) for value in selector.get("table_ids") or [] if str(value)}
    block_uids = {str(value) for value in selector.get("block_uids") or [] if str(value)}
    source_kind = normalize_source_kind(selector.get("source_kind"))
    page_contains = normalize_text(selector.get("page_contains"))
    text_contains_values = selector.get("text_contains")
    if isinstance(text_contains_values, str) or not isinstance(text_contains_values, list):
        text_contains_values = [text_contains_values]
    text_contains_values = [normalize_text(value) for value in text_contains_values]
    text_contains_values = [value for value in text_contains_values if value]
    if table_ids or block_uids or source_kind or page_contains:
        source_rows = [row for row in source_rows if _selector_matches(row, {**selector, "text_contains": None})]
    if text_contains_values:
        table_text: dict[tuple[str, str], list[str]] = {}
        for row in source_rows:
            key = (str(row.get("source_kind") or ""), str(row.get("table_id") or ""))
            table_text.setdefault(key, []).append(str(row.get("evidence_text") or ""))
        matching_table_keys = {
            key
            for key, evidence_parts in table_text.items()
            if all(str(text).casefold() in "\n".join(evidence_parts).casefold() for text in text_contains_values)
        }
        source_rows = [
            row
            for row in source_rows
            if (str(row.get("source_kind") or ""), str(row.get("table_id") or "")) in matching_table_keys
        ]
    return source_rows


def select_source_rows_for_block_uids(
    raw_rows: pd.DataFrame | None,
    ocr_candidates_df: pd.DataFrame | None,
    block_uids: list[str] | tuple[str, ...] | set[str] | None,
) -> list[dict[str, Any]]:
    """Return rows for an explicit block selection; an empty selection means no rows."""
    selected_block_uids = [str(value) for value in (block_uids or []) if str(value)]
    if not selected_block_uids:
        return []
    return select_source_rows(raw_rows, ocr_candidates_df, {"block_uids": selected_block_uids})


def build_profile_table_catalog(
    raw_rows: pd.DataFrame | None,
    raw_table_summary_df: pd.DataFrame | None = None,
    ocr_candidates_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return user-selectable raw/OCR tables for the UI builder."""
    rows: list[dict[str, Any]] = []
    if raw_table_summary_df is not None and not raw_table_summary_df.empty:
        for _, table in raw_table_summary_df.iterrows():
            table_id = str(table.get("table_id") or "")
            block_uid = source_block_uid_from_parts("raw_table", table.get("page"), table_id, "", table.get("preview"))
            rows.append(
                {
                    "table_key": block_uid,
                    "block_uid": block_uid,
                    "source_kind": "raw_table",
                    "page": table.get("page"),
                    "table_id": table_id,
                    "block_title": table.get("block_title") or table_id,
                    "rows_count": table.get("raw_rows_count"),
                    "columns_count": table.get("column_count"),
                    "numbers_count": table.get("numbers_count"),
                    "preview": table.get("preview"),
                    "table_score": table.get("table_score"),
                    "text_layer_quality": table.get("text_layer_quality"),
                    "extraction_method": "pdfplumber_table",
                }
            )
    elif raw_rows is not None and not raw_rows.empty:
        for table_id, group in raw_rows.dropna(subset=["table_id"]).groupby("table_id", sort=False):
            source_rows = _source_rows_from_raw(group)
            page = group["page"].dropna().iloc[0] if "page" in group and group["page"].notna().any() else None
            preview = "\n".join(str(row["evidence_text"]) for row in source_rows[:3])
            block_uid = source_block_uid_from_parts("raw_table", page, table_id, "", preview)
            rows.append(
                {
                    "table_key": block_uid,
                    "block_uid": block_uid,
                    "source_kind": "raw_table",
                    "page": page,
                    "table_id": str(table_id),
                    "block_title": str(table_id),
                    "rows_count": len(source_rows),
                    "columns_count": max((len(row["cells"]) for row in source_rows), default=0),
                    "numbers_count": sum(len(row.get("numeric_tokens") or []) for row in source_rows),
                    "preview": preview,
                    "table_score": group["table_score"].dropna().iloc[0] if "table_score" in group and group["table_score"].notna().any() else None,
                    "text_layer_quality": group["text_layer_quality"].dropna().iloc[0] if "text_layer_quality" in group and group["text_layer_quality"].notna().any() else None,
                    "extraction_method": "pdfplumber_table",
                }
            )

    if ocr_candidates_df is not None and not ocr_candidates_df.empty:
        for _, candidate in ocr_candidates_df.iterrows():
            preview = candidate.get("preview") or candidate.get("block_text")
            block_id = str(
                candidate.get("ocr_block_id")
                or stable_text_id("ocr_candidate", candidate.get("page"), candidate.get("block_title"), str(preview or "")[:160])
            )
            source_rows = _source_rows_from_ocr_candidates(pd.DataFrame([candidate]))
            block_uid = source_block_uid_from_parts(
                "ocr_candidate",
                candidate.get("page"),
                block_id,
                candidate.get("block_title"),
                preview,
            )
            rows.append(
                {
                    "table_key": block_uid,
                    "block_uid": block_uid,
                    "source_kind": "ocr_candidate",
                    "page": candidate.get("page"),
                    "table_id": block_id,
                    "block_title": candidate.get("block_title"),
                    "rows_count": len(source_rows),
                    "columns_count": max((len(row["cells"]) for row in source_rows), default=0),
                    "numbers_count": candidate.get("numbers_count") or sum(len(row.get("numeric_tokens") or []) for row in source_rows),
                    "preview": preview,
                    "table_score": candidate.get("table_score"),
                    "text_layer_quality": "ocr",
                    "extraction_method": candidate.get("extraction_method") or "ocr_candidate",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "table_key",
            "block_uid",
            "source_kind",
            "page",
            "table_id",
            "block_title",
            "rows_count",
            "columns_count",
            "numbers_count",
            "preview",
            "table_score",
            "text_layer_quality",
            "extraction_method",
        ],
    )


def source_rows_to_preview_df(source_rows: list[dict[str, Any]], limit: int = 50) -> pd.DataFrame:
    max_columns = max((len(row.get("cells") or []) for row in source_rows[:limit]), default=0)
    rows = []
    for source_row in source_rows[:limit]:
        row = {
            "row_uid": source_row.get("row_uid") or source_row_uid(source_row),
            "block_uid": source_row.get("block_uid") or source_block_uid(source_row),
            "source_kind": public_source_kind(source_row.get("source_kind")),
            "page": source_row.get("page"),
            "block_title": source_row.get("block_title"),
            "table_id": source_row.get("table_id"),
            "row_index_in_table": source_row.get("row_index_in_table"),
            "text_part": source_row.get("text_part"),
            "numeric_tokens": " | ".join(str(token) for token in source_row.get("numeric_tokens") or []),
            "evidence_text": source_row.get("evidence_text"),
        }
        cells = source_row.get("cells") or []
        for index in range(max_columns):
            row[f"column_{index + 1}"] = cells[index] if index < len(cells) else ""
        rows.append(row)
    base_columns = [
        "row_uid",
        "block_uid",
        "source_kind",
        "page",
        "block_title",
        "table_id",
        "row_index_in_table",
        "text_part",
        "numeric_tokens",
        "evidence_text",
    ]
    column_names = base_columns + [f"column_{index + 1}" for index in range(max_columns)]
    return pd.DataFrame(rows, columns=column_names)


def apply_table_reconstruction(
    source_rows: list[dict[str, Any]],
    table_reconstruction: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return source rows with optionally rebuilt cells from evidence_text."""
    reconstruction = table_reconstruction or {}
    method = str(reconstruction.get("method") or "none").strip()
    if method in {"", "none"}:
        return list(source_rows)

    if method != "split_by_regex":
        return list(source_rows)

    pattern = str(reconstruction.get("pattern") or "").strip()
    if not pattern:
        return list(source_rows)

    rebuilt_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        evidence_text = str(source_row.get("evidence_text") or "")
        try:
            cells = [cell.strip() for cell in re.split(pattern, evidence_text) if cell.strip()]
        except re.error:
            cells = []
        if not cells:
            cells = list(source_row.get("cells") or [])

        rebuilt = dict(source_row)
        rebuilt["cells"] = cells
        rebuilt["numeric_tokens"] = extract_numeric_tokens(evidence_text)
        rebuilt["text_part"] = text_without_numeric_tokens(evidence_text)
        rebuilt["table_reconstruction_method"] = "split_by_regex"
        rebuilt["table_reconstruction_pattern"] = pattern
        rebuilt_rows.append(rebuilt)
    return rebuilt_rows


def _source_row_selection_keys(source_row: dict[str, Any]) -> set[str]:
    keys: set[str] = {source_row_uid(source_row)}
    if source_row.get("row_uid"):
        keys.add(str(source_row.get("row_uid")))
    for column in ["source_row_id", "row_index_in_table"]:
        value = source_row.get(column)
        if is_missing(value):
            continue
        keys.add(str(value))
        try:
            keys.add(str(int(value)))
        except (TypeError, ValueError):
            pass

    table_id = source_row.get("table_id")
    row_index = source_row.get("row_index_in_table")
    if not is_missing(table_id) and not is_missing(row_index):
        keys.add(f"{table_id}:{row_index}")
        keys.add(f"{normalize_source_kind(source_row.get('source_kind'))}:{table_id}:{row_index}")
        keys.add(f"{public_source_kind(source_row.get('source_kind'))}:{table_id}:{row_index}")
        try:
            keys.add(f"{table_id}:{int(row_index)}")
            keys.add(f"{normalize_source_kind(source_row.get('source_kind'))}:{table_id}:{int(row_index)}")
            keys.add(f"{public_source_kind(source_row.get('source_kind'))}:{table_id}:{int(row_index)}")
        except (TypeError, ValueError):
            pass
    return keys


def normalize_row_filters(row_filters: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    """Accept both old list filters and readable config-style row_filters mappings."""
    if not row_filters:
        return []
    if isinstance(row_filters, list):
        return row_filters
    if not isinstance(row_filters, dict):
        return []

    filters: list[dict[str, Any]] = []
    selected_rows = row_filters.get("selected_row_uids") or row_filters.get("selected_source_rows")
    if (row_filters.get("use_manual_rows") or row_filters.get("mode") == "manual") and selected_rows:
        filters.append(
            {
                "type": "manual_selected_rows",
                "selected_source_rows": selected_rows or [],
            }
        )

    keep_after_text = row_filters.get("keep_after") or row_filters.get("start_after_contains")
    if keep_after_text:
        filters.append({"type": "keep_after", "text": keep_after_text})

    keep_until_text = row_filters.get("keep_until") or row_filters.get("stop_before_contains")
    if keep_until_text:
        filters.append({"type": "keep_until", "text": keep_until_text})

    keep_contains_text = row_filters.get("keep_text_contains")
    if keep_contains_text:
        filters.append({"type": "keep_text_contains", "text": keep_contains_text})

    if row_filters.get("keep_numeric_rows_only"):
        filters.append({"type": "keep_numeric_rows_only"})
    if row_filters.get("skip_empty_code"):
        filters.append({"type": "skip_empty_code"})
    if row_filters.get("skip_empty_values") or row_filters.get("skip_empty_value_columns"):
        filters.append({"type": "skip_empty_value_columns"})
    if row_filters.get("skip_dash_values"):
        filters.append({"type": "skip_dash_values"})
    return filters


def apply_row_filters(
    source_rows: list[dict[str, Any]],
    row_filters: list[dict[str, Any]] | dict[str, Any] | None,
    column_mapping: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    filters = normalize_row_filters(row_filters)
    result = list(source_rows)
    for row_filter in filters:
        filter_type = str(row_filter.get("type") or "")
        text = str(row_filter.get("text") or "")
        if filter_type == "manual_selected_rows":
            selected_keys = {
                str(value)
                for value in row_filter.get("selected_source_rows") or row_filter.get("source_rows") or []
                if str(value).strip()
            }
            if selected_keys:
                result = [
                    row
                    for row in result
                    if _source_row_selection_keys(row) & selected_keys
                ]
        elif filter_type == "keep_after":
            kept: list[dict[str, Any]] = []
            active = False
            for row in result:
                if not active and text and text.casefold() in str(row.get("evidence_text") or "").casefold():
                    active = True
                    continue
                if active:
                    kept.append(row)
            result = kept
        elif filter_type in {"keep_until", "until"}:
            kept = []
            for row in result:
                if text and text.casefold() in str(row.get("evidence_text") or "").casefold():
                    break
                kept.append(row)
            result = kept
        elif filter_type == "keep_text_contains":
            result = [
                row
                for row in result
                if text.casefold() in str(row.get("evidence_text") or "").casefold()
            ]
        elif filter_type == "keep_numeric_rows_only":
            result = [
                row
                for row in result
                if any(normalize_user_number(cell) is not None for cell in row.get("cells") or [])
            ]
        elif filter_type == "skip_empty_code":
            code_indexes = _column_indexes_for_roles(column_mapping or {}, {"code"})
            result = [row for row in result if any(_cell_at(row, index) for index in code_indexes)]
        elif filter_type == "skip_empty_value_columns":
            value_indexes = _column_indexes_for_roles(column_mapping or {}, VALUE_ROLES)
            if value_indexes:
                result = [row for row in result if any(not is_empty_profile_value(_cell_at(row, index)) for index in value_indexes)]
            else:
                result = [
                    row
                    for row in result
                    if any(not is_empty_profile_value(token) for token in row.get("numeric_tokens") or row.get("cells") or [])
                ]
        elif filter_type == "skip_dash_values":
            value_indexes = _column_indexes_for_roles(column_mapping or {}, VALUE_ROLES)
            if value_indexes:
                result = [
                    row
                    for row in result
                    if any(_normalize_key(_cell_at(row, index)) not in {"-", "—", "–"} for index in value_indexes)
                ]
            else:
                result = [
                    row
                    for row in result
                    if any(_normalize_key(token) not in {"-", "—", "–"} for token in row.get("numeric_tokens") or row.get("cells") or [])
                ]
        elif filter_type == "stop_before_next_section":
            kept = []
            for row in result:
                evidence = str(row.get("evidence_text") or "").strip()
                if kept and _looks_like_section_title(evidence, row.get("cells") or []):
                    break
                kept.append(row)
            result = kept
    return result


def _looks_like_section_title(evidence: str, cells: list[str]) -> bool:
    if len(cells) <= 2 and re.match(r"^\d+\s+\D", evidence):
        return True
    numbers = [normalize_user_number(cell) for cell in cells]
    return bool(cells and re.match(r"^\d+\s+\D", evidence) and not any(value is not None for value in numbers[1:]))


def _column_indexes_for_roles(column_mapping: dict[str, dict[str, Any]], roles: set[str]) -> list[int]:
    indexes = []
    for column_key, config in column_mapping.items():
        if str(config.get("role") or "ignore") not in roles:
            continue
        match = re.search(r"(\d+)$", str(column_key))
        if match:
            indexes.append(int(match.group(1)) - 1)
    return indexes


def _cell_at(source_row: dict[str, Any], index: int) -> str:
    cells = source_row.get("cells") or []
    return str(cells[index]).strip() if 0 <= index < len(cells) else ""


def _first_role_value(
    source_row: dict[str, Any],
    column_mapping: dict[str, dict[str, Any]],
    roles: set[str],
) -> str | None:
    for index in _column_indexes_for_roles(column_mapping, roles):
        value = normalize_text(_cell_at(source_row, index))
        if value:
            return value
    return None


def _build_warning_status(warnings: list[str], value_missing: bool) -> tuple[str, str]:
    if value_missing:
        return "failed", "manual_required"
    if warnings:
        return "warning", "needs_review"
    return "passed", "auto_approved"


def _value_label_from_role(role: str) -> str | None:
    if role == "value_direct":
        return "direct"
    if role == "value_intraport":
        return "intraport_movement"
    return None


def _structured_rows_for_source_row(
    source_row: dict[str, Any],
    table_config: dict[str, Any],
    profile_config: dict[str, Any],
    row_id_start: int,
) -> list[dict[str, Any]]:
    column_mapping = table_config.get("column_mapping") or {}
    value_columns = [
        (column_key, config)
        for column_key, config in column_mapping.items()
        if str(config.get("role") or "ignore") in VALUE_ROLES
    ]
    if not value_columns:
        return []

    code = _first_role_value(source_row, column_mapping, {"code"})
    name = _first_role_value(source_row, column_mapping, {"name"}) or _first_role_value(
        source_row,
        column_mapping,
        {"custom_text"},
    )
    row_unit = _first_role_value(source_row, column_mapping, {"unit"})
    row_currency = _first_role_value(source_row, column_mapping, {"currency"})
    row_year = _first_role_value(source_row, column_mapping, {"year"})
    row_category = _first_role_value(source_row, column_mapping, {"category"})
    row_region = _first_role_value(source_row, column_mapping, {"region"})
    row_country = _first_role_value(source_row, column_mapping, {"country"})

    rows: list[dict[str, Any]] = []
    for offset, (column_key, value_config) in enumerate(value_columns):
        match = re.search(r"(\d+)$", str(column_key))
        if not match:
            continue
        column_index = int(match.group(1)) - 1
        raw_value = _cell_at(source_row, column_index)
        value_type = str(value_config.get("value_type") or "numeric")
        value = normalize_user_number(raw_value, value_type=value_type)
        if value is None:
            continue

        role = str(value_config.get("role") or "value")
        metric = normalize_text(value_config.get("metric")) or ("percent" if role == "percent" else "value")
        unit = normalize_user_unit(value_config.get("unit_override") or value_config.get("unit") or row_unit)
        currency = normalize_user_currency(value_config.get("currency_override") or value_config.get("currency") or row_currency)
        year = value_config.get("year") if not is_missing(value_config.get("year")) else row_year
        tariff_type = normalize_text(value_config.get("tariff_type")) or _value_label_from_role(role)
        scenario = normalize_text(value_config.get("scenario"))
        category = normalize_text(value_config.get("category_label")) or row_category
        warnings: list[str] = []
        if is_missing(name):
            warnings.append("name missing")
        if is_missing(unit) and "unit" in (profile_config.get("validation") or {}).get("required_fields", []):
            warnings.append("unit missing")
        if value <= 0 and bool((profile_config.get("validation") or {}).get("value_positive")):
            warnings.append("value <= 0")
        validation_status, review_status = _build_warning_status(warnings, value_missing=False)

        row = {
            "source_file": source_row.get("source_file"),
            "source_type": "pdf",
            "source_kind": source_row.get("source_kind"),
            "page": source_row.get("page"),
            "table_id": source_row.get("table_id"),
            "row_index_in_table": source_row.get("row_index_in_table"),
            "source_row_id": source_row.get("source_row_id"),
            "section_name": table_config.get("section_name") or profile_config.get("profile_name"),
            "profile_name": profile_config.get("profile_name"),
            "row_id": row_id_start + offset,
            "code": code,
            "name": name,
            "commodity": name,
            "operation": name,
            "indicator": metric,
            "metric": metric,
            "value": value,
            "unit": unit,
            "currency": currency,
            "date": None,
            "year": year,
            "region": row_region,
            "country": row_country,
            "scenario": scenario,
            "tariff_type": tariff_type,
            "category": category,
            "evidence_text": source_row.get("evidence_text"),
            "extraction_method": "user_profile_parser",
            "extraction_level": "structured",
            "text_layer_quality": source_row.get("text_layer_quality"),
            "confidence": 0.9 if validation_status == "passed" else 0.75,
            "validation_status": validation_status,
            "warnings": "; ".join(warnings),
            "review_status": review_status,
        }
        rows.append(row)
    return rows


def _token_mapping_rows_for_source_row(
    source_row: dict[str, Any],
    table_config: dict[str, Any],
    profile_config: dict[str, Any],
    row_id_start: int,
) -> list[dict[str, Any]]:
    token_mapping = table_config.get("token_mapping") or {}
    if not token_mapping:
        return []

    numeric_tokens = source_row.get("numeric_tokens") or extract_numeric_tokens(source_row.get("evidence_text"))
    text_part = normalize_text(source_row.get("text_part")) or text_without_numeric_tokens(source_row.get("evidence_text"))
    rows: list[dict[str, Any]] = []
    for offset, (token_key, token_config) in enumerate(token_mapping.items()):
        if not bool(token_config.get("enabled", True)):
            continue
        match = re.search(r"(\d+)$", str(token_key))
        if not match:
            continue
        token_index = int(match.group(1)) - 1
        if token_index >= len(numeric_tokens):
            continue
        raw_value = numeric_tokens[token_index]
        value_type = str(token_config.get("value_type") or "numeric")
        value = normalize_user_number(raw_value, value_type=value_type)
        if value is None:
            continue

        metric = normalize_text(token_config.get("metric")) or "value"
        unit = normalize_user_unit(token_config.get("unit") or token_config.get("unit_override"))
        currency = normalize_user_currency(token_config.get("currency") or token_config.get("currency_override"))
        year = token_config.get("year")
        scenario = normalize_text(token_config.get("scenario"))
        warnings: list[str] = []
        if is_missing(text_part):
            warnings.append("name missing")
        if value <= 0 and bool((profile_config.get("validation") or {}).get("value_positive")):
            warnings.append("value <= 0")
        validation_status, review_status = _build_warning_status(warnings, value_missing=False)
        rows.append(
            {
                "source_file": source_row.get("source_file"),
                "source_type": "pdf",
                "source_kind": source_row.get("source_kind"),
                "page": source_row.get("page"),
                "table_id": source_row.get("table_id"),
                "row_index_in_table": source_row.get("row_index_in_table"),
                "source_row_id": source_row.get("source_row_id"),
                "section_name": table_config.get("section_name") or profile_config.get("profile_name"),
                "profile_name": profile_config.get("profile_name"),
                "row_id": row_id_start + offset,
                "code": None,
                "name": text_part,
                "commodity": text_part,
                "operation": text_part,
                "indicator": metric,
                "metric": metric,
                "value": value,
                "unit": unit,
                "currency": currency,
                "date": None,
                "year": year,
                "region": None,
                "country": None,
                "scenario": scenario,
                "tariff_type": scenario,
                "category": normalize_text(token_config.get("category")),
                "evidence_text": source_row.get("evidence_text"),
                "extraction_method": "user_profile_parser",
                "extraction_level": "structured",
                "text_layer_quality": source_row.get("text_layer_quality"),
                "confidence": 0.9 if validation_status == "passed" else 0.75,
                "validation_status": validation_status,
                "warnings": "; ".join(warnings),
                "review_status": review_status,
            }
        )
    return rows


def _block_selector_to_table_selector(block_config: dict[str, Any]) -> dict[str, Any]:
    selector = dict(block_config.get("table_selector") or block_config.get("selector") or {})
    if block_config.get("source_kind") and not selector.get("source_kind"):
        selector["source_kind"] = block_config.get("source_kind")
    if selector.get("source_kind"):
        selector["source_kind"] = normalize_source_kind(selector.get("source_kind"))
    if selector.get("block_ids") and not selector.get("table_ids"):
        selector["table_ids"] = selector.get("block_ids")
    return selector


def profile_table_configs(profile_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return old/new profile block configs in the table_config shape used by the parser."""
    table_configs: list[dict[str, Any]] = []
    profile_blocks = profile_config.get("blocks") or []
    if not profile_blocks:
        legacy_tables = profile_config.get("tables") or []
        for table_config in legacy_tables:
            if isinstance(table_config, dict):
                table_configs.append(table_config)
        return table_configs

    for block_config in profile_blocks:
        if not isinstance(block_config, dict):
            continue
        row_selection = block_config.get("row_selection")
        row_filters = block_config.get("row_filters")
        table_configs.append(
            {
                "section_name": block_config.get("section_name") or profile_config.get("profile_name"),
                "table_selector": _block_selector_to_table_selector(block_config),
                "row_filters": row_selection if row_selection is not None else row_filters,
                "table_reconstruction": block_config.get("table_reconstruction"),
                "column_mapping": block_config.get("column_mapping") or {},
                "token_mapping": block_config.get("token_mapping") or {},
            }
        )
    return table_configs


def _profile_config_for_application(profile_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(profile_config)
    normalized["tables"] = profile_table_configs(profile_config)
    return normalized


def apply_user_profile_to_sources(
    raw_rows: pd.DataFrame | None,
    ocr_candidates_df: pd.DataFrame | None,
    profile_config: dict[str, Any],
) -> pd.DataFrame:
    """Apply a saved user source profile to raw/OCR table rows."""
    output_rows: list[dict[str, Any]] = []
    row_id = 1
    application_config = _profile_config_for_application(profile_config)
    for table_config in application_config.get("tables") or []:
        table_selector = table_config.get("table_selector") or {}
        column_mapping = table_config.get("column_mapping") or {}
        source_rows = select_source_rows(raw_rows, ocr_candidates_df, table_selector)
        source_rows = apply_table_reconstruction(source_rows, table_config.get("table_reconstruction"))
        source_rows = apply_row_filters(source_rows, table_config.get("row_filters") or [], column_mapping)
        for source_row in source_rows:
            rows = _token_mapping_rows_for_source_row(source_row, table_config, application_config, row_id)
            if not rows:
                rows = _structured_rows_for_source_row(source_row, table_config, application_config, row_id)
            output_rows.extend(rows)
            row_id += len(rows)

    if not output_rows:
        return pd.DataFrame(columns=USER_PROFILE_EXPORT_COLUMNS)
    result = ensure_standard_columns(pd.DataFrame(output_rows))
    for column in USER_PROFILE_EXTRA_COLUMNS:
        if column not in result.columns:
            result[column] = None
    ordered = [column for column in USER_PROFILE_EXPORT_COLUMNS if column in result.columns]
    extra = [column for column in result.columns if column not in ordered]
    return result[ordered + extra].copy()


def _document_frame(document: Any, key: str, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    if fallback is not None:
        return fallback
    if isinstance(document, dict):
        value = document.get(key)
        return value if isinstance(value, pd.DataFrame) else pd.DataFrame()
    value = getattr(document, key, None)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _rows_for_review(structured_rows: pd.DataFrame) -> pd.DataFrame:
    if structured_rows is None or structured_rows.empty:
        return pd.DataFrame(columns=USER_PROFILE_EXPORT_COLUMNS)
    validation_status = structured_rows.get("validation_status", pd.Series("", index=structured_rows.index)).fillna("").astype(str)
    review_status = structured_rows.get("review_status", pd.Series("", index=structured_rows.index)).fillna("").astype(str)
    warnings = structured_rows.get("warnings", pd.Series("", index=structured_rows.index)).fillna("").astype(str).str.strip()
    value_missing = structured_rows.get("value", pd.Series(pd.NA, index=structured_rows.index)).isna()
    mask = validation_status.ne("passed") | review_status.eq("needs_review") | warnings.ne("") | value_missing
    return structured_rows.loc[mask].copy()


def apply_user_profile(
    document: Any,
    profile_config: dict[str, Any],
    *,
    raw_rows: pd.DataFrame | None = None,
    ocr_candidates_df: pd.DataFrame | None = None,
    ocr_runner: Any | None = None,
) -> dict[str, Any]:
    """Apply a complete saved user profile, including extraction source decisions."""
    extraction = profile_config.get("extraction") or {}
    source = str(extraction.get("source") or "pdf_text_layer")
    profile_raw_rows = _document_frame(document, "raw_rows", raw_rows)
    profile_ocr_candidates = _document_frame(document, "ocr_candidates_df", ocr_candidates_df)
    ocr_result = pd.DataFrame()
    ocr_ran = False

    if source in {"ocr", "mixed"} and profile_ocr_candidates.empty and callable(ocr_runner):
        ocr_payload = ocr_runner(document, extraction.get("ocr") or {})
        ocr_ran = True
        if isinstance(ocr_payload, dict):
            candidate_payload = ocr_payload.get("ocr_candidates_df")
            result_payload = ocr_payload.get("ocr_result_df")
            profile_ocr_candidates = candidate_payload if isinstance(candidate_payload, pd.DataFrame) else pd.DataFrame()
            ocr_result = result_payload if isinstance(result_payload, pd.DataFrame) else pd.DataFrame()
        elif isinstance(ocr_payload, pd.DataFrame):
            profile_ocr_candidates = ocr_payload

    ocr_required = bool(source == "ocr" and profile_ocr_candidates.empty and not ocr_ran)

    if source == "ocr":
        application_raw_rows = pd.DataFrame()
        application_ocr_candidates = profile_ocr_candidates
    elif source == "mixed":
        application_raw_rows = profile_raw_rows
        application_ocr_candidates = profile_ocr_candidates
    else:
        application_raw_rows = profile_raw_rows
        application_ocr_candidates = pd.DataFrame()

    structured_rows = apply_user_profile_to_sources(
        application_raw_rows,
        application_ocr_candidates,
        profile_config,
    )
    rows_for_review = _rows_for_review(structured_rows)
    audit_trail = select_user_profile_export_columns(structured_rows)
    return {
        "structured_rows": structured_rows,
        "rows_for_review": rows_for_review,
        "audit_trail": audit_trail,
        "ocr_result_df": ocr_result,
        "ocr_candidates_df": profile_ocr_candidates,
        "ocr_ran": ocr_ran,
        "status": "ocr_required" if ocr_required else "ok",
        "extraction_source": source,
    }


def profile_matches_document(
    profile_config: dict[str, Any],
    raw_rows: pd.DataFrame | None = None,
    ocr_candidates_df: pd.DataFrame | None = None,
) -> bool:
    keywords = (profile_config.get("document_match") or {}).get("keywords") or []
    keywords = [str(keyword).strip().casefold() for keyword in keywords if str(keyword).strip()]
    if not keywords:
        return False
    text_parts: list[str] = []
    if raw_rows is not None and not raw_rows.empty and "evidence_text" in raw_rows.columns:
        text_parts.extend(raw_rows["evidence_text"].fillna("").astype(str).head(500).tolist())
    if ocr_candidates_df is not None and not ocr_candidates_df.empty:
        for column in ["block_text", "preview"]:
            if column in ocr_candidates_df.columns:
                text_parts.extend(ocr_candidates_df[column].fillna("").astype(str).head(100).tolist())
    document_text = "\n".join(text_parts).casefold()
    return all(keyword in document_text for keyword in keywords)


def find_matching_user_profiles(
    profiles: dict[str, dict[str, Any]],
    raw_rows: pd.DataFrame | None = None,
    ocr_candidates_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    return [
        profile
        for profile in profiles.values()
        if profile_matches_document(profile, raw_rows=raw_rows, ocr_candidates_df=ocr_candidates_df)
    ]


def dump_user_profile_json(profile_config: dict[str, Any]) -> str:
    return json.dumps(profile_config, ensure_ascii=False, indent=2)


def dump_user_profile_yaml(profile_config: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(profile_config, allow_unicode=True, sort_keys=False)
    except Exception:
        return dump_user_profile_json(profile_config)


def load_user_profile_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            import yaml

            loaded = yaml.safe_load(text)
        except Exception:
            loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"User profile must be a mapping: {path}")
    return loaded


def load_user_profiles(profiles_dir: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if not profiles_dir.exists():
        return profiles
    for path in sorted(list(profiles_dir.glob("*.yaml")) + list(profiles_dir.glob("*.yml")) + list(profiles_dir.glob("*.json"))):
        try:
            profile = load_user_profile_file(path)
        except Exception:
            continue
        profile_name = str(profile.get("profile_name") or path.stem)
        profile["profile_name"] = profile_name
        profiles[profile_name] = profile
    return profiles


def save_user_profile(profile_config: dict[str, Any], profiles_dir: Path) -> Path:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_name = str(profile_config.get("profile_name") or "user_profile")
    safe_name = safe_file_name(profile_name).replace(".", "_")
    path = profiles_dir / f"{safe_name}.yaml"
    path.write_text(dump_user_profile_yaml(profile_config), encoding="utf-8")
    return path


def select_user_profile_export_columns(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=USER_PROFILE_EXPORT_COLUMNS)
    ordered = [column for column in USER_PROFILE_EXPORT_COLUMNS if column in df.columns]
    extra = [column for column in df.columns if column not in ordered]
    return df[ordered + extra].copy()
