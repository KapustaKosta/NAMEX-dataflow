from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from src.numeric_token_reconstruction import reconstruct_numeric_tokens
from src.profile_parser_prototype import parse_russian_number


METRIC_OPTIONS = [
    "volume",
    "trade_value",
    "value",
    "price",
    "quantity",
    "share_pct",
    "change_abs",
    "change_pct",
    "volume_change_abs",
    "volume_change_pct",
    "value_change_abs",
    "value_change_pct",
    "other",
]
YEAR_OPTIONS = [None, 2020, 2021, 2022, 2023, 2024, "current_period", "previous_period", "year_start"]
UNIT_OPTIONS = [None, "thousand_tons", "million_usd", "percent", "rub_per_kg", "tons", "kg", "units"]
CURRENCY_OPTIONS = [None, "USD", "RUB", "EUR"]

MAPPING_FIELDS = [
    "ignore",
    "volume_2023",
    "volume_2024",
    "value_2023",
    "value_2024",
    "change_abs",
    "change_pct",
    "share_pct",
    "rank",
    "other",
]

NO_PRESET = "Не применять preset"
TRADE_2023_2024_PRESET = "Trade table 2023/2024"
YEAR_SERIES_2020_2024_PRESET = "Year series 2020-2024"
PRICE_TABLE_PRESET = "Price table"
TOKENS_ONLY_PRESET = "Только сохранить токены без mapping"

MAPPING_PRESETS = {
    NO_PRESET: {},
    TRADE_2023_2024_PRESET: {
        "token_1": {"enabled": True, "metric": "volume", "year": 2023, "unit": "thousand_tons", "currency": None},
        "token_2": {"enabled": True, "metric": "trade_value", "year": 2023, "unit": "million_usd", "currency": "USD"},
        "token_3": {"enabled": True, "metric": "volume", "year": 2024, "unit": "thousand_tons", "currency": None},
        "token_4": {"enabled": True, "metric": "trade_value", "year": 2024, "unit": "million_usd", "currency": "USD"},
        "token_5": {"enabled": True, "metric": "volume_change_abs", "year": 2024, "unit": "thousand_tons", "currency": None},
        "token_6": {"enabled": True, "metric": "volume_change_pct", "year": 2024, "unit": "percent", "currency": None},
        "token_7": {"enabled": True, "metric": "value_change_abs", "year": 2024, "unit": "million_usd", "currency": "USD"},
        "token_8": {"enabled": True, "metric": "value_change_pct", "year": 2024, "unit": "percent", "currency": None},
    },
    YEAR_SERIES_2020_2024_PRESET: {
        "token_1": {"enabled": True, "metric": "value", "year": 2020, "unit": None, "currency": None},
        "token_2": {"enabled": True, "metric": "value", "year": 2021, "unit": None, "currency": None},
        "token_3": {"enabled": True, "metric": "value", "year": 2022, "unit": None, "currency": None},
        "token_4": {"enabled": True, "metric": "value", "year": 2023, "unit": None, "currency": None},
        "token_5": {"enabled": True, "metric": "value", "year": 2024, "unit": None, "currency": None},
    },
    PRICE_TABLE_PRESET: {
        "token_1": {"enabled": True, "metric": "price", "year": "current_period", "unit": None, "currency": None},
        "token_2": {"enabled": True, "metric": "change_pct", "year": "previous_period", "unit": "percent", "currency": None},
        "token_3": {"enabled": True, "metric": "change_pct", "year": "year_start", "unit": "percent", "currency": None},
    },
    TOKENS_ONLY_PRESET: {},
}

MAPPED_COMPLEX_COLUMNS = [
    "source_file",
    "source_type",
    "page",
    "section_id",
    "section_title",
    "row_id",
    "commodity",
    "country",
    "rank",
    "metric",
    "year",
    "value",
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

MAPPING_PREVIEW_COLUMNS = [
    "section_id",
    "section_title",
    "row_id",
    "commodity",
    "rank",
    "raw_numeric_tokens",
    "parsed_numeric_tokens",
    "evidence_text",
]

DEFAULT_MAPPING_WARNING = "mapped from complex OCR table; verify column meaning"
MISSING_TOKEN_WARNING = "mapped token is missing"
AMBIGUOUS_MAPPING_WARNING = "mapped field is ambiguous"


def _empty_mapped_df() -> pd.DataFrame:
    return pd.DataFrame(columns=MAPPED_COMPLEX_COLUMNS)


def _decode_tokens(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    raw = str(value).strip()
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return [token.strip() for token in raw.split("|") if token.strip()]
    return decoded if isinstance(decoded, list) else []


def _coerce_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, int | float):
        return float(value)
    return parse_russian_number(str(value))


def _token_index(token_key: str) -> int | None:
    match = re.fullmatch(r"token_(\d+)", str(token_key))
    if not match:
        return None
    index = int(match.group(1))
    return index if index >= 1 else None


def _normalize_nullable(value: Any) -> Any:
    if value in {"", "None", "none", "null"}:
        return None
    return value


def _legacy_field_to_attributes(field: str) -> dict[str, Any]:
    if field == "ignore":
        return {"enabled": False, "metric": "other", "year": None, "unit": None, "currency": None, "label": ""}
    legacy = {
        "volume_2023": ("volume", 2023),
        "volume_2024": ("volume", 2024),
        "value_2023": ("trade_value", 2023),
        "value_2024": ("trade_value", 2024),
        "change_abs": ("change_abs", 2024),
        "change_pct": ("change_pct", 2024),
        "share_pct": ("share_pct", None),
        "rank": ("rank", None),
        "other": ("other", None),
    }
    metric, year = legacy.get(field, ("other", None))
    return {"enabled": True, "metric": metric, "year": year, "unit": None, "currency": None, "label": ""}


def normalize_token_attributes(value: Any) -> dict[str, Any]:
    """Return a schema-v2 token mapping item from new or legacy input."""
    if isinstance(value, dict):
        enabled = bool(value.get("enabled", True))
        metric = str(value.get("metric") or "other")
        if metric not in METRIC_OPTIONS:
            metric = "other"
        year = _normalize_nullable(value.get("year"))
        unit = _normalize_nullable(value.get("unit"))
        currency = _normalize_nullable(value.get("currency"))
        return {
            "enabled": enabled,
            "metric": metric,
            "year": year,
            "unit": unit if unit in UNIT_OPTIONS else None,
            "currency": currency if currency in CURRENCY_OPTIONS else None,
            "label": str(value.get("label") or ""),
        }
    return _legacy_field_to_attributes(str(value or "ignore"))


def token_mapping_from_preset(preset_name: str, tokens_count: int) -> dict[str, dict[str, Any]]:
    """Build a schema-v2 token mapping from a preset and token count."""
    if tokens_count <= 0:
        return {}
    if preset_name == TOKENS_ONLY_PRESET:
        return {
            f"token_{index}": {"enabled": False, "metric": "other", "year": None, "unit": None, "currency": None, "label": ""}
            for index in range(1, tokens_count + 1)
        }

    preset = MAPPING_PRESETS.get(preset_name, {})
    token_mapping: dict[str, dict[str, Any]] = {}
    for index in range(1, tokens_count + 1):
        token_key = f"token_{index}"
        token_mapping[token_key] = normalize_token_attributes(
            preset.get(token_key, {"enabled": False, "metric": "other", "year": None, "unit": None, "currency": None})
        )
    return token_mapping


def _mapping_items(mapping_config: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    mapping = mapping_config.get("token_mapping")
    if mapping is None:
        mapping = mapping_config.get("mapping") or {}

    items: list[tuple[int, str, dict[str, Any]]] = []
    for token_key, attributes in mapping.items():
        index = _token_index(str(token_key))
        if index is None:
            continue
        normalized = normalize_token_attributes(attributes)
        if not normalized["enabled"]:
            continue
        items.append((index, str(token_key), normalized))
    return sorted(items, key=lambda item: item[0])


def suggest_mapping_preset(section_title: str, evidence_text: str, tokens_count: int) -> str:
    """Suggest a reusable preset from section context without locking the user in."""
    text = f"{section_title}\n{evidence_text}".casefold()
    if "2020-2024" in text:
        return YEAR_SERIES_2020_2024_PRESET
    if "цены" in text or "price" in text:
        return PRICE_TABLE_PRESET
    trade_markers = ("экспорт", "импорт", "export", "import")
    structure_markers = ("2023", "2024", "прирост", "объем", "объём", "стоимость", "value", "volume")
    if tokens_count >= 4 and any(marker in text for marker in trade_markers) and any(
        marker in text for marker in structure_markers
    ):
        return TRADE_2023_2024_PRESET
    return NO_PRESET


def build_mapping_config(
    section_id: str,
    section_title: str,
    token_mapping: dict[str, Any],
    mapping_verified: bool = False,
) -> dict[str, Any]:
    return {
        "section_id": section_id,
        "section_title": section_title,
        "mapping_schema_version": "2",
        "mapping_type": "token_attribute_mapping",
        "token_mapping": {
            str(token_key): normalize_token_attributes(attributes)
            for token_key, attributes in token_mapping.items()
        },
        "created_from": "prototype_complex_wide",
        "requires_review": True,
        "mapping_verified": mapping_verified,
    }


def build_mapping_preview(complex_df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact preview of complex rows and decoded token lists."""
    if complex_df is None or complex_df.empty:
        return pd.DataFrame(columns=MAPPING_PREVIEW_COLUMNS)

    preview = complex_df.copy()
    if "extraction_level" in preview.columns:
        preview = preview.loc[preview["extraction_level"].eq("prototype_complex_wide")].copy()
    if preview.empty:
        return pd.DataFrame(columns=MAPPING_PREVIEW_COLUMNS)

    if "raw_numeric_tokens" in preview.columns:
        preview["raw_numeric_tokens"] = preview["raw_numeric_tokens"].apply(_decode_tokens)
    if "parsed_numeric_tokens" in preview.columns:
        preview["parsed_numeric_tokens"] = preview["parsed_numeric_tokens"].apply(_decode_tokens)
    return preview[[column for column in MAPPING_PREVIEW_COLUMNS if column in preview.columns]].copy()


def apply_complex_mapping(
    complex_df: pd.DataFrame,
    mapping_config: dict,
) -> pd.DataFrame:
    """Apply a schema-v2 token attribute mapping to complex trade rows."""
    if complex_df is None or complex_df.empty:
        return _empty_mapped_df()

    mapping_items = _mapping_items(mapping_config)
    if not mapping_items:
        return _empty_mapped_df()

    rows_df = complex_df.copy()
    section_id = mapping_config.get("section_id")
    if section_id and "section_id" in rows_df.columns:
        rows_df = rows_df.loc[rows_df["section_id"].astype(str).eq(str(section_id))].copy()
    if "extraction_level" in rows_df.columns:
        rows_df = rows_df.loc[rows_df["extraction_level"].eq("prototype_complex_wide")].copy()
    if rows_df.empty:
        return _empty_mapped_df()

    review_status = "mapped_by_user" if mapping_config.get("mapping_verified") else "needs_review"
    mapped_rows: list[dict[str, Any]] = []
    next_row_id = 1
    for _, source_row in rows_df.iterrows():
        raw_tokens = _decode_tokens(source_row.get("raw_numeric_tokens"))
        reconstruction = reconstruct_numeric_tokens(
            evidence_text=str(source_row.get("evidence_text") or ""),
            raw_tokens=[str(token) for token in raw_tokens],
            mapping_config=mapping_config,
        )
        reconstructed_raw_tokens = reconstruction["reconstructed_raw_tokens"]
        reconstructed_values = reconstruction["reconstructed_values"]
        reconstruction_warnings = reconstruction["reconstruction_warnings"]
        reconstruction_status = reconstruction["reconstruction_status"]
        reconstruction_method = reconstruction["reconstruction_method"]

        for token_index, token_key, attributes in mapping_items:
            raw_value = reconstructed_raw_tokens[token_index - 1] if token_index <= len(reconstructed_raw_tokens) else None
            parsed_value = reconstructed_values[token_index - 1] if token_index <= len(reconstructed_values) else None
            value = _coerce_number(parsed_value)
            warnings = [DEFAULT_MAPPING_WARNING]
            warnings.extend(reconstruction_warnings)
            if raw_value is None or value is None:
                warnings.append(MISSING_TOKEN_WARNING)
            if attributes["metric"] == "other":
                warnings.append(AMBIGUOUS_MAPPING_WARNING)

            warnings = list(dict.fromkeys(warnings))
            has_issue = (
                raw_value is None
                or value is None
                or attributes["metric"] == "other"
                or reconstruction_status != "ok"
            )
            mapped_rows.append(
                {
                    "source_file": source_row.get("source_file"),
                    "source_type": source_row.get("source_type"),
                    "page": source_row.get("page"),
                    "section_id": source_row.get("section_id"),
                    "section_title": source_row.get("section_title"),
                    "row_id": next_row_id,
                    "commodity": source_row.get("commodity"),
                    "country": source_row.get("country"),
                    "rank": source_row.get("rank"),
                    "metric": attributes["metric"],
                    "year": attributes["year"],
                    "value": value,
                    "unit": attributes["unit"],
                    "currency": attributes["currency"],
                    "raw_value": raw_value,
                    "normalized_value": value,
                    "normalization_method": "manual_complex_mapping",
                    "mapping_token": token_key,
                    "mapping_label": attributes["label"],
                    "raw_numeric_tokens_original": json.dumps(raw_tokens, ensure_ascii=False),
                    "reconstructed_raw_tokens": json.dumps(reconstructed_raw_tokens, ensure_ascii=False),
                    "reconstruction_method": reconstruction_method,
                    "reconstruction_status": reconstruction_status,
                    "reconstruction_warnings": "; ".join(reconstruction_warnings),
                    "evidence_text": source_row.get("evidence_text"),
                    "extraction_method": "manual_complex_mapping",
                    "extraction_level": "mapped_complex_structured",
                    "confidence": 0.6 if has_issue else 0.75,
                    "validation_status": "needs_review" if has_issue else "passed_with_warning",
                    "warnings": "; ".join(warnings),
                    "review_status": review_status,
                }
            )
            next_row_id += 1

    if not mapped_rows:
        return _empty_mapped_df()
    return pd.DataFrame(mapped_rows, columns=MAPPED_COMPLEX_COLUMNS)
