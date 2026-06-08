from __future__ import annotations

import re
from typing import Any

from src.profile_parser_prototype import parse_russian_number


RECONSTRUCTION_WARNING = "numeric tokens reconstructed from OCR; verify values"
FAILED_RECONSTRUCTION_WARNING = "failed to reconstruct expected numeric tokens"
POSSIBLE_LOST_DECIMAL_WARNING = "possible lost decimal separator; value was not auto repaired"


def _token_index(token_key: str) -> int | None:
    match = re.fullmatch(r"token_(\d+)", str(token_key))
    if not match:
        return None
    index = int(match.group(1))
    return index if index >= 1 else None


def _legacy_enabled(value: Any) -> bool:
    return str(value or "ignore") != "ignore"


def _enabled_token_attributes(mapping_config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    mapping = mapping_config.get("token_mapping")
    if mapping is None:
        mapping = mapping_config.get("mapping") or {}

    attributes_by_index: dict[int, dict[str, Any]] = {}
    for token_key, attributes in mapping.items():
        index = _token_index(str(token_key))
        if index is None:
            continue
        if isinstance(attributes, dict):
            enabled = bool(attributes.get("enabled", True))
            if not enabled:
                continue
            attributes_by_index[index] = {
                "metric": str(attributes.get("metric") or "other"),
                "unit": attributes.get("unit"),
                "currency": attributes.get("currency"),
                "year": attributes.get("year"),
            }
        elif _legacy_enabled(attributes):
            attributes_by_index[index] = _legacy_attributes(str(attributes))
    return attributes_by_index


def _legacy_attributes(field: str) -> dict[str, Any]:
    legacy = {
        "volume_2023": ("volume", "thousand_tons", None, 2023),
        "volume_2024": ("volume", "thousand_tons", None, 2024),
        "value_2023": ("trade_value", "million_usd", "USD", 2023),
        "value_2024": ("trade_value", "million_usd", "USD", 2024),
        "change_abs": ("change_abs", None, None, 2024),
        "change_pct": ("change_pct", "percent", None, 2024),
        "share_pct": ("share_pct", "percent", None, None),
        "rank": ("rank", None, None, None),
        "other": ("other", None, None, None),
    }
    metric, unit, currency, year = legacy.get(field, ("other", None, None, None))
    return {"metric": metric, "unit": unit, "currency": currency, "year": year}


def _expected_count(mapping_config: dict[str, Any]) -> int:
    enabled = _enabled_token_attributes(mapping_config)
    return max(enabled.keys(), default=0)


def _normalize_raw_token(token: Any) -> str:
    return str(token).replace("\u00a0", " ").strip()


def _split_space_joined_tokens(raw_tokens: list[str]) -> tuple[list[str], bool]:
    split_tokens: list[str] = []
    changed = False
    for token in raw_tokens:
        normalized = _normalize_raw_token(token)
        if " " in normalized and "," not in normalized and "." not in normalized:
            parts = [part for part in normalized.split() if part]
            if len(parts) > 1 and all(re.fullmatch(r"[+-]?\d+", part) for part in parts):
                split_tokens.extend(parts)
                changed = True
                continue
        split_tokens.append(normalized)
    return split_tokens, changed


def _should_try_join_split(current_attributes: dict[str, Any], next_attributes: dict[str, Any] | None) -> bool:
    current_unit = current_attributes.get("unit")
    next_unit = None if next_attributes is None else next_attributes.get("unit")
    current_metric = str(current_attributes.get("metric") or "")
    next_metric = "" if next_attributes is None else str(next_attributes.get("metric") or "")
    current_is_pct = current_unit == "percent" or current_metric.endswith("_pct") or current_metric == "change_pct"
    next_is_expected_number = next_unit in {"percent", "million_usd", "thousand_tons"} or next_metric.endswith("_pct")
    return current_is_pct and bool(next_is_expected_number)


def _split_joined_long_token(
    tokens: list[str],
    expected_count: int,
    attributes_by_index: dict[int, dict[str, Any]],
) -> tuple[list[str], bool]:
    if len(tokens) >= expected_count:
        return tokens, False

    for position, token in enumerate(tokens):
        clean = token.lstrip("+-")
        sign = "-" if token.startswith("-") else ""
        if not re.fullmatch(r"\d{6}", clean):
            continue
        token_index = position + 1
        if not _should_try_join_split(
            attributes_by_index.get(token_index, {}),
            attributes_by_index.get(token_index + 1),
        ):
            continue
        first = f"{sign}{clean[:3]}"
        second = clean[3:]
        return tokens[:position] + [first, second] + tokens[position + 1 :], True
    return tokens, False


def _insert_missing_before_trailing_percent(
    tokens: list[str],
    expected_count: int,
    attributes_by_index: dict[int, dict[str, Any]],
) -> tuple[list[str | None], bool]:
    if len(tokens) != expected_count - 1 or not tokens:
        return list(tokens), False
    last_attributes = attributes_by_index.get(expected_count, {})
    previous_attributes = attributes_by_index.get(expected_count - 1, {})
    last_is_percent = last_attributes.get("unit") == "percent" or str(last_attributes.get("metric") or "").endswith("_pct")
    previous_is_not_percent = previous_attributes.get("unit") != "percent" and not str(
        previous_attributes.get("metric") or ""
    ).endswith("_pct")
    if last_is_percent and previous_is_not_percent:
        return list(tokens[:-1]) + [None, tokens[-1]], True
    return list(tokens), False


def _digits_without_separator(raw_token: str) -> str:
    return raw_token.strip().lstrip("+-")


def _signed_decimal(raw_token: str, value: float) -> float:
    return -abs(value) if raw_token.strip().startswith("-") else value


def _repair_value(raw_token: str | None, attributes: dict[str, Any]) -> tuple[float | None, str | None]:
    if raw_token is None:
        return None, FAILED_RECONSTRUCTION_WARNING
    parsed = parse_russian_number(raw_token)
    if parsed is None:
        return None, FAILED_RECONSTRUCTION_WARNING

    raw_text = _normalize_raw_token(raw_token)
    if "," in raw_text or "." in raw_text:
        return parsed, None

    digits = _digits_without_separator(raw_text)
    metric = str(attributes.get("metric") or "")
    unit = attributes.get("unit")
    is_percent = unit == "percent" or metric.endswith("_pct") or metric == "change_pct"
    is_decimal_unit = unit in {"thousand_tons", "million_usd"}
    is_small_change_abs = metric.endswith("_change_abs") or metric == "change_abs"

    if is_percent and 1 <= len(digits) <= 3:
        return _signed_decimal(raw_text, int(digits) / 10), RECONSTRUCTION_WARNING
    if (is_decimal_unit or is_small_change_abs) and 1 <= len(digits) <= 3:
        return _signed_decimal(raw_text, int(digits) / 10), RECONSTRUCTION_WARNING
    if (is_percent or is_decimal_unit or is_small_change_abs) and len(digits) >= 4:
        return parsed, POSSIBLE_LOST_DECIMAL_WARNING
    return parsed, None


def reconstruct_numeric_tokens(
    evidence_text: str,
    raw_tokens: list[str],
    mapping_config: dict,
) -> dict[str, Any]:
    """Reconstruct OCR numeric tokens according to the expected mapping schema."""
    expected_count = _expected_count(mapping_config)
    normalized_raw_tokens = [_normalize_raw_token(token) for token in raw_tokens]
    attributes_by_index = _enabled_token_attributes(mapping_config)
    warnings: list[str] = []
    method_flags: set[str] = set()

    if expected_count <= 0:
        return {
            "expected_count": 0,
            "reconstructed_raw_tokens": [],
            "reconstructed_values": [],
            "reconstruction_status": "failed",
            "reconstruction_warnings": [FAILED_RECONSTRUCTION_WARNING],
            "reconstruction_method": "none",
        }

    candidate_tokens, space_split = _split_space_joined_tokens(normalized_raw_tokens)
    if space_split:
        warnings.append(RECONSTRUCTION_WARNING)
        method_flags.add("schema_guided_split")

    candidate_tokens, joined_split = _split_joined_long_token(candidate_tokens, expected_count, attributes_by_index)
    if joined_split:
        warnings.append(RECONSTRUCTION_WARNING)
        method_flags.add("schema_guided_split")

    reconstructed_tokens, inserted_missing = _insert_missing_before_trailing_percent(
        candidate_tokens,
        expected_count,
        attributes_by_index,
    )
    if inserted_missing:
        warnings.append(FAILED_RECONSTRUCTION_WARNING)
        method_flags.add("partial_repair")

    if len(reconstructed_tokens) > expected_count:
        reconstructed_tokens = reconstructed_tokens[:expected_count]
        warnings.append("extra numeric tokens were truncated during reconstruction")
        method_flags.add("partial_repair")

    if len(reconstructed_tokens) < expected_count:
        reconstructed_tokens = reconstructed_tokens + [None] * (expected_count - len(reconstructed_tokens))
        warnings.append(FAILED_RECONSTRUCTION_WARNING)
        method_flags.add("partial_repair")

    values: list[float | None] = []
    for token_index, raw_token in enumerate(reconstructed_tokens, start=1):
        value, warning = _repair_value(raw_token, attributes_by_index.get(token_index, {}))
        values.append(value)
        if warning:
            warnings.append(warning)
            if warning == RECONSTRUCTION_WARNING:
                method_flags.add("decimal_repair")
            else:
                method_flags.add("partial_repair")

    unique_warnings = list(dict.fromkeys(warnings))
    if any(value is None for value in values):
        status = "needs_review"
    elif unique_warnings:
        status = "needs_review"
    else:
        status = "ok"

    if len(values) != expected_count:
        status = "failed"
    if not method_flags:
        method = "none"
    elif "partial_repair" in method_flags:
        method = "partial_repair"
    elif "schema_guided_split" in method_flags:
        method = "schema_guided_split"
    else:
        method = "decimal_repair"

    return {
        "expected_count": expected_count,
        "reconstructed_raw_tokens": reconstructed_tokens,
        "reconstructed_values": values,
        "reconstruction_status": status,
        "reconstruction_warnings": unique_warnings,
        "reconstruction_method": method,
    }
