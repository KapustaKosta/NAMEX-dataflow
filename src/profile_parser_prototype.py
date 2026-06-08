from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd


PROTOTYPE_COLUMNS = [
    "source_file",
    "source_type",
    "page",
    "section_id",
    "section_title",
    "section_parse_mode",
    "row_id",
    "indicator",
    "commodity",
    "country",
    "rank",
    "year",
    "value",
    "raw_value",
    "normalized_value",
    "normalization_method",
    "raw_numeric_tokens",
    "parsed_numeric_tokens",
    "unit",
    "currency",
    "evidence_text",
    "extraction_method",
    "extraction_level",
    "confidence",
    "validation_status",
    "warnings",
    "review_status",
]

NUMERIC_TOKEN_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?%?$")
PURE_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
YEAR_FIELD_PATTERN = re.compile(r"^value_(\d{4})$")
WORD_PATTERN = re.compile(r"[^\W\d_]", re.UNICODE)
OCR_DECIMAL_WARNING = "decimal separator may be lost by OCR; value divided by 10"
THOUSAND_TONS_LARGE_WARNING = "suspicious large value for thousand_tons; check OCR decimal separator"
THOUSAND_TONS_SUSPICIOUS_THRESHOLD = 100000
COMPLEX_TRADE_WARNING = "complex trade table; column groups require manual mapping"
YEAR_SERIES_FIELDS_2020_2024 = {"value_2020", "value_2021", "value_2022", "value_2023", "value_2024"}
COMPLEX_TRADE_FIELDS_2023_2024 = {"value_2023", "value_2024", "change_abs", "change_pct"}


def parse_russian_number(text: str) -> float | None:
    """Parse a Russian/OCR numeric token into float."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if WORD_PATTERN.search(raw):
        return None
    cleaned = raw.replace("%", "").replace("\u00a0", " ").strip()
    if not re.fullmatch(r"[+-]?\d[\d ]*(?:[.,]\d+)?", cleaned):
        return None
    cleaned = cleaned.replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_ocr_number(
    raw_value: str,
    section_title: str,
    unit_hint: str | None,
    evidence_text: str,
) -> tuple[float | None, list[str]]:
    """Normalize numeric OCR tokens and return review warnings."""
    parsed = parse_russian_number(raw_value)
    if parsed is None:
        return None, ["manual review required: value is not parseable"]

    raw_text = str(raw_value or "").strip()
    has_decimal_separator = "," in raw_text or "." in raw_text
    normalized_section = _normalized(section_title)
    normalized_unit = _normalized(unit_hint)
    is_million_usd = "млн" in normalized_unit and "долл" in normalized_unit
    is_export_import_country = any(
        keyword in normalized_section for keyword in ("стра", "экспорт", "импорт", "country", "export", "import")
    )
    if not has_decimal_separator and parsed >= 1000 and is_million_usd and is_export_import_country:
        return parsed / 10, [OCR_DECIMAL_WARNING]

    is_thousand_tons = "тыс" in normalized_unit and "тонн" in normalized_unit
    if (
        not has_decimal_separator
        and parsed >= THOUSAND_TONS_SUSPICIOUS_THRESHOLD
        and is_thousand_tons
    ):
        return parsed, [THOUSAND_TONS_LARGE_WARNING]

    return parsed, []


def _empty_prototype_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PROTOTYPE_COLUMNS)


def _normalized(text: object) -> str:
    return str(text or "").replace("ё", "е").casefold()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def section_parse_mode(section: dict[str, Any]) -> str:
    """Classify a draft section before choosing a parser strategy."""
    expected_fields = {str(field) for field in section.get("expected_fields") or []}
    title = _normalized(section.get("title"))
    if COMPLEX_TRADE_FIELDS_2023_2024.issubset(expected_fields):
        return "complex_trade_2023_2024"
    if YEAR_SERIES_FIELDS_2020_2024.issubset(expected_fields) and _contains_any(
        title, ("страны", "СЃС‚СЂР°РЅС‹", "country")
    ):
        return "year_series_2020_2024"
    if YEAR_SERIES_FIELDS_2020_2024.issubset(expected_fields) and _contains_any(
        title, ("производство", "РїСЂРѕРёР·РІРѕРґСЃС‚РІРѕ", "production")
    ):
        return "production_2020_2024"
    return "unknown"


def _normalization_method(raw_value: str, normalized_value: float | None, warnings: list[str]) -> str:
    if normalized_value is None:
        return "manual_review_required"
    if OCR_DECIMAL_WARNING in warnings:
        return "ocr_decimal_divide_by_10"
    raw_text = str(raw_value or "").strip()
    if any(separator in raw_text for separator in (",", ".", " ", "\u00a0", "%")):
        return "russian_number_parse"
    return "none"


def _year_fields(expected_fields: list[str]) -> list[int]:
    years: list[int] = []
    for field in expected_fields:
        match = YEAR_FIELD_PATTERN.fullmatch(str(field))
        if match:
            years.append(int(match.group(1)))
    return years


def _unit_currency(unit_hint: str | None) -> tuple[str | None, str | None]:
    if not unit_hint:
        return None, None
    currency = "USD" if "долл" in _normalized(unit_hint) else None
    return unit_hint, currency


def _token_infos(line: str) -> list[dict[str, Any]]:
    return [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for match in re.finditer(r"\S+", line)
    ]


def _is_numeric_token(token: str) -> bool:
    return bool(NUMERIC_TOKEN_PATTERN.fullmatch(token))


def _combine_numeric_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numbers: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        token_text = str(token["text"])
        if not _is_numeric_token(token_text):
            index += 1
            continue

        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        combined_text = token_text
        end = token["end"]
        if (
            next_token is not None
            and PURE_INTEGER_PATTERN.fullmatch(token_text)
            and 1 <= len(token_text.lstrip("+-")) <= 3
            and re.fullmatch(r"\d{3}(?:[.,]\d+)?%?", str(next_token["text"]))
        ):
            combined_text = f"{token_text} {next_token['text']}"
            end = next_token["end"]
            index += 1

        value = parse_russian_number(combined_text)
        if value is not None:
            numbers.append(
                {
                    "text": combined_text,
                    "raw_value": combined_text,
                    "value": value,
                    "start": token["start"],
                    "end": end,
                }
            )
        index += 1
    return numbers


def _has_leading_rank(line: str, first_number: dict[str, Any] | None, required_value_count: int, numbers_count: int) -> bool:
    if first_number is None:
        return False
    if first_number["start"] > 2:
        return False
    rank_value = first_number["value"]
    if int(rank_value) != rank_value or not (0 < rank_value < 1000):
        return False
    return numbers_count > required_value_count


def _parse_numeric_line(line: str, required_value_count: int, allow_rank: bool) -> dict[str, Any] | None:
    tokens = _token_infos(line)
    numbers = _combine_numeric_tokens(tokens)
    if len(numbers) < required_value_count:
        return None

    first_number = numbers[0] if numbers else None
    has_rank = allow_rank and _has_leading_rank(line, first_number, required_value_count, len(numbers))
    rank = int(first_number["value"]) if has_rank and first_number is not None else None
    value_numbers = numbers[1:] if has_rank else numbers
    if len(value_numbers) < required_value_count:
        return None

    label_start = first_number["end"] if has_rank and first_number is not None else 0
    label_end = value_numbers[0]["start"]
    label = line[label_start:label_end].strip(" -–—:;")
    if not label:
        return None

    extra_numbers = value_numbers[required_value_count:]
    return {
        "label": label,
        "rank": rank,
        "values": value_numbers[:required_value_count],
        "extra_numbers": extra_numbers,
    }


def _parse_complex_numeric_line(line: str) -> dict[str, Any] | None:
    tokens = _token_infos(line)
    numbers = _combine_numeric_tokens(tokens)
    if not numbers:
        return None

    first_number = numbers[0]
    has_rank = _has_leading_rank(line, first_number, required_value_count=1, numbers_count=len(numbers))
    rank = int(first_number["value"]) if has_rank else None
    value_numbers = numbers[1:] if has_rank else numbers
    if not value_numbers:
        return None

    label_start = first_number["end"] if has_rank else 0
    label_end = value_numbers[0]["start"]
    label = line[label_start:label_end].strip(" -вЂ“вЂ”:;")
    if not label:
        return None

    return {
        "label": label,
        "rank": rank,
        "numbers": value_numbers,
    }


def _section_entity_kind(section_title: str, expected_fields: list[str]) -> str:
    normalized = _normalized(section_title)
    if "страны" in normalized or "country" in expected_fields:
        return "country"
    if "производство" in normalized or "commodity" in expected_fields:
        return "commodity"
    return "commodity"


def _validate_row(row: dict[str, Any], year_based: bool = True) -> dict[str, Any]:
    warnings: list[str] = list(row.get("_normalization_warnings") or [])
    structural_warnings: list[str] = []
    if row.get("value") is None:
        structural_warnings.append("value is missing")
    if year_based and row.get("year") is None:
        structural_warnings.append("year is missing")
    if not row.get("commodity") and not row.get("country"):
        structural_warnings.append("commodity or country is missing")
    if not row.get("evidence_text"):
        structural_warnings.append("evidence_text is missing")
    if row.get("_extra_numbers"):
        structural_warnings.append("extra numeric values were left in evidence_text")

    if structural_warnings:
        row["validation_status"] = "needs_review"
        row["warnings"] = "low confidence parse from OCR; " + "; ".join(warnings + structural_warnings)
        row["review_status"] = "needs_review"
        row["confidence"] = min(float(row.get("confidence") or 0.0), 0.55)
    elif warnings:
        row["validation_status"] = "needs_review"
        row["warnings"] = "; ".join(warnings)
        row["review_status"] = "needs_review"
        if row.get("normalization_method") == "ocr_decimal_divide_by_10":
            row["confidence"] = min(float(row.get("confidence") or 0.0), 0.85)
        else:
            row["confidence"] = min(float(row.get("confidence") or 0.0), 0.75)
    else:
        row["validation_status"] = "passed"
        row["warnings"] = ""
        row["review_status"] = "auto_approved"
    row.pop("_extra_numbers", None)
    row.pop("_normalization_warnings", None)
    return row


def _build_base_row(
    profile_draft: dict[str, Any],
    section: dict[str, Any],
    source_file: str,
    row_id: int,
    line: str,
    parse_mode: str,
) -> dict[str, Any]:
    unit, currency = _unit_currency(section.get("unit_hint"))
    return {
        "source_file": source_file,
        "source_type": profile_draft.get("source_type") or "pdf",
        "page": section.get("page"),
        "section_id": section.get("section_id"),
        "section_title": section.get("title"),
        "section_parse_mode": parse_mode,
        "row_id": row_id,
        "indicator": section.get("section_id"),
        "commodity": None,
        "country": None,
        "rank": None,
        "year": None,
        "value": None,
        "raw_value": None,
        "normalized_value": None,
        "normalization_method": "none",
        "raw_numeric_tokens": None,
        "parsed_numeric_tokens": None,
        "unit": unit,
        "currency": currency,
        "evidence_text": line,
        "extraction_method": "draft_profile_parser",
        "extraction_level": "prototype_structured",
        "confidence": float(section.get("section_confidence") or 0.65),
        "validation_status": "needs_review",
        "warnings": "",
        "review_status": "needs_review",
    }


def _parse_year_section(
    profile_draft: dict[str, Any],
    section: dict[str, Any],
    row_id_start: int,
    parse_mode: str,
) -> tuple[list[dict[str, Any]], int]:
    expected_fields = [str(field) for field in section.get("expected_fields") or []]
    years = _year_fields(expected_fields)
    if not years:
        return [], row_id_start

    block_text = str(section.get("block_text") or section.get("preview") or "")
    lines = [line.strip() for line in block_text.splitlines() if line.strip()]
    entity_kind = _section_entity_kind(str(section.get("title") or ""), expected_fields)
    allow_rank = True
    source_file = str((profile_draft.get("metadata") or {}).get("source_file") or "")
    rows: list[dict[str, Any]] = []
    row_id = row_id_start

    for line in lines:
        parsed = _parse_numeric_line(line, required_value_count=len(years), allow_rank=allow_rank)
        if not parsed:
            continue

        for year, number_info in zip(years, parsed["values"]):
            raw_value = str(number_info.get("raw_value") or number_info["text"])
            normalized_value, normalization_warnings = normalize_ocr_number(
                raw_value=raw_value,
                section_title=str(section.get("title") or ""),
                unit_hint=section.get("unit_hint"),
                evidence_text=line,
            )
            row = _build_base_row(profile_draft, section, source_file, row_id, line, parse_mode)
            row["rank"] = parsed["rank"]
            row["year"] = year
            row["raw_value"] = raw_value
            row["normalized_value"] = normalized_value
            row["normalization_method"] = _normalization_method(raw_value, normalized_value, normalization_warnings)
            row["value"] = normalized_value
            row[entity_kind] = parsed["label"]
            row["_extra_numbers"] = parsed["extra_numbers"]
            row["_normalization_warnings"] = normalization_warnings
            rows.append(_validate_row(row, year_based=True))
            row_id += 1

    return rows, row_id


def _parse_complex_trade_section(
    profile_draft: dict[str, Any],
    section: dict[str, Any],
    row_id_start: int,
    parse_mode: str,
) -> tuple[list[dict[str, Any]], int]:
    block_text = str(section.get("block_text") or section.get("preview") or "")
    lines = [line.strip() for line in block_text.splitlines() if line.strip()]
    source_file = str((profile_draft.get("metadata") or {}).get("source_file") or "")
    rows: list[dict[str, Any]] = []
    row_id = row_id_start

    for line in lines:
        parsed = _parse_complex_numeric_line(line)
        if not parsed:
            continue

        row = _build_base_row(profile_draft, section, source_file, row_id, line, parse_mode)
        row["rank"] = parsed["rank"]
        row["commodity"] = parsed["label"]
        row["raw_numeric_tokens"] = json.dumps(
            [str(number["raw_value"]) for number in parsed["numbers"]],
            ensure_ascii=False,
        )
        row["parsed_numeric_tokens"] = json.dumps(
            [number["value"] for number in parsed["numbers"]],
            ensure_ascii=False,
        )
        row["extraction_level"] = "prototype_complex_wide"
        row["validation_status"] = "needs_review"
        row["warnings"] = COMPLEX_TRADE_WARNING
        row["review_status"] = "needs_review"
        row["confidence"] = min(float(row.get("confidence") or 0.0), 0.5)
        rows.append(row)
        row_id += 1

    return rows, row_id


def parse_sections_from_draft(profile_draft: dict[str, Any]) -> pd.DataFrame:
    """Parse prototype structured rows from a source-profile draft."""
    if not profile_draft or not profile_draft.get("target_sections"):
        return _empty_prototype_df()

    rows: list[dict[str, Any]] = []
    row_id = 1
    for section in profile_draft.get("target_sections") or []:
        expected_fields = [str(field) for field in section.get("expected_fields") or []]
        parse_mode = section_parse_mode(section)
        if parse_mode == "complex_trade_2023_2024":
            parsed_rows, row_id = _parse_complex_trade_section(profile_draft, section, row_id, parse_mode)
            rows.extend(parsed_rows)
        elif _year_fields(expected_fields):
            parsed_rows, row_id = _parse_year_section(profile_draft, section, row_id, parse_mode)
            rows.extend(parsed_rows)

    if not rows:
        return _empty_prototype_df()
    return pd.DataFrame(rows, columns=PROTOTYPE_COLUMNS)
