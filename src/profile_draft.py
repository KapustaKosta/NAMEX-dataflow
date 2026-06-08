from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd


DEFAULT_PROFILE_NAME = "agro_kazakhstan_review"
DEFAULT_DISPLAY_NAME = "Обзор ВЭД / Казахстан / АПК"
DEFAULT_VALIDATION_RULES = [
    "numeric columns must be parseable",
    "year columns must be in expected range",
    "country/name field must be non-empty",
]
FALLBACK_EXPECTED_FIELDS = ["name", "value", "unit"]
WEAK_TITLE_PREFIXES = (
    "в то же время",
    "в топ",
    "в ton",
    "при этом",
    "также",
    "большая часть",
)
SENTENCE_TITLE_PREFIXES = WEAK_TITLE_PREFIXES + ("в ", "при ", "однако")

UNIT_PATTERNS = [
    "млн долл. США",
    "млн долл США",
    "млн долл",
    "тыс. тонн",
    "тыс тонн",
    "млрд шт.",
    "млрд шт",
    "млн тонн",
    "тонн",
    "долл.",
    "долл",
    "%",
]

CYRILLIC_TRANSLITERATION = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


def _normalized(text: object) -> str:
    return str(text or "").replace("ё", "е").casefold()


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def suggest_section_id(title: str, index: int = 1) -> str:
    normalized = _normalized(title)
    if "основные страны" in normalized and "экспорт" in normalized:
        return "main_export_countries"
    if "основные страны" in normalized and "импорт" in normalized:
        return "main_import_countries"
    if "товарная структура" in normalized and "импорт" in normalized:
        return "import_commodity_structure"
    if "товарная структура" in normalized and "экспорт" in normalized:
        return "export_commodity_structure"
    if "производство" in normalized and "растениевод" in normalized:
        return "crop_production"
    if "производство" in normalized and "животновод" in normalized:
        return "livestock_production"

    transliterated = normalized.translate(CYRILLIC_TRANSLITERATION)
    slug = re.sub(r"[^a-z0-9]+", "_", transliterated).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:80] if slug else f"section_{index}"


def suggest_expected_fields(title: str, preview: str = "") -> list[str]:
    normalized = _normalized(f"{title}\n{preview}")
    if "товарная структура" in normalized:
        return ["rank", "commodity", "value", "share_pct"]
    if "2023-2024" in normalized:
        return ["commodity", "value_2023", "value_2024", "change_abs", "change_pct"]
    if "2020-2024" in normalized:
        return ["rank", "name", "value_2020", "value_2021", "value_2022", "value_2023", "value_2024"]
    if "производство" in normalized:
        return ["commodity", "value_2020", "value_2021", "value_2022", "value_2023", "value_2024"]
    return list(FALLBACK_EXPECTED_FIELDS)


def expected_fields_are_fallback(expected_fields: list[str]) -> bool:
    return expected_fields == FALLBACK_EXPECTED_FIELDS


def expected_fields_have_years(expected_fields: list[str]) -> bool:
    return any(re.fullmatch(r"value_\d{4}", field) for field in expected_fields)


def suggest_unit_hint(title: str, preview: str = "") -> str | None:
    normalized = _normalized(f"{title}\n{preview}")
    for unit in UNIT_PATTERNS:
        if _normalized(unit) in normalized:
            if unit == "млн долл США":
                return "млн долл. США"
            if unit == "тыс тонн":
                return "тыс. тонн"
            if unit == "млрд шт":
                return "млрд шт."
            if unit == "долл":
                return "долл."
            return unit
    return None


def suggest_parser_hint(candidate_type: str, table_score: float | None) -> str:
    if candidate_type == "table":
        return "OCR table-like block: rows look like name + numeric values"
    if candidate_type == "mixed":
        return "OCR mixed block: split prose from table-like rows before parsing"
    if candidate_type == "paragraph":
        return "OCR paragraph-like block: useful context, not a primary table parser target"
    if candidate_type == "chart_text":
        return "OCR chart text: numeric labels may need manual chart interpretation"
    if table_score is not None and table_score >= 0.65:
        return "OCR table-like block: inspect row structure before implementing parser"
    return "OCR candidate block: requires developer review"


def title_looks_like_sentence(title: str) -> bool:
    normalized = _normalized(title).strip()
    if not normalized:
        return False
    if normalized.startswith(SENTENCE_TITLE_PREFIXES):
        return True
    explicit_table_title = (
        "товарная структура" in normalized
        or "основные страны" in normalized
        or "производство" in normalized
        or "2020-2024" in normalized
        or "2023-2024" in normalized
    )
    if normalized.endswith(".") and len(normalized) >= 45 and not explicit_table_title:
        return True
    sentence_punctuation = sum(normalized.count(char) for char in ".!?")
    return sentence_punctuation >= 2 and len(normalized) >= 60 and not explicit_table_title


def title_has_weak_prefix(title: str) -> bool:
    normalized = _normalized(title).strip()
    return normalized.startswith(WEAK_TITLE_PREFIXES)


def evaluate_section_quality(
    title: str,
    candidate_type: str,
    table_score: float | None,
    expected_fields: list[str],
    unit_hint: str | None,
) -> dict[str, Any]:
    score = float(table_score or 0.0)
    fallback_fields = expected_fields_are_fallback(expected_fields)
    has_year_columns = expected_fields_have_years(expected_fields)
    sentence_title = title_looks_like_sentence(title)
    weak_prefix = title_has_weak_prefix(title)
    warnings: list[str] = []

    if fallback_fields:
        warnings.append("expected_fields were inferred by fallback rule")
    if sentence_title:
        warnings.append("block title looks like a sentence, not a table title")
    if candidate_type == "mixed":
        warnings.append("candidate_type is mixed, parser rules require manual review")
    elif candidate_type in {"paragraph", "chart_text"}:
        warnings.append(f"candidate_type is {candidate_type}, section is not a stable OCR table")
    if not unit_hint and not has_year_columns:
        warnings.append("unit_hint is missing")
    if score < 0.75:
        warnings.append("table_score is below recommended threshold")
    if weak_prefix:
        warnings.append("block title starts with prose marker")

    if candidate_type in {"paragraph", "chart_text"} or score < 0.5 or weak_prefix:
        section_quality = "weak"
    elif (
        candidate_type == "table"
        and score >= 0.75
        and not fallback_fields
        and bool(unit_hint or has_year_columns)
        and not sentence_title
    ):
        section_quality = "good"
    else:
        section_quality = "needs_review"

    confidence = score
    if section_quality == "good":
        confidence += 0.05
    if fallback_fields:
        confidence -= 0.20
    if sentence_title:
        confidence -= 0.15
    if candidate_type == "mixed":
        confidence -= 0.10
    elif candidate_type in {"paragraph", "chart_text"}:
        confidence -= 0.25
    if not unit_hint and not has_year_columns:
        confidence -= 0.10
    if weak_prefix:
        confidence -= 0.15

    return {
        "section_quality": section_quality,
        "section_warnings": warnings,
        "section_confidence": max(0.0, min(1.0, round(confidence, 2))),
    }


def summarize_profile_draft_sections(target_sections: list[dict[str, Any]]) -> dict[str, Any]:
    good_sections = sum(1 for section in target_sections if section.get("section_quality") == "good")
    needs_review_sections = sum(1 for section in target_sections if section.get("section_quality") == "needs_review")
    weak_sections = sum(1 for section in target_sections if section.get("section_quality") == "weak")
    has_warnings = bool(needs_review_sections or weak_sections)
    return {
        "total_sections": len(target_sections),
        "good_sections": good_sections,
        "needs_review_sections": needs_review_sections,
        "weak_sections": weak_sections,
        "has_warnings": has_warnings,
        "ready_for_parser_prototype": bool(good_sections),
        "requires_developer_review": True,
    }


def candidate_is_good_profile_section(candidate: pd.Series | dict[str, Any]) -> bool:
    title = str(candidate.get("block_title") or "")
    preview = str(candidate.get("preview") or candidate.get("block_text") or "")
    expected_fields = suggest_expected_fields(title, preview)
    quality = evaluate_section_quality(
        title=title,
        candidate_type=str(candidate.get("candidate_type") or "unknown"),
        table_score=_clean_scalar(candidate.get("table_score")),
        expected_fields=expected_fields,
        unit_hint=suggest_unit_hint(title, preview),
    )
    return quality["section_quality"] == "good"


def build_profile_draft(
    source_file: str,
    selected_candidates_df: pd.DataFrame,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Build a source-profile draft from manually selected OCR candidates."""
    draft_profile_name = profile_name or DEFAULT_PROFILE_NAME
    target_sections: list[dict[str, Any]] = []

    if selected_candidates_df is not None and not selected_candidates_df.empty:
        candidates = selected_candidates_df.copy().reset_index(drop=True)
        for index, candidate in candidates.iterrows():
            title = str(candidate.get("block_title") or f"section_{index + 1}")
            preview = str(candidate.get("preview") or candidate.get("block_text") or "")
            candidate_type = str(candidate.get("candidate_type") or "unknown")
            table_score = _clean_scalar(candidate.get("table_score"))
            information_score = _clean_scalar(candidate.get("information_score"))
            expected_fields = suggest_expected_fields(title, preview)
            unit_hint = suggest_unit_hint(title, preview)
            quality = evaluate_section_quality(
                title=title,
                candidate_type=candidate_type,
                table_score=table_score,
                expected_fields=expected_fields,
                unit_hint=unit_hint,
            )
            section = {
                "section_id": suggest_section_id(title, index=index + 1),
                "title": title,
                "page": _clean_scalar(candidate.get("page")),
                "candidate_type": candidate_type,
                "table_score": table_score,
                "information_score": information_score,
                "section_quality": quality["section_quality"],
                "section_confidence": quality["section_confidence"],
                "section_warnings": quality["section_warnings"],
                "expected_fields": expected_fields,
                "unit_hint": unit_hint,
                "parser_hint": suggest_parser_hint(candidate_type, table_score),
                "validation_rules": list(DEFAULT_VALIDATION_RULES),
                "preview": preview,
                "block_text": str(candidate.get("block_text") or preview),
            }
            target_sections.append(section)

    profile_draft_summary = summarize_profile_draft_sections(target_sections)
    return {
        "profile_name": draft_profile_name,
        "display_name": DEFAULT_DISPLAY_NAME,
        "source_type": "pdf",
        "document_profile": draft_profile_name,
        "extraction_strategy": "ocr_profile_parser",
        "requires_ocr": True,
        "ocr_lang": "rus+eng",
        "profile_draft_summary": profile_draft_summary,
        "target_sections": target_sections,
        "validation_rules": list(DEFAULT_VALIDATION_RULES),
        "metadata": {
            "source_file": source_file,
            "created_from": "ocr_candidates",
            "note": "draft profile, requires developer review",
        },
    }


def dump_profile_draft_json(profile_draft: dict[str, Any]) -> str:
    """Serialize the draft to pretty UTF-8 JSON."""
    return json.dumps(profile_draft, ensure_ascii=False, indent=2)


def _format_yaml_scalar(value: Any) -> str:
    value = _clean_scalar(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _append_yaml_scalar(lines: list[str], key_prefix: str, value: Any, child_indent: int) -> None:
    value = _clean_scalar(value)
    if isinstance(value, str) and "\n" in value:
        lines.append(f"{key_prefix} |")
        child_spaces = " " * child_indent
        lines.extend(f"{child_spaces}{line}" for line in value.splitlines())
    else:
        lines.append(f"{key_prefix} {_format_yaml_scalar(value)}")


def _dump_yaml(value: Any, indent: int = 0) -> list[str]:
    spaces = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{spaces}{key}:")
                lines.extend(_dump_yaml(item, indent + 2))
            else:
                _append_yaml_scalar(lines, f"{spaces}{key}:", item, indent + 2)
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{spaces}[]"]
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{spaces}-")
                lines.extend(_dump_yaml(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{spaces}-")
                lines.extend(_dump_yaml(item, indent + 2))
            else:
                if isinstance(item, str) and "\n" in item:
                    lines.append(f"{spaces}- |")
                    lines.extend(f"{' ' * (indent + 2)}{line}" for line in item.splitlines())
                else:
                    lines.append(f"{spaces}- {_format_yaml_scalar(item)}")
        return lines
    return [f"{spaces}{_format_yaml_scalar(value)}"]


def dump_profile_draft_yaml(profile_draft: dict[str, Any]) -> str:
    """Serialize the draft to a compact YAML-compatible text format."""
    return "\n".join(_dump_yaml(profile_draft)) + "\n"
