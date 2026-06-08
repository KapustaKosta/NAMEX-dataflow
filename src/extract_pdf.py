from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import STANDARD_COLUMNS
from .document_profiles import (
    PROFILE_PARSER_CONFIDENCE_THRESHOLD,
    detect_document_profile_with_confidence,
    has_strong_fish_market_markers,
)
from .parsers.fish_market_report import parse_fish_market_report
from .pdf_quality import detect_bad_text_layer
from .raw_table_analysis import build_raw_table_summary
from .utils import empty_standard_dataframe, ensure_standard_columns


def _clean_text(text: object) -> str | None:
    if text is None:
        return None
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in str(text).splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned or None


def _table_row_to_text(row: list[object]) -> str | None:
    cells = [_clean_text(cell) for cell in row]
    cells = [cell for cell in cells if cell]
    return " | ".join(cells) if cells else None


def _split_page_text(text: str) -> list[str]:
    lines = [_clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if lines:
        return lines
    return textwrap.wrap(_clean_text(text) or "", width=900)


def _base_record(
    source_file: str,
    page_number: int,
    row_id: int,
    section_name: str,
    extraction_method: str,
    confidence: float,
) -> dict[str, object]:
    record = {column: None for column in STANDARD_COLUMNS}
    record["source_file"] = source_file
    record["source_type"] = "pdf"
    record["page"] = page_number
    record["row_id"] = row_id
    record["section_name"] = section_name
    record["bbox"] = None
    record["extraction_method"] = extraction_method
    record["extraction_level"] = "raw"
    record["confidence"] = confidence
    return record


def _attach_profile_metadata(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    extraction_strategy: str,
    raw_summary: dict[str, int] | None = None,
    text_layer_quality: dict[str, object] | None = None,
) -> pd.DataFrame:
    result = df
    profile_metadata = dict(metadata)
    profile_metadata["selected_extraction_strategy"] = extraction_strategy
    if text_layer_quality:
        profile_metadata["text_layer_quality"] = text_layer_quality
    table_summary = build_raw_table_summary(result)
    if not result.empty and not table_summary.empty and "table_id" in result.columns:
        result = result.merge(
            table_summary[["table_id", "table_score", "table_reason"]],
            on="table_id",
            how="left",
        )
    result.attrs["profile_detection"] = profile_metadata
    result.attrs["raw_extraction_summary"] = dict(raw_summary or {})
    result.attrs["raw_table_summary"] = table_summary.to_dict("records")
    print(
        "document profile detection: "
        f"profile={profile_metadata.get('profile_name')} "
        f"confidence={profile_metadata.get('profile_confidence')} "
        f"reason={profile_metadata.get('profile_reason')} "
        f"selected_extraction_strategy={extraction_strategy}"
    )
    return result


def extract_pdf(file_path: str, profile_override: str | None = None) -> pd.DataFrame:
    """Extract table rows or text fragments from a PDF via pdfplumber."""
    import pdfplumber

    source_file = Path(file_path).name
    records = []
    raw_rows_by_page: dict[int, list[str]] = {}
    profile_text_parts: list[str] = []
    raw_summary = {
        "text_pages": 0,
        "table_count": 0,
        "table_rows": 0,
        "structured_rows": 0,
    }
    row_id = 1

    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_raw_rows: list[str] = []
            tables = page.extract_tables() or []
            table_row_specs: list[dict[str, object]] = []

            for table_index, table in enumerate(tables, start=1):
                table_had_rows = False
                table_id = f"page_{page_index}_table_{table_index}"
                row_index_in_table = 1
                for row in table:
                    evidence_text = _table_row_to_text(row)
                    if evidence_text:
                        table_had_rows = True
                        table_row_specs.append(
                            {
                                "table_id": table_id,
                                "row_index_in_table": row_index_in_table,
                                "evidence_text": evidence_text,
                            }
                        )
                        page_raw_rows.append(evidence_text)
                        row_index_in_table += 1
                if table_had_rows:
                    raw_summary["table_count"] += 1
                    page_raw_rows.append("")

            page_text = page.extract_text() or ""
            cleaned_page_text = _clean_text(page_text)
            if cleaned_page_text:
                profile_text_parts.append(cleaned_page_text)

            if not page_raw_rows:
                page_raw_rows.extend(_split_page_text(page_text))

            raw_rows_by_page[page_index] = page_raw_rows

            if cleaned_page_text:
                record = _base_record(
                    source_file,
                    page_index,
                    row_id,
                    section_name="raw_page_text",
                    extraction_method="pdfplumber_text",
                    confidence=0.7,
                )
                record["evidence_text"] = cleaned_page_text
                records.append(record)
                raw_summary["text_pages"] += 1
                row_id += 1

            if table_row_specs:
                for table_row_spec in table_row_specs:
                    record = _base_record(
                        source_file,
                        page_index,
                        row_id,
                        section_name="raw_pdf_table",
                        extraction_method="pdfplumber_table",
                        confidence=0.75,
                    )
                    record["table_id"] = table_row_spec["table_id"]
                    record["row_index_in_table"] = table_row_spec["row_index_in_table"]
                    record["evidence_text"] = table_row_spec["evidence_text"]
                    records.append(record)
                    raw_summary["table_rows"] += 1
                    row_id += 1

    profile_text = "\n".join(profile_text_parts + [row for rows in raw_rows_by_page.values() for row in rows])
    text_layer_quality = detect_bad_text_layer(profile_text)
    auto_detection = detect_document_profile_with_confidence(profile_text, source_file)
    if profile_override:
        profile_metadata = {
            "profile_name": profile_override,
            "profile_confidence": 1.0,
            "profile_reason": "Профиль выбран пользователем вручную",
            "profile_selection": "manual",
            "auto_profile_name": auto_detection.get("profile_name"),
            "auto_profile_confidence": auto_detection.get("profile_confidence"),
            "auto_profile_reason": auto_detection.get("profile_reason"),
        }
    else:
        profile_metadata = dict(auto_detection)
        profile_metadata["profile_selection"] = "auto"

    profile = str(profile_metadata.get("profile_name") or "generic_pdf")
    profile_confidence = float(profile_metadata.get("profile_confidence") or 0.0)
    bad_text_layer = bool(text_layer_quality.get("bad_text_layer"))
    text_layer_warning = (
        "PDF text layer contains many CID tokens; OCR is recommended"
        if bad_text_layer
        else ""
    )
    for record in records:
        record["text_layer_quality"] = "bad" if bad_text_layer else "ok"
        record["text_layer_warning"] = text_layer_warning
        if bad_text_layer:
            record["confidence"] = min(float(record.get("confidence") or 0.0), 0.35)

    markers_confirmed = has_strong_fish_market_markers(profile_text, source_file)
    can_attempt_fish_parser = (
        profile == "fish_market_report"
        and (profile_override is not None or profile_confidence >= PROFILE_PARSER_CONFIDENCE_THRESHOLD)
    )

    if can_attempt_fish_parser:
        parsed_pages = []
        for page_number, raw_rows in raw_rows_by_page.items():
            parsed_df = parse_fish_market_report(
                raw_rows,
                source_file=source_file,
                page=page_number,
                require_strong_markers=True,
                profile_markers_confirmed=markers_confirmed,
            )
            if not parsed_df.empty:
                parsed_pages.append(parsed_df)

        if parsed_pages:
            fish_df = pd.concat(parsed_pages, ignore_index=True)
            fish_df["row_id"] = range(1, len(fish_df) + 1)
            raw_summary["structured_rows"] = int(len(fish_df))
            return _attach_profile_metadata(
                ensure_standard_columns(fish_df),
                profile_metadata,
                extraction_strategy="profile_parser",
                raw_summary=raw_summary,
                text_layer_quality=text_layer_quality,
            )

    if not records:
        return _attach_profile_metadata(
            empty_standard_dataframe(),
            profile_metadata,
            extraction_strategy="pdfplumber",
            raw_summary=raw_summary,
            text_layer_quality=text_layer_quality,
        )

    return _attach_profile_metadata(
        ensure_standard_columns(pd.DataFrame(records)),
        profile_metadata,
        extraction_strategy="pdfplumber",
        raw_summary=raw_summary,
        text_layer_quality=text_layer_quality,
    )
