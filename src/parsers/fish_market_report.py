from __future__ import annotations

import re

import pandas as pd

from ..constants import STANDARD_COLUMNS
from ..document_profiles import has_strong_fish_market_markers
from ..utils import empty_standard_dataframe, ensure_standard_columns


NUMBER_PATTERN = re.compile(r"[+-]?\s*\d[\d\s\u00a0]*(?:,\d+)(?:\s*%)?")
MISSING_VALUE_TOKENS = {"", "-", "—", "–"}
CATCH_INDICATORS = [
    ("catch_volume", "thousand_tons"),
    ("yoy_change", "percent"),
    ("quota_utilization", "percent"),
]
WHOLESALE_INDICATORS = [
    ("wholesale_price", "RUB/kg"),
    ("weekly_change", "percent"),
    ("ytd_change", "percent"),
]
WORLD_PRICE_INDICATORS = [
    ("world_price", None),
    ("monthly_change", "percent"),
    ("yearly_change", "percent"),
]
RETAIL_INDICATORS = [
    ("retail_price_frozen_fish", "RUB/kg"),
    ("weekly_change", "percent"),
    ("ytd_change", "percent"),
]
RETAIL_REGIONS = ["РФ", "ЦФО", "СЗФО", "ДВФО"]
WHOLESALE_SECTION_BY_REGION = {
    "Дальний Восток": "wholesale_far_east",
    "Северо-Запад": "wholesale_north_west",
    "Центр": "wholesale_center",
}


def parse_russian_number(value: str | None) -> float | None:
    """Parse Russian-formatted numbers such as '1 947,4', '+ 33,3%', '81,93', and '-'."""
    if value is None:
        return None

    cleaned = str(value).replace("\u00a0", "")
    cleaned = cleaned.replace(" ", "").replace("%", "").replace(",", ".").strip()
    if cleaned in MISSING_VALUE_TOKENS:
        return None

    cleaned = re.sub(r"[^0-9+\-.]", "", cleaned)
    if cleaned in MISSING_VALUE_TOKENS:
        return None

    return float(cleaned)


def _clean_row(row: object) -> str | None:
    text = str(row).replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = text.strip("| ")
    return text or None


def _one_line(row: object) -> str | None:
    text = _clean_row(row)
    if not text:
        return None
    return re.sub(r"\s+", " ", text).strip()


def _clean_number_text(value: object) -> str:
    text = str(value).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^([+-])\s+", r"\1", text)
    if text in MISSING_VALUE_TOKENS:
        return "-"
    return text


def _numbers(row: str) -> list[str]:
    return [_clean_number_text(number) for number in NUMBER_PATTERN.findall(row)]


def _is_value_token(value: object) -> bool:
    text = _clean_number_text(value)
    if text in MISSING_VALUE_TOKENS:
        return True
    return bool(NUMBER_PATTERN.search(text))


def _metric_values(row: str, expected: int = 3) -> list[str]:
    if "|" in row:
        values = [_clean_number_text(cell) for cell in _cells(row) if _is_value_token(cell)]
        if values:
            return values[:expected]
    return _numbers(row)[:expected]


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.split("|")]


def _split_segments(raw_rows: list[str]) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    for raw_row in raw_rows:
        row = _clean_row(raw_row)
        if row:
            current.append(row)
            continue
        if current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def split_multiline_table_row(row: list[object]) -> list[list[str | None]]:
    """Split a pdfplumber row with multiline cells into normal row-shaped lists."""
    split_cells = []
    for cell in row:
        if cell is None:
            split_cells.append([None])
            continue

        text = str(cell).replace("\u00a0", " ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        split_cells.append(lines or [None])

    max_lines = max(len(cell_lines) for cell_lines in split_cells)
    expanded_rows = []
    for line_index in range(max_lines):
        expanded_cells = []
        for cell_lines in split_cells:
            expanded_cells.append(cell_lines[line_index] if line_index < len(cell_lines) else None)
        expanded_rows.append(expanded_cells)
    return expanded_rows


def _expand_multiline_row(row: str) -> list[str]:
    cells = _cells(row)
    if not any("\n" in cell for cell in cells):
        return [row]

    expanded_rows = []
    for expanded_cells in split_multiline_table_row(cells):
        text_cells = ["" if cell is None else cell for cell in expanded_cells]
        expanded_rows.append(" | ".join(text_cells).strip(" |"))
    return expanded_rows


def _base_record(
    source_file: str,
    page: int,
    row_id: int,
    commodity: str,
    evidence_text: str,
    region: str | None = None,
    section_name: str | None = None,
) -> dict[str, object]:
    record = {column: None for column in STANDARD_COLUMNS}
    record["source_file"] = source_file
    record["source_type"] = "pdf"
    record["page"] = page
    record["row_id"] = row_id
    record["section_name"] = section_name
    record["commodity"] = commodity
    record["region"] = region
    record["evidence_text"] = evidence_text
    record["bbox"] = None
    record["extraction_method"] = "fish_market_report_parser"
    record["extraction_level"] = "structured"
    record["confidence"] = 0.95
    return record


def _append_indicator_records(
    records: list[dict[str, object]],
    source_file: str,
    page: int,
    commodity: str,
    values: list[str | None],
    indicators: list[tuple[str, str | None]],
    evidence_text: str,
    region: str | None = None,
    first_unit: str | None = None,
    section_name: str | None = None,
) -> None:
    for index, (indicator, unit) in enumerate(indicators):
        record = _base_record(
            source_file=source_file,
            page=page,
            row_id=len(records) + 1,
            commodity=commodity,
            evidence_text=evidence_text,
            region=region,
            section_name=section_name,
        )
        record["indicator"] = indicator
        record["value"] = parse_russian_number(values[index])
        record["unit"] = first_unit if index == 0 and first_unit else unit
        records.append(record)


def _is_possible_commodity(row: str) -> bool:
    if NUMBER_PATTERN.search(row):
        return False
    if len(row) > 80:
        return False
    lowered = row.casefold()
    skipped_words = [
        "улов",
        "таблица",
        "показатель",
        "освоение",
        "изменение",
        "руб./кг",
        "кроны/кг",
        "евро/кг",
        "норвегия",
        "греция",
    ]
    return not any(word in lowered for word in skipped_words)


def _parse_catch_table(segment: list[str], source_file: str, page: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    pending_commodity: str | None = None
    seen_items: set[tuple[str, tuple[str, str, str]]] = set()

    for raw_row in segment:
        row = _one_line(raw_row)
        if not row:
            continue

        values = _metric_values(row, expected=3)
        if len(values) >= 3 and pending_commodity:
            first_three = values[:3]
            item_key = (pending_commodity, tuple(first_three))
            if item_key in seen_items:
                pending_commodity = None
                continue
            seen_items.add(item_key)

            evidence_text = f"{pending_commodity} | {first_three[0]} | {first_three[1]} | {first_three[2]}"
            _append_indicator_records(
                records=records,
                source_file=source_file,
                page=page,
                commodity=pending_commodity,
                values=first_three,
                indicators=CATCH_INDICATORS,
                evidence_text=evidence_text,
                section_name="catch_main_species",
            )
            pending_commodity = None
            continue

        if _is_possible_commodity(row):
            pending_commodity = row

    return records


def _parse_price_row(row: str) -> tuple[str, list[str]] | None:
    cells = _cells(row)
    if len(cells) < 4:
        return None

    commodity = cells[0].strip()
    if not commodity or "руб" in commodity.casefold() or "базис" in commodity.casefold():
        return None

    values = [_clean_number_text(cell) for cell in cells[1:4]]
    if any(not _is_value_token(value) for value in values):
        return None

    return commodity, values


def _detect_wholesale_region(rows: list[tuple[str, list[str]]]) -> str | None:
    commodities = {commodity for commodity, _ in rows}
    if {"Треска тихоок.", "Минтай", "Сельдь тихоок.", "Камбала"}.issubset(commodities):
        return "Дальний Восток"
    if {"Пикша", "Треска атлант.", "Скумбрия атлант.", "Сельдь атлант."}.issubset(commodities):
        return "Северо-Запад"
    if {"Скумбрия атлант.", "Минтай", "Сельдь тихоок.", "Сельдь атлант."}.issubset(commodities):
        return "Центр"
    if ("Треска тихоок." in commodities or "Сельдь тихоок." in commodities) and "Камбала" in commodities:
        return "Дальний Восток"
    if "Пикша" in commodities and "Треска атлант." in commodities:
        return "Северо-Запад"
    if "Мойва атлант." in commodities or ("Скумбрия атлант." in commodities and "Минтай" in commodities):
        return "Центр"
    return None


def _parse_wholesale_table(segment: list[str], source_file: str, page: int) -> list[dict[str, object]]:
    parsed_rows: list[tuple[str, list[str]]] = []
    for raw_row in segment:
        for expanded_row in _expand_multiline_row(raw_row):
            row = _one_line(expanded_row)
            if not row:
                continue
            parsed_row = _parse_price_row(row)
            if parsed_row:
                parsed_rows.append(parsed_row)

    region = _detect_wholesale_region(parsed_rows)
    if not region:
        return []

    records: list[dict[str, object]] = []
    section_name = WHOLESALE_SECTION_BY_REGION.get(region)
    for commodity, values in parsed_rows:
        evidence_text = f"{region} | {commodity} | {values[0]} | {values[1]} | {values[2]}"
        _append_indicator_records(
            records=records,
            source_file=source_file,
            page=page,
            commodity=commodity,
            values=values,
            indicators=WHOLESALE_INDICATORS,
            evidence_text=evidence_text,
            region=region,
            section_name=section_name,
        )
    return records


def _parse_world_header(row: str) -> tuple[str, str, str] | None:
    lowered = row.casefold()
    if "кроны/кг" not in lowered and "евро/кг" not in lowered:
        return None

    match = re.match(r"^(.*?)\s*\(([^)]+)\),\s*(.*?)$", row)
    if not match:
        return None

    commodity = match.group(1).strip()
    region = match.group(2).strip()
    unit_text = match.group(3).casefold()
    if "кроны/кг" in unit_text:
        unit = "NOK/kg"
    elif "евро/кг" in unit_text:
        unit = "EUR/kg"
    else:
        return None
    return commodity, region, unit


def _parse_world_price_table(segment: list[str], source_file: str, page: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    pending_item: tuple[str, str, str, str] | None = None

    for raw_row in segment:
        row = _one_line(raw_row)
        if not row:
            continue

        header = _parse_world_header(row)
        if header:
            commodity, region, unit = header
            pending_item = (commodity, region, unit, row)
            continue

        numbers = _numbers(row)
        if len(numbers) >= 3 and pending_item:
            commodity, region, unit, header_text = pending_item
            first_three = numbers[:3]
            evidence_text = f"{header_text} | {first_three[0]} | {first_three[1]} | {first_three[2]}"
            _append_indicator_records(
                records=records,
                source_file=source_file,
                page=page,
                commodity=commodity,
                values=first_three,
                indicators=WORLD_PRICE_INDICATORS,
                evidence_text=evidence_text,
                region=region,
                first_unit=unit,
                section_name="export_market_prices",
            )
            pending_item = None

    return records


def _parse_retail_table(segment: list[str], source_file: str, page: int) -> list[dict[str, object]]:
    numeric_rows = []
    for raw_row in segment:
        row = _one_line(raw_row)
        if not row:
            continue
        numbers = _numbers(row)
        if numbers:
            numeric_rows.append((row, numbers))

    price_values = next((numbers for _, numbers in numeric_rows if len(numbers) == 4 and "%" not in _), None)
    weekly_values = next((numbers for row, numbers in numeric_rows if len(numbers) == 4 and "%" in row), None)
    ytd_values = None

    for row, numbers in numeric_rows:
        if len(numbers) == 4 and "%" in row and numbers != weekly_values:
            ytd_values = numbers
            break

    if ytd_values is None:
        single_percent = next((numbers[0] for _, numbers in numeric_rows if len(numbers) == 1), None)
        three_percents = next((numbers for _, numbers in numeric_rows if len(numbers) == 3), None)
        if single_percent and three_percents:
            ytd_values = three_percents + [single_percent]

    if not price_values or not weekly_values or not ytd_values:
        return []

    records: list[dict[str, object]] = []
    for index, region in enumerate(RETAIL_REGIONS):
        values = [price_values[index], weekly_values[index], ytd_values[index]]
        evidence_text = f"{region} | {values[0]} | {values[1]} | {values[2]}"
        _append_indicator_records(
            records=records,
            source_file=source_file,
            page=page,
            commodity="рыба мороженая",
            values=values,
            indicators=RETAIL_INDICATORS,
            evidence_text=evidence_text,
            region=region,
            section_name="retail_frozen_fish",
        )
    return records


def parse_fish_market_report(
    raw_rows: list[str],
    source_file: str,
    page: int,
    *,
    require_strong_markers: bool = True,
    profile_markers_confirmed: bool = False,
) -> pd.DataFrame:
    """Parse key tables from the fish market PDF report into long-format rows."""
    raw_text = "\n".join(str(row) for row in raw_rows)
    if require_strong_markers and not profile_markers_confirmed and not has_strong_fish_market_markers(raw_text, source_file):
        print("fish_market_report_parser skipped: no strong profile markers")
        return empty_standard_dataframe()

    records: list[dict[str, object]] = []

    for segment in _split_segments(raw_rows):
        records.extend(_parse_catch_table(segment, source_file, page))
        records.extend(_parse_wholesale_table(segment, source_file, page))
        records.extend(_parse_world_price_table(segment, source_file, page))
        records.extend(_parse_retail_table(segment, source_file, page))

    if not records:
        return empty_standard_dataframe()

    result = ensure_standard_columns(pd.DataFrame(records))
    result["row_id"] = range(1, len(result) + 1)
    return result
