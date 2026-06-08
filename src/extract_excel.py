from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .constants import STANDARD_COLUMNS
from .utils import empty_standard_dataframe, ensure_standard_columns, none_if_missing, row_to_evidence_text


COLUMN_ALIASES = {
    "indicator": ["indicator", "metric", "parameter", "field", "name", "description"],
    "commodity": ["commodity", "product", "goods", "item", "instrument", "asset", "cargo"],
    "region": ["region", "market", "country", "area"],
    "route": ["route", "direction", "lane", "origin destination", "origin_destination"],
    "date": ["date", "day", "period", "report date", "report_date", "trade date", "trade_date"],
    "value": ["value", "price", "amount", "volume", "quantity", "qty", "rate", "tariff", "cost"],
    "unit": ["unit", "units", "measure", "uom", "unit of measure", "unit_of_measure"],
    "currency": ["currency", "ccy", "cur"],
}


def _normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[_\-.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _compact_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_column_name(name))


def _build_column_map(columns: list[object]) -> dict[str, object]:
    normalized = {_normalize_column_name(column): column for column in columns}
    compacted = {_compact_column_name(column): column for column in columns}
    column_map: dict[str, object] = {}

    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_column_name(alias)
            compact_alias = _compact_column_name(alias)
            if normalized_alias in normalized:
                column_map[target] = normalized[normalized_alias]
                break
            if compact_alias in compacted:
                column_map[target] = compacted[compact_alias]
                break

        if target in column_map:
            continue

        for column in columns:
            column_name = _normalize_column_name(column)
            if any(alias in column_name for alias in aliases):
                column_map[target] = column
                break

    return column_map


def _read_csv(file_path: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin1"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            last_error = exc
            try:
                return pd.read_csv(file_path, encoding=encoding, sep=None, engine="python")
            except Exception as fallback_exc:
                last_error = fallback_exc
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def _read_tabular_file(file_path: str) -> dict[str | None, pd.DataFrame]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return {None: _read_csv(file_path)}
    if suffix == ".xlsx":
        return pd.read_excel(file_path, sheet_name=None)
    raise ValueError(f"Unsupported tabular file extension: {suffix}")


def _source_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    return "csv" if suffix == ".csv" else "xlsx"


def extract_excel(file_path: str) -> pd.DataFrame:
    """Extract rows from CSV or XLSX into the standard DataFrame schema."""
    source_file = Path(file_path).name
    source_type = _source_type(file_path)
    sheets = _read_tabular_file(file_path)
    records = []

    for sheet_name, raw_df in sheets.items():
        if raw_df.empty:
            continue

        column_map = _build_column_map(list(raw_df.columns))
        for row_index, row in raw_df.iterrows():
            record = {column: None for column in STANDARD_COLUMNS}
            record["source_file"] = source_file
            record["source_type"] = source_type
            record["sheet"] = sheet_name
            record["row_id"] = int(row_index) + 1
            record["evidence_text"] = row_to_evidence_text(row)
            record["bbox"] = None
            record["extraction_method"] = "pandas"
            record["extraction_level"] = "structured"
            record["confidence"] = 0.9

            for target, source_column in column_map.items():
                record[target] = none_if_missing(row.get(source_column))

            records.append(record)

    if not records:
        return empty_standard_dataframe()

    return ensure_standard_columns(pd.DataFrame(records))
