from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .constants import ALLOWED_CURRENCIES, ALLOWED_UNITS, CURRENCY_MAP, UNIT_MAP
from .utils import ensure_standard_columns, is_missing


def _normalize_key(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\\", "/")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_unit(unit: str) -> str | None:
    """Normalize measurement units to the configured unit vocabulary."""
    if is_missing(unit):
        return None
    text = str(unit).strip()
    if text in ALLOWED_UNITS:
        return text
    key = _normalize_key(text)
    compact_key = key.replace(" ", "")
    return UNIT_MAP.get(key) or UNIT_MAP.get(compact_key) or text


def normalize_currency(currency: str) -> str | None:
    """Normalize currency names and symbols to RUB, USD, or EUR."""
    if is_missing(currency):
        return None
    text = str(currency).strip()
    if text in ALLOWED_CURRENCIES:
        return text
    key = _normalize_key(text)
    return CURRENCY_MAP.get(key) or text.upper()


def _normalize_value(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("\u00a0", " ").replace(" ", "")
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


def _normalize_date(value: Any) -> str | None:
    if is_missing(value):
        return None

    text = str(value).strip()
    iso_like = re.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text)
    if iso_like:
        parsed = pd.to_datetime(text, errors="coerce", yearfirst=True)
    else:
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)

    if not pd.isna(parsed):
        return parsed.strftime("%Y-%m-%d")

    return text or None


def _normalize_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize units, currencies, values, dates, and core text fields."""
    result = ensure_standard_columns(df)

    result["unit"] = result["unit"].apply(normalize_unit)
    result["currency"] = result["currency"].apply(normalize_currency)
    result["value"] = result["value"].apply(_normalize_value)
    result["date"] = result["date"].apply(_normalize_date)

    for column in ["indicator", "commodity", "region", "route", "evidence_text"]:
        result[column] = result[column].apply(_normalize_text)

    result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce")
    return result
