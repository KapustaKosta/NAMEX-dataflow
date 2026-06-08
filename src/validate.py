from __future__ import annotations

import pandas as pd

from .constants import ALLOWED_CURRENCIES, ALLOWED_UNITS
from .utils import ensure_standard_columns, is_missing


CHANGE_INDICATORS = {
    "yoy_change",
    "weekly_change",
    "ytd_change",
    "monthly_change",
    "yearly_change",
}


SOFT_MISSING_VALUE_METHODS = {
    "fish_market_report_parser",
}

RAW_EXTRACTION_LEVELS = {"raw", "raw_ocr"}


def _row_extraction_level(row: pd.Series) -> str:
    level = row.get("extraction_level")
    if not is_missing(level):
        return str(level)
    return "structured"


def _duplicate_mask(df: pd.DataFrame) -> pd.Series:
    key_columns = ["source_file", "date", "commodity", "indicator", "region"]
    key_df = df[key_columns].replace("", pd.NA)
    has_business_key = key_df[["date", "commodity", "indicator", "region"]].notna().any(axis=1)
    return key_df.duplicated(keep=False) & has_business_key


def _review_status(validation_status: str, confidence: float | None) -> str:
    if validation_status == "failed":
        return "manual_required"
    if validation_status == "passed" and confidence is not None and confidence >= 0.95:
        return "auto_approved"
    return "needs_review"


def validate_extracted_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate extracted data and add status, warning, and review columns."""
    result = ensure_standard_columns(df)
    if not result.empty:
        result["extraction_level"] = result.apply(_row_extraction_level, axis=1)
    duplicate_rows = _duplicate_mask(result) if not result.empty else pd.Series(dtype=bool)

    statuses = []
    warning_values = []
    review_statuses = []

    for index, row in result.iterrows():
        warnings: list[str] = []
        critical_error = False
        extraction_level = _row_extraction_level(row)

        if extraction_level in RAW_EXTRACTION_LEVELS:
            if is_missing(row.get("evidence_text")):
                warnings.append("missing evidence_text")
            if row.get("text_layer_quality") == "bad":
                warnings.append("bad PDF text layer; OCR recommended")
            statuses.append("raw_extracted")
            warning_values.append("; ".join(warnings))
            if extraction_level == "raw_ocr" and is_missing(row.get("evidence_text")):
                review_statuses.append("manual_required")
            else:
                review_statuses.append("needs_ocr" if row.get("text_layer_quality") == "bad" else "needs_profile_setup")
            continue

        value = pd.to_numeric(row.get("value"), errors="coerce")
        if pd.isna(value):
            warnings.append("value missing or not numeric")
            if row.get("extraction_method") not in SOFT_MISSING_VALUE_METHODS:
                critical_error = True
        elif value <= 0 and row.get("indicator") not in CHANGE_INDICATORS:
            warnings.append("value <= 0")

        unit = row.get("unit")
        if not is_missing(unit) and unit not in ALLOWED_UNITS:
            warnings.append("unknown unit")

        currency = row.get("currency")
        if not is_missing(currency) and currency not in ALLOWED_CURRENCIES:
            warnings.append("unknown currency")

        if len(duplicate_rows) > 0 and bool(duplicate_rows.loc[index]):
            warnings.append("duplicate source/date/commodity/indicator")

        if is_missing(row.get("evidence_text")):
            warnings.append("missing evidence_text")

        if critical_error:
            validation_status = "failed"
        elif warnings:
            validation_status = "warning"
        else:
            validation_status = "passed"

        confidence_value = pd.to_numeric(row.get("confidence"), errors="coerce")
        confidence = None if pd.isna(confidence_value) else float(confidence_value)

        statuses.append(validation_status)
        warning_values.append("; ".join(warnings))
        review_statuses.append(_review_status(validation_status, confidence))

    result["validation_status"] = statuses
    result["warnings"] = warning_values
    result["review_status"] = review_statuses
    return result
