from __future__ import annotations

import re

import pandas as pd

from .pdf_quality import detect_bad_text_layer


USEFUL_TABLE_KEYWORDS = [
    "экспорт",
    "импорт",
    "товарооборот",
    "млн",
    "тыс",
    "долл",
    "тонн",
    "цена",
    "стоимость",
    "объем",
    "объём",
    "динамика",
    "год",
    "период",
]

CONTENTS_KEYWORDS = [
    "содержание",
    "оглавление",
]

NUMBER_PATTERN = re.compile(r"\d")


def _clean_row(row: object) -> str:
    return re.sub(r"\s+", " ", str(row or "")).strip()


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.split("|") if cell.strip()]


def _is_page_number_row(row: str) -> bool:
    cells = _cells(row) or [row.strip()]
    if not cells:
        return False
    short_numeric_cells = 0
    for cell in cells:
        normalized = re.sub(r"[^0-9]", "", cell)
        if normalized and len(normalized) <= 3 and normalized == cell.strip().strip("."):
            short_numeric_cells += 1
    return short_numeric_cells == len(cells)


def _column_count(rows: list[str]) -> int:
    if not rows:
        return 0
    return max((len(_cells(row)) for row in rows), default=0)


def _preview(rows: list[str], limit: int = 3) -> str:
    return "\n".join(rows[:limit])


def score_raw_table(table_rows: list[str]) -> dict[str, object]:
    """Score whether raw PDF table rows are useful for creating a source profile."""
    rows = [_clean_row(row) for row in table_rows]
    rows = [row for row in rows if row]
    row_count = len(rows)
    if not rows:
        return {"table_score": 0.0, "table_reason": "пустая таблица"}

    text = "\n".join(rows).casefold()
    text_layer_quality = detect_bad_text_layer(text)
    if bool(text_layer_quality["bad_text_layer"]):
        return {
            "table_score": 0.3,
            "table_reason": "таблица содержит технические CID-токены, требуется OCR",
            "text_layer_quality": "bad",
            "text_layer_warning": "PDF text layer contains many CID tokens; OCR is recommended",
        }

    column_count = _column_count(rows)
    separator_rows = sum(1 for row in rows if "|" in row and len(_cells(row)) >= 2)
    numeric_rows = sum(1 for row in rows if NUMBER_PATTERN.search(row))
    short_rows = sum(1 for row in rows if len(row) <= 20)
    page_number_rows = sum(1 for row in rows if _is_page_number_row(row))
    keyword_matches = [keyword for keyword in USEFUL_TABLE_KEYWORDS if keyword in text]
    contents_matches = [keyword for keyword in CONTENTS_KEYWORDS if keyword in text]

    score = 0.0
    positive_reasons: list[str] = []
    negative_reasons: list[str] = []

    if row_count >= 3:
        score += 0.18
        positive_reasons.append("несколько строк")
    if row_count >= 8:
        score += 0.08
        positive_reasons.append("много строк")
    if numeric_rows:
        score += min(0.28, 0.08 + 0.04 * numeric_rows)
        positive_reasons.append("есть числовые значения")
    if separator_rows:
        score += min(0.2, 0.05 + 0.03 * separator_rows)
        positive_reasons.append("есть разделители колонок")
    if column_count >= 3:
        score += 0.14
        positive_reasons.append(f"колонок: {column_count}")
    if keyword_matches:
        score += min(0.24, 0.08 + 0.04 * len(keyword_matches))
        positive_reasons.append("слова: " + ", ".join(keyword_matches[:5]))

    if contents_matches:
        score -= 0.25
        negative_reasons.append("похоже на содержание")
    if row_count and short_rows / row_count >= 0.7:
        score -= 0.18
        negative_reasons.append("много коротких строк")
    if row_count and page_number_rows / row_count >= 0.5:
        score -= 0.3
        negative_reasons.append("похоже на номера страниц")
    if numeric_rows <= 1 and not keyword_matches:
        score -= 0.12
        negative_reasons.append("мало чисел")

    score = max(0.0, min(1.0, round(score, 2)))

    if positive_reasons and negative_reasons:
        reason = "; ".join(positive_reasons) + "; ограничения: " + ", ".join(negative_reasons)
    elif positive_reasons:
        reason = "; ".join(positive_reasons)
    else:
        reason = ", ".join(negative_reasons) or "нет устойчивых табличных признаков"

    return {
        "table_score": score,
        "table_reason": reason,
        "text_layer_quality": "ok",
        "text_layer_warning": "",
    }


def build_raw_table_summary(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw pdfplumber table rows by table_id for profile setup."""
    required_columns = {"table_id", "evidence_text"}
    if raw_rows.empty or not required_columns.issubset(set(raw_rows.columns)):
        return pd.DataFrame(
            columns=[
                "source_file",
                "table_id",
                "page",
                "raw_rows_count",
                "column_count",
                "preview",
                "table_score",
                "table_reason",
                "text_layer_quality",
                "text_layer_warning",
            ]
        )

    table_rows = raw_rows[raw_rows["table_id"].notna()].copy()
    if table_rows.empty:
        return pd.DataFrame(
            columns=[
                "source_file",
                "table_id",
                "page",
                "raw_rows_count",
                "column_count",
                "preview",
                "table_score",
                "table_reason",
                "text_layer_quality",
                "text_layer_warning",
            ]
        )

    summary_rows = []
    sort_columns = [column for column in ["page", "table_id", "row_index_in_table", "row_id"] if column in table_rows.columns]
    if sort_columns:
        table_rows = table_rows.sort_values(sort_columns)

    for table_id, group in table_rows.groupby("table_id", sort=False):
        rows = [_clean_row(value) for value in group["evidence_text"].tolist()]
        rows = [row for row in rows if row]
        score = score_raw_table(rows)
        source_file = group["source_file"].dropna().iloc[0] if "source_file" in group and group["source_file"].notna().any() else None
        page = group["page"].dropna().iloc[0] if "page" in group and group["page"].notna().any() else None
        summary_rows.append(
            {
                "source_file": source_file,
                "table_id": table_id,
                "page": page,
                "raw_rows_count": len(rows),
                "column_count": _column_count(rows),
                "preview": _preview(rows),
                "table_score": score["table_score"],
                "table_reason": score["table_reason"],
                "text_layer_quality": score.get("text_layer_quality"),
                "text_layer_warning": score.get("text_layer_warning"),
            }
        )

    return pd.DataFrame(summary_rows)
