from __future__ import annotations

from io import BytesIO

import pandas as pd


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export a DataFrame to an in-memory XLSX file."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="clean_data")
    return buffer.getvalue()


def export_to_csv(df: pd.DataFrame) -> bytes:
    """Export a DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8-sig")
