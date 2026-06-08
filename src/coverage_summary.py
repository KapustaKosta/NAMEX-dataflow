from __future__ import annotations

import pandas as pd


FISH_MARKET_SECTIONS = [
    {
        "section_name": "catch_main_species",
        "block_name": "Улов основных видов рыбы",
        "expected_rows": 15,
    },
    {
        "section_name": "wholesale_far_east",
        "block_name": "Оптовые цены / Дальний Восток",
        "expected_rows": 12,
    },
    {
        "section_name": "wholesale_north_west",
        "block_name": "Оптовые цены / Северо-Запад",
        "expected_rows": 12,
    },
    {
        "section_name": "wholesale_center",
        "block_name": "Оптовые цены / Центр",
        "expected_rows": 12,
    },
    {
        "section_name": "export_market_prices",
        "block_name": "Цены на внешних рынках сбыта",
        "expected_rows": 12,
    },
    {
        "section_name": "retail_frozen_fish",
        "block_name": "Розничные цены на мороженую рыбу",
        "expected_rows": 12,
    },
]


def infer_document_profile_from_rows(df: pd.DataFrame) -> str:
    """Infer a profile from extracted rows for coverage reporting."""
    if "extraction_method" in df.columns and (df["extraction_method"] == "fish_market_report_parser").any():
        return "fish_market_report"
    return "generic"


def _generic_section_specs(df: pd.DataFrame) -> list[dict[str, object]]:
    if "section_name" not in df.columns:
        return []

    section_names = [
        section_name
        for section_name in df["section_name"].dropna().unique().tolist()
        if str(section_name).strip()
    ]
    return [
        {
            "section_name": section_name,
            "block_name": str(section_name),
            "expected_rows": None,
        }
        for section_name in section_names
    ]


def build_coverage_summary(df: pd.DataFrame, profile: str | None = None) -> pd.DataFrame:
    """Build a section-level document coverage summary from validated rows."""
    if df.empty:
        profile = profile or "generic"
    else:
        profile = profile or infer_document_profile_from_rows(df)

    if profile == "generic_pdf":
        return pd.DataFrame()

    section_specs = FISH_MARKET_SECTIONS if profile == "fish_market_report" else _generic_section_specs(df)
    rows = []

    for spec in section_specs:
        section_name = spec["section_name"]
        section_df = df[df.get("section_name").eq(section_name)] if "section_name" in df.columns else pd.DataFrame()
        actual_rows = int(len(section_df))
        warning_rows = int((section_df.get("validation_status") == "warning").sum()) if not section_df.empty else 0
        error_rows = int((section_df.get("validation_status") == "failed").sum()) if not section_df.empty else 0

        rows.append(
            {
                "section_name": section_name,
                "block_name": spec["block_name"],
                "found": actual_rows > 0,
                "expected_rows": spec["expected_rows"],
                "actual_rows": actual_rows,
                "warning_rows": warning_rows,
                "error_rows": error_rows,
                "issue_rows": warning_rows + error_rows,
            }
        )

    return pd.DataFrame(rows)


def coverage_counts(coverage_df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """Return found count, total count, and missing block names."""
    if coverage_df.empty:
        return 0, 0, []

    found_count = int(coverage_df["found"].sum())
    total_count = int(len(coverage_df))
    missing_blocks = coverage_df.loc[~coverage_df["found"], "block_name"].tolist()
    return found_count, total_count, missing_blocks
