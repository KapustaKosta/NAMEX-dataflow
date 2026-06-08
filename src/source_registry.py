from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "fish_market_report": {
        "display_name": "Рыбные ряды / НАЦРЫБРЕСУРС",
        "source_type": "pdf",
        "document_profile": "fish_market_report",
        "description": "Еженедельный обзор ситуации на рынке рыбы",
        "expected_sections": [
            "catch_main_species",
            "wholesale_far_east",
            "wholesale_north_west",
            "wholesale_center",
            "export_market_prices",
            "retail_frozen_fish",
        ],
        "target_fields": [
            "section_name",
            "indicator",
            "commodity",
            "region",
            "route",
            "date",
            "value",
            "unit",
            "currency",
            "evidence_text",
        ],
        "update_frequency": "weekly",
        "extraction_strategy": "profile_parser",
        "requires_ocr": False,
        "uses_llm": False,
    },
    "generic_pdf": {
        "display_name": "Универсальный PDF",
        "source_type": "pdf",
        "document_profile": "generic_pdf",
        "description": "Базовое извлечение текста и таблиц из PDF",
        "extraction_strategy": "pdfplumber",
        "requires_ocr": False,
        "uses_llm": False,
    },
    "generic_table": {
        "display_name": "Универсальная таблица CSV/XLSX",
        "source_type": "table",
        "document_profile": "generic_table",
        "description": "Базовая обработка CSV/XLSX с нормализацией и валидацией",
        "extraction_strategy": "pandas",
        "requires_ocr": False,
        "uses_llm": False,
    },
}


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "sources.yaml"


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def _load_yaml_with_fallback(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return _parse_simple_sources_yaml(path)


def _parse_simple_sources_yaml(path: Path) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    current_profile: str | None = None
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_profile = line[:-1].strip()
            registry[current_profile] = {}
            current_list_key = None
            continue

        if current_profile is None:
            continue

        stripped = line.strip()
        if stripped.startswith("- ") and current_list_key:
            registry[current_profile].setdefault(current_list_key, []).append(_coerce_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            continue

        key, value = stripped.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if value == "":
            registry[current_profile][key] = []
            current_list_key = key
        else:
            registry[current_profile][key] = _coerce_scalar(value)
            current_list_key = None

    return registry


def load_source_registry(config_path: str = "configs/sources.yaml") -> dict[str, dict[str, Any]]:
    """Load source registry config without failing if config or YAML support is absent."""
    path = Path(config_path)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        path = cwd_path if cwd_path.exists() else _default_config_path()

    if not path.exists():
        return DEFAULT_SOURCE_REGISTRY.copy()

    registry = _load_yaml_with_fallback(path)
    if not registry:
        return DEFAULT_SOURCE_REGISTRY.copy()

    merged = DEFAULT_SOURCE_REGISTRY.copy()
    merged.update(registry)
    return merged


def get_source_config(profile_name: str) -> dict[str, Any]:
    """Return source config by profile name, falling back to generic profile."""
    registry = load_source_registry()
    return registry.get(profile_name) or registry.get("generic_pdf") or DEFAULT_SOURCE_REGISTRY["generic_pdf"]


def get_display_name(profile_name: str) -> str:
    """Return a user-facing source profile name."""
    config = get_source_config(profile_name)
    return str(config.get("display_name") or profile_name or "Универсальный источник")


def get_expected_sections(profile_name: str) -> list[str]:
    """Return expected section names for a source profile."""
    config = get_source_config(profile_name)
    sections = config.get("expected_sections") or []
    return list(sections) if isinstance(sections, list) else []
