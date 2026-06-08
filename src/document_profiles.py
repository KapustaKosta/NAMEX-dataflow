from __future__ import annotations

from typing import Any


PROFILE_PARSER_CONFIDENCE_THRESHOLD = 0.85

FISH_MARKET_STRONG_MARKERS = [
    "Рыбные ряды",
    "НАЦРЫБРЕСУРС",
    "НЦБРП",
    "nacrybresurs",
    "Обзор ситуации на рынке рыбы",
    "Рыбные ряды. Обзор ситуации на рынке рыбы",
]

FISH_MARKET_WEAK_MARKERS = [
    "рыба",
    "рыбы",
    "рыбной продукции",
    "рынок рыбы",
    "ribi",
]


def _matching_markers(haystack: str, markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker.casefold() in haystack]


def has_strong_fish_market_markers(text: str, file_name: str = "") -> bool:
    """Return True when text or file name contains profile-specific fish report markers."""
    haystack = f"{text or ''} {file_name or ''}".casefold()
    return bool(_matching_markers(haystack, FISH_MARKET_STRONG_MARKERS))


def detect_document_profile_with_confidence(text: str, file_name: str) -> dict[str, Any]:
    """Detect document profile with confidence and an explanation for the UI/debug output."""
    haystack = f"{text or ''} {file_name or ''}".casefold()
    strong_markers = _matching_markers(haystack, FISH_MARKET_STRONG_MARKERS)
    weak_markers = _matching_markers(haystack, FISH_MARKET_WEAK_MARKERS)

    if strong_markers:
        confidence = 0.98 if len(strong_markers) >= 2 else 0.9
        return {
            "profile_name": "fish_market_report",
            "profile_confidence": confidence,
            "profile_reason": "Найдены ключевые признаки: " + ", ".join(f'"{marker}"' for marker in strong_markers),
            "matched_markers": strong_markers,
        }

    if weak_markers:
        return {
            "profile_name": "generic_pdf",
            "profile_confidence": 0.35,
            "profile_reason": (
                "Найдены слова про рыбу, но специфические признаки отчёта "
                '"Рыбные ряды" / "НАЦРЫБРЕСУРС" отсутствуют. Используется универсальное извлечение.'
            ),
            "matched_markers": weak_markers,
        }

    return {
        "profile_name": "generic_pdf",
        "profile_confidence": 0.2,
        "profile_reason": "Специфические признаки профильного отчёта не найдены. Используется универсальное извлечение.",
        "matched_markers": [],
    }


def detect_document_profile(text: str, file_name: str) -> str:
    """Backward-compatible wrapper that returns only the profile name."""
    return str(detect_document_profile_with_confidence(text, file_name)["profile_name"])
