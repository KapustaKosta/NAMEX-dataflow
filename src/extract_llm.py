from __future__ import annotations


def build_extraction_prompt(text: str) -> str:
    """Build a strict JSON extraction prompt for a future LLM fallback."""
    return (
        "Extract structured data from the text below. Return strict JSON only. "
        "The JSON object must contain these keys: date, commodity, indicator, "
        "value, unit, currency, evidence_text.\n\n"
        f"Text:\n{text}"
    )


def extract_with_llm_mock(text: str) -> dict:
    """Return a mock LLM response without calling any external API."""
    return {
        "date": None,
        "commodity": None,
        "indicator": None,
        "value": None,
        "unit": None,
        "currency": None,
        "evidence_text": text[:500] if text else None,
    }
