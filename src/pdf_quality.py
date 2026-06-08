from __future__ import annotations

import re


CID_PATTERN = re.compile(r"\(cid:\d+\)")
READABLE_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]")


def detect_bad_text_layer(text: str) -> dict[str, object]:
    """Detect broken PDF text layers that produce CID tokens or unreadable glyph text."""
    text = str(text or "")
    if not text.strip():
        return {
            "bad_text_layer": False,
            "reason": "text layer is empty",
            "cid_ratio": 0.0,
            "readable_ratio": 0.0,
        }

    cid_count = len(CID_PATTERN.findall(text))
    text_length = max(len(text), 1)
    cid_ratio = cid_count * len("(cid:0000)") / text_length

    non_space_chars = [char for char in text if not char.isspace()]
    readable_chars = READABLE_PATTERN.findall(text)
    readable_ratio = len(readable_chars) / max(len(non_space_chars), 1)

    technical_chars = sum(1 for char in non_space_chars if char.isdigit() or char in "()[]{}:;,.+-_/|")
    technical_ratio = technical_chars / max(len(non_space_chars), 1)

    reasons: list[str] = []
    if cid_count > 20:
        reasons.append(f"many CID tokens: {cid_count}")
    if cid_ratio > 0.05:
        reasons.append(f"high CID ratio: {cid_ratio:.2f}")
    if readable_ratio < 0.3:
        reasons.append(f"low readable ratio: {readable_ratio:.2f}")
    if technical_ratio > 0.75 and readable_ratio < 0.4:
        reasons.append("mostly technical tokens")

    return {
        "bad_text_layer": bool(reasons),
        "reason": "; ".join(reasons) if reasons else "text layer looks readable",
        "cid_ratio": round(cid_ratio, 4),
        "readable_ratio": round(readable_ratio, 4),
    }

