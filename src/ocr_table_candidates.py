from __future__ import annotations

import re

import pandas as pd


OCR_CANDIDATE_COLUMNS = [
    "source_file",
    "page",
    "ocr_block_id",
    "block_title",
    "candidate_type",
    "block_text",
    "preview",
    "rows_count",
    "numbers_count",
    "table_score",
    "information_score",
    "score",
    "reason",
    "extraction_method",
    "extraction_level",
    "review_status",
]

HEADER_KEYWORDS = [
    "товарная структура",
    "основные страны",
    "структура",
    "производство",
    "импорт",
    "экспорт",
    "торговля",
    "2020-2024",
    "2023-2024",
    "млн долл",
    "тыс. тонн",
]

HEADER_REJECT_PREFIXES = [
    "в 2024",
    "за 2024",
    "по сравнению",
    "относительно",
    "при этом",
    "в топ",
    "топ-",
    "одновременно",
    "суммарно",
    "крупнейшим",
    "основным ",
    "основными ",
    "наиболее ",
]

TABLE_KEYWORDS = [
    "товар",
    "страна",
    "страны",
    "культура",
    "импорт",
    "экспорт",
    "производство",
    "структура",
    "торговля",
]

FALLBACK_TABLE_KEYWORDS = [
    "тариф",
    "груз",
    "услуг",
    "тн",
    "руб",
    "сут",
    "контейнер",
    "автотранспорт",
    "металлопрокат",
    "хранение",
    "грузы",
    "генеральные",
    "зерновые",
    "дополнительные",
    "подача",
    "техника",
    "насыпные",
]

SCORE_KEYWORDS = ["импорт", "экспорт", "производство", "структура"]
TABLE_STRUCTURE_KEYWORDS = ["товар", "страна", "страны", "культура", "показатель", "наименование", "значение"]
UNIT_KEYWORDS = ["млн долл", "долл", "тыс. тонн", "тыс тонн", "%"]
YEAR_PATTERN = re.compile(r"\b(?:2020|2021|2022|2023|2024)\b")
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s*%)?(?!\w)")
WORD_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]+")
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]")
PUNCTUATION_PATTERN = re.compile(r"[.,;:!?]")
SPACE_PATTERN = re.compile(r"[ \t]+")


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=OCR_CANDIDATE_COLUMNS)


def _clean_line(line: object) -> str:
    return SPACE_PATTERN.sub(" ", str(line or "")).strip()


def _non_empty_lines(text: str) -> list[str]:
    return [_clean_line(line) for line in text.splitlines() if _clean_line(line)]


def _normalized(text: str) -> str:
    return text.replace("ё", "е").casefold()


def _keyword_matches(text: str, keywords: list[str]) -> list[str]:
    normalized = _normalized(text)
    return [keyword for keyword in keywords if keyword in normalized]


def _numbers_count(text: str) -> int:
    return len(NUMBER_PATTERN.findall(text))


def _line_numbers_count(line: str) -> int:
    return len(NUMBER_PATTERN.findall(line))


def _alpha_count(line: str) -> int:
    return sum(1 for char in line if char.isalpha())


def _word_count(line: str) -> int:
    return len(WORD_PATTERN.findall(line))


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 2)))


def _line_is_structured_numeric(line: str, min_numbers: int = 2) -> bool:
    return _line_numbers_count(line) >= min_numbers and _alpha_count(line) >= 2


def _max_consecutive_structured_numeric_run(lines: list[str], min_numbers: int = 3) -> int:
    best_run = 0
    current_run = 0
    for line in lines:
        number_count = _line_numbers_count(line)
        if _alpha_count(line) >= 2 and min_numbers <= number_count <= 6:
            current_run += 1
            best_run = max(best_run, current_run)
        else:
            current_run = 0
    return best_run


def _has_explicit_table_heading(lines: list[str]) -> bool:
    if not lines:
        return False
    heading = _normalized(lines[0])
    has_period = bool(_keyword_matches(heading, ["2020-2024", "2023-2024"]))
    return (
        "товарная структура" in heading
        or "основные страны" in heading
        or ("производство" in heading and bool(_keyword_matches(heading, UNIT_KEYWORDS)))
        or ("экспорт" in heading and has_period)
        or ("импорт" in heading and has_period)
    )


def detect_paragraph_like_block(block_text: str) -> dict[str, object]:
    """Return text-flow diagnostics for separating prose from real OCR tables."""
    lines = _non_empty_lines(block_text)
    text = "\n".join(lines)
    rows_count = len(lines)
    char_count = max(1, len(text))
    avg_line_length = sum(len(line) for line in lines) / rows_count if rows_count else 0.0
    punctuation_count = len(PUNCTUATION_PATTERN.findall(text))
    punctuation_ratio = punctuation_count / char_count
    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.findall(text)]
    sentence_count = len(sentences)
    long_sentence_count = sum(1 for sentence in sentences if len(sentence) >= 90 or _word_count(sentence) >= 14)
    lines_with_many_words = sum(1 for line in lines if _word_count(line) >= 10)
    lines_with_multiple_numbers = sum(1 for line in lines if _line_numbers_count(line) >= 3)
    structured_line_count = sum(1 for line in lines if _line_is_structured_numeric(line, min_numbers=3))
    structured_line_ratio = structured_line_count / rows_count if rows_count else 0.0

    many_text_lines = rows_count >= 8 and (lines_with_many_words / rows_count) >= 0.40
    sentence_dense = sentence_count >= 2 and (punctuation_ratio >= 0.015 or long_sentence_count >= 1)
    long_line_block = avg_line_length >= 85 and sentence_count >= 1
    short_prose_block = rows_count <= 4 and avg_line_length >= 70 and sentence_count >= 1
    few_structured_rows = structured_line_ratio < 0.35 and lines_with_multiple_numbers < 4

    is_paragraph_like = bool(
        few_structured_rows
        and (
            sentence_dense
            or long_line_block
            or short_prose_block
            or (many_text_lines and sentence_count >= 2)
        )
    )

    return {
        "avg_line_length": avg_line_length,
        "punctuation_ratio": punctuation_ratio,
        "sentence_count": sentence_count,
        "long_sentence_count": long_sentence_count,
        "lines_with_many_words": lines_with_many_words,
        "lines_with_multiple_numbers": lines_with_multiple_numbers,
        "structured_line_ratio": structured_line_ratio,
        "structured_line_count": structured_line_count,
        "is_paragraph_like": is_paragraph_like,
    }


def _is_heading_line(line: str) -> bool:
    cleaned = _clean_line(line)
    if len(cleaned) < 8 or len(cleaned) > 220:
        return False
    if _numbers_count(cleaned) >= 6:
        return False
    if len(cleaned) >= 90 and (re.search(r"[.!?]", cleaned) or cleaned.count(",") >= 2):
        return False
    first_alpha = next((char for char in cleaned if char.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return False
    normalized = _normalized(cleaned)
    if any(normalized.startswith(prefix) for prefix in HEADER_REJECT_PREFIXES):
        return False
    strong_keywords = _keyword_matches(
        cleaned,
        [
            "товарная структура",
            "основные страны",
            "структура",
            "производство",
            "импорт",
            "экспорт",
            "торговля",
        ],
    )
    period_keywords = _keyword_matches(cleaned, ["2020-2024", "2023-2024"])
    return bool(strong_keywords or period_keywords)


def _looks_like_candidate(block_text: str) -> bool:
    lines = _non_empty_lines(block_text)
    if len(lines) < 2:
        return False
    if _numbers_count(block_text) == 0:
        return False
    return bool(_keyword_matches(block_text, TABLE_KEYWORDS))


def _make_preview(block_text: str, max_lines: int = 8, max_chars: int = 900) -> str:
    lines = _non_empty_lines(block_text)[:max_lines]
    preview = "\n".join(lines)
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 1].rstrip() + "…"


def _ocr_candidate_stats(block_text: str) -> dict[str, object]:
    lines = _non_empty_lines(block_text)
    text = "\n".join(lines)
    normalized = _normalized(text)
    numbers_count = _numbers_count(text)
    year_matches = sorted(set(YEAR_PATTERN.findall(text)))
    keyword_matches = [keyword for keyword in SCORE_KEYWORDS if keyword in normalized]
    unit_matches = [keyword for keyword in UNIT_KEYWORDS if keyword in normalized]
    structure_keyword_matches = [keyword for keyword in TABLE_STRUCTURE_KEYWORDS if keyword in normalized]
    numeric_rows = sum(1 for line in lines if NUMBER_PATTERN.search(line))
    separator_rows = sum(1 for line in lines if "|" in line or "\t" in line or re.search(r"\s{2,}", line))
    line_number_counts = [_line_numbers_count(line) for line in lines]
    label_numeric_rows = sum(1 for line in lines if _line_is_structured_numeric(line, min_numbers=2))
    single_number_label_rows = sum(
        1 for line, number_count in zip(lines, line_number_counts) if number_count == 1 and _alpha_count(line) >= 3
    )
    lines_with_3plus_numbers = sum(1 for line in lines if _line_is_structured_numeric(line, min_numbers=3))
    isolated_numeric_rows = sum(
        1
        for line, number_count in zip(lines, line_number_counts)
        if number_count >= 1 and _alpha_count(line) == 0 and len(line) <= 30
    )
    percent_values = len(re.findall(r"\d+(?:[.,]\d+)?\s*%", text))
    repeated_numeric_width = False
    numeric_widths = [number_count for number_count in line_number_counts if number_count >= 2]
    if numeric_widths:
        repeated_numeric_width = max(numeric_widths.count(width) for width in set(numeric_widths)) >= 3
    max_numeric_run = _max_consecutive_structured_numeric_run(lines, min_numbers=3)
    explicit_table_heading = _has_explicit_table_heading(lines)
    numeric_lines_after_heading = sum(1 for line in lines[1:] if _line_is_structured_numeric(line, min_numbers=2))
    has_min_structured_rows = label_numeric_rows >= 4
    has_repeated_numeric_structure = repeated_numeric_width or max_numeric_run >= 3
    has_explicit_table_with_rows = explicit_table_heading and numeric_lines_after_heading >= 2
    sentence_rows = sum(
        1
        for line in lines
        if (len(line) >= 80 and re.search(r"[.!?]", line)) or (len(line) >= 100 and line.count(",") >= 2)
    )
    avg_line_len = sum(len(line) for line in lines) / len(lines) if lines else 0.0
    word_count = len(WORD_PATTERN.findall(text))
    paragraph_metrics = detect_paragraph_like_block(block_text)
    paragraph_like = bool(paragraph_metrics["is_paragraph_like"])
    chart_like = (
        (isolated_numeric_rows >= 3 and label_numeric_rows <= 1)
        or (percent_values >= 4 and label_numeric_rows <= 1 and len(lines) <= 8)
    )
    return {
        "lines": lines,
        "text": text,
        "numbers_count": numbers_count,
        "year_matches": year_matches,
        "keyword_matches": keyword_matches,
        "unit_matches": unit_matches,
        "structure_keyword_matches": structure_keyword_matches,
        "numeric_rows": numeric_rows,
        "separator_rows": separator_rows,
        "label_numeric_rows": label_numeric_rows,
        "single_number_label_rows": single_number_label_rows,
        "lines_with_3plus_numbers": lines_with_3plus_numbers,
        "isolated_numeric_rows": isolated_numeric_rows,
        "percent_values": percent_values,
        "repeated_numeric_width": repeated_numeric_width,
        "max_numeric_run": max_numeric_run,
        "explicit_table_heading": explicit_table_heading,
        "numeric_lines_after_heading": numeric_lines_after_heading,
        "has_min_structured_rows": has_min_structured_rows,
        "has_repeated_numeric_structure": has_repeated_numeric_structure,
        "has_explicit_table_with_rows": has_explicit_table_with_rows,
        "sentence_rows": sentence_rows,
        "avg_line_len": avg_line_len,
        "word_count": word_count,
        "paragraph_metrics": paragraph_metrics,
        "paragraph_like": paragraph_like,
        "chart_like": chart_like,
    }


def _score_information(stats: dict[str, object]) -> float:
    score = 0.0
    numbers_count = int(stats["numbers_count"])
    rows_count = len(stats["lines"])
    word_count = int(stats["word_count"])

    if numbers_count >= 8:
        score += 0.25
    elif numbers_count >= 3:
        score += 0.18
    elif numbers_count >= 1:
        score += 0.08

    if stats["keyword_matches"]:
        score += 0.25
    if stats["year_matches"]:
        score += 0.18
    if stats["unit_matches"]:
        score += 0.14
    if rows_count >= 3:
        score += 0.08
    if word_count >= 25:
        score += 0.12
    elif word_count >= 12:
        score += 0.06
    paragraph_metrics = dict(stats.get("paragraph_metrics") or {})
    if bool(paragraph_metrics.get("is_paragraph_like")):
        if stats["keyword_matches"] or numbers_count:
            score += 0.12
        if int(paragraph_metrics.get("sentence_count") or 0) >= 2:
            score += 0.08

    return _clamp_score(score)


def _score_table(stats: dict[str, object]) -> float:
    score = 0.0
    rows_count = len(stats["lines"])
    label_numeric_rows = int(stats["label_numeric_rows"])
    single_number_label_rows = int(stats["single_number_label_rows"])
    numeric_rows = int(stats["numeric_rows"])

    if label_numeric_rows >= 4:
        score += 0.28
    elif label_numeric_rows >= 2:
        score += 0.14
    elif single_number_label_rows >= 3:
        score += 0.10

    if stats["repeated_numeric_width"]:
        score += 0.18
    if int(stats["max_numeric_run"]) >= 3:
        score += 0.14
    if bool(stats["has_explicit_table_with_rows"]):
        score += 0.25
    year_matches = list(stats["year_matches"])
    if len(year_matches) >= 2:
        score += 0.10
    elif year_matches:
        score += 0.08
    if stats["unit_matches"]:
        score += 0.10
    if stats["structure_keyword_matches"]:
        score += 0.10
    if rows_count >= 5:
        score += 0.06
    elif rows_count >= 3:
        score += 0.03
    if int(stats["separator_rows"]) >= 2:
        score += 0.05
    if numeric_rows >= 3:
        score += 0.05

    if bool(stats["paragraph_like"]):
        has_embedded_table_rows = label_numeric_rows >= 3 or bool(stats["has_repeated_numeric_structure"])
        score = min(score, 0.60 if has_embedded_table_rows else 0.45)
    if label_numeric_rows <= 1 and not bool(stats["repeated_numeric_width"]):
        score -= 0.12
    if bool(stats["chart_like"]):
        score -= 0.12
    if rows_count < 3:
        score -= 0.10

    return _clamp_score(score)


def _classify_candidate(stats: dict[str, object], table_score: float, information_score: float) -> str:
    label_numeric_rows = int(stats["label_numeric_rows"])
    has_table_condition = bool(
        stats["has_min_structured_rows"]
        or stats["has_repeated_numeric_structure"]
        or stats["has_explicit_table_with_rows"]
    )
    if bool(stats["chart_like"]) and table_score < 0.55:
        return "chart_text"
    if bool(stats["paragraph_like"]):
        if has_table_condition and (label_numeric_rows >= 3 or int(stats["numeric_lines_after_heading"]) >= 2):
            return "mixed"
        return "paragraph"
    if has_table_condition and table_score >= 0.55:
        return "table"
    if has_table_condition and label_numeric_rows >= 2:
        return "mixed"
    if information_score >= 0.50 and table_score < 0.45:
        return "paragraph"
    if table_score >= 0.45:
        return "mixed"
    if information_score >= 0.30:
        return "paragraph"
    return "unknown"


def _format_candidate_reason(candidate_type: str, stats: dict[str, object]) -> str:
    reasons: list[str] = []
    label_numeric_rows = int(stats["label_numeric_rows"])
    year_matches = list(stats["year_matches"])
    unit_matches = list(stats["unit_matches"])

    if candidate_type == "table":
        reasons.append("похоже на таблицу")
        if label_numeric_rows:
            reasons.append("несколько строк имеют структуру 'название + несколько чисел'")
        if stats["keyword_matches"]:
            reasons.append("есть ключевые слова: " + ", ".join(list(stats["keyword_matches"])[:4]))
        if bool(stats["repeated_numeric_width"]):
            reasons.append("повторяется число числовых колонок")
        if year_matches:
            reasons.append("есть годы " + ", ".join(year_matches[:5]))
        if unit_matches:
            reasons.append("есть единицы измерения: " + ", ".join(unit_matches[:3]))
    elif candidate_type == "paragraph":
        reasons.append("информативный текстовый блок")
        if stats["keyword_matches"] and int(stats["numbers_count"]):
            reasons.append("много ключевых слов и чисел")
        elif stats["keyword_matches"]:
            reasons.append("есть ключевые слова: " + ", ".join(list(stats["keyword_matches"])[:4]))
        elif int(stats["numbers_count"]):
            reasons.append("есть числа")
        else:
            reasons.append("похоже на обычный абзац")
        reasons.append("строки выглядят как связный текст, а не как повторяющиеся записи таблицы")
    elif candidate_type == "chart_text":
        reasons.append("похоже на OCR текста с графика")
        reasons.append("много отдельных чисел или процентов")
        reasons.append("нет устойчивых строк таблицы")
    elif candidate_type == "mixed":
        reasons.append("смешанный блок")
        if bool(stats["paragraph_like"]):
            reasons.append("есть связный текст")
        if label_numeric_rows:
            reasons.append("есть отдельные строки с числовыми значениями")
    else:
        reasons.append("тип не определён")
        reasons.append("нет устойчивых табличных признаков")

    return ": ".join([reasons[0], ", ".join(reasons[1:])]) if len(reasons) > 1 else reasons[0]


def analyze_ocr_candidate(block_text: str) -> dict[str, object]:
    stats = _ocr_candidate_stats(block_text)
    table_score = _score_table(stats)
    information_score = _score_information(stats)
    candidate_type = _classify_candidate(stats, table_score, information_score)
    score = _clamp_score((table_score * 0.75) + (information_score * 0.25))
    return {
        "candidate_type": candidate_type,
        "table_score": table_score,
        "information_score": information_score,
        "score": score,
        "reason": _format_candidate_reason(candidate_type, stats),
    }


def score_ocr_candidate(block_text: str) -> tuple[float, str]:
    analysis = analyze_ocr_candidate(block_text)
    return float(analysis["score"]), str(analysis["reason"])


def _extract_blocks_from_text(text: str) -> list[tuple[str, str]]:
    raw_lines = [_clean_line(line) for line in str(text or "").splitlines()]
    blocks: list[tuple[str, str]] = []
    index = 0

    while index < len(raw_lines):
        line = raw_lines[index]
        if not _is_heading_line(line):
            index += 1
            continue

        title = line
        block_lines = [line]
        blank_gap = 0
        cursor = index + 1

        while cursor < len(raw_lines):
            next_line = raw_lines[cursor]
            if not next_line:
                blank_gap += 1
                if blank_gap >= 2 and len(_non_empty_lines("\n".join(block_lines))) >= 3:
                    cursor += 1
                    break
                cursor += 1
                continue

            blank_gap = 0
            normalized = _normalized(next_line)
            if normalized.startswith("источник:") or normalized.startswith("источник "):
                break
            if _is_heading_line(next_line):
                if len(_non_empty_lines("\n".join(block_lines))) < 3:
                    title = next_line
                    block_lines = [next_line]
                    cursor += 1
                    continue
                break

            block_lines.append(next_line)
            cursor += 1

        block_text = "\n".join(_non_empty_lines("\n".join(block_lines)))
        if _looks_like_candidate(block_text):
            blocks.append((title, block_text))

        index = max(cursor, index + 1)

    return blocks


def _has_fallback_criteria(text: str) -> bool:
    """Check if raw OCR page has enough data to create fallback candidate."""
    lines = _non_empty_lines(text)
    if not lines or len(lines) < 5:
        return False
    
    text_len = len(text)
    if text_len < 100:
        return False
    
    numbers = _numbers_count(text)
    if numbers < 3:
        return False
    
    has_keywords = any(keyword in _normalized(text) for keyword in FALLBACK_TABLE_KEYWORDS)
    return bool(numbers >= 3 or (len(lines) >= 8 and text_len >= 300) or has_keywords)


def extract_ocr_table_candidates(ocr_df: pd.DataFrame) -> pd.DataFrame:
    if ocr_df.empty or "evidence_text" not in ocr_df.columns:
        return _empty_candidates()

    rows = ocr_df.copy()
    if "extraction_method" in rows.columns:
        rows = rows[rows["extraction_method"].fillna("").astype(str).eq("tesseract_ocr")]
    if "extraction_level" in rows.columns:
        rows = rows[rows["extraction_level"].fillna("").astype(str).eq("raw_ocr")]
    if rows.empty:
        return _empty_candidates()

    sort_columns = [column for column in ["page", "row_id"] if column in rows.columns]
    if sort_columns:
        rows = rows.sort_values(sort_columns)

    candidate_rows = []
    for _, row in rows.iterrows():
        source_file = row.get("source_file")
        page = row.get("page")
        evidence_text = str(row.get("evidence_text") or "")
        
        blocks = _extract_blocks_from_text(evidence_text)
        found_blocks = False
        
        for block_index, (title, block_text) in enumerate(blocks, start=1):
            rows_count = len(_non_empty_lines(block_text))
            numbers_count = _numbers_count(block_text)
            analysis = analyze_ocr_candidate(block_text)
            candidate_rows.append(
                {
                    "source_file": source_file,
                    "page": page,
                    "ocr_block_id": f"ocr_p{page}_b{block_index}",
                    "block_title": title,
                    "candidate_type": analysis["candidate_type"],
                    "block_text": block_text,
                    "preview": _make_preview(block_text),
                    "rows_count": rows_count,
                    "numbers_count": numbers_count,
                    "table_score": analysis["table_score"],
                    "information_score": analysis["information_score"],
                    "score": analysis["score"],
                    "reason": analysis["reason"],
                    "extraction_method": "tesseract_ocr_candidate",
                    "extraction_level": "ocr_candidate",
                    "review_status": "needs_profile_setup",
                }
            )
            found_blocks = True
        
        if not found_blocks and _has_fallback_criteria(evidence_text):
            lines = _non_empty_lines(evidence_text)
            text = "\n".join(lines)
            rows_count = len(lines)
            numbers_count = _numbers_count(evidence_text)
            fallback_score = _clamp_score(min(0.4, 0.1 * min(numbers_count / 5.0, 1.0) + 0.2))
            
            candidate_rows.append(
                {
                    "source_file": source_file,
                    "page": page,
                    "ocr_block_id": f"ocr_p{page}_fallback",
                    "block_title": f"OCR стр. {page} (fallback)",
                    "candidate_type": "paragraph",
                    "block_text": text,
                    "preview": _make_preview(evidence_text),
                    "rows_count": rows_count,
                    "numbers_count": numbers_count,
                    "table_score": 0.1,
                    "information_score": 0.4,
                    "score": fallback_score,
                    "reason": "fallback: недостаточно структуры для выделения блоков, но текст содержит достаточно данных",
                    "extraction_method": "tesseract_ocr_candidate_fallback",
                    "extraction_level": "ocr_candidate",
                    "review_status": "needs_profile_setup",
                }
            )

    if not candidate_rows:
        return _empty_candidates()

    return pd.DataFrame(candidate_rows, columns=OCR_CANDIDATE_COLUMNS)

