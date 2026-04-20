from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(text: str) -> bool:
    if len(text) > 120:
        return False
    if re.match(r"^(Chapter|Appendix)\b", text):
        return True
    if re.match(r"^\d+(\.\d+)*\b", text):
        return True
    return False


def clean_page_text(raw_text: str) -> str:
    text = normalize_whitespace(raw_text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines: list[str] = []
    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"\d+\s+[A-Za-z].*", line) and len(line.split()) <= 8:
            continue
        cleaned_lines.append(line)

    merged: list[str] = []
    paragraph: list[str] = []
    for line in cleaned_lines:
        if not line:
            if paragraph:
                merged.append(" ".join(paragraph))
                paragraph = []
            continue
        if is_heading(line):
            if paragraph:
                merged.append(" ".join(paragraph))
                paragraph = []
            merged.append(line)
            continue
        paragraph.append(line)
    if paragraph:
        merged.append(" ".join(paragraph))
    return "\n\n".join(part.strip() for part in merged if part.strip())


def split_into_sentences(text: str) -> list[str]:
    compact = text.replace("\n", " ").strip()
    if not compact:
        return []
    parts = re.split(r"(?<=[\.\?\!;:])\s+(?=[A-Z0-9\"“‘(])", compact)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in split_into_sentences(text):
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_length = 0
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start:start + max_chars].strip())
            continue

        projected = current_length + len(sentence) + (1 if current else 0)
        if current and projected > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_length = len(sentence)
            continue

        current.append(sentence)
        current_length = projected

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def normalize_term(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 2:
        return ""
    if re.fullmatch(r"[\W_]+", text):
        return ""
    return text


def ascii_letter_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))
