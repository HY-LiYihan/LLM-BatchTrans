from __future__ import annotations

from .models import TranslationResult
from .text_utils import ascii_letter_count, chinese_char_count


def find_english_heavy_results(results: list[TranslationResult]) -> list[TranslationResult]:
    flagged: list[TranslationResult] = []
    for item in results:
        if not item.translation:
            continue
        if ascii_letter_count(item.translation) > chinese_char_count(item.translation):
            flagged.append(item)
    return flagged


def build_review_markdown(results: list[TranslationResult]) -> str:
    failures = [item for item in results if item.status != "ok"]
    english_heavy = find_english_heavy_results(results)

    lines = [
        "# 自检报告",
        "",
        f"- 总块数：{len(results)}",
        f"- 成功块数：{len(results) - len(failures)}",
        f"- 失败块数：{len(failures)}",
        f"- 中文占比可疑块数：{len(english_heavy)}",
        "",
        "## 失败块",
    ]

    if failures:
        for item in failures:
            lines.append(f"- {item.block_id} (page {item.page}): {item.error_message or 'unknown error'}")
    else:
        lines.append("- 无")

    lines.extend(["", "## 中文占比可疑块"])
    if english_heavy:
        for item in english_heavy[:20]:
            lines.append(f"- {item.block_id} (page {item.page})")
    else:
        lines.append("- 无")

    return "\n".join(lines)
