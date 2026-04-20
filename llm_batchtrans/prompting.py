from __future__ import annotations

from .models import Block


def build_translation_messages(block: Block) -> list[dict]:
    system_prompt = (
        "你是一名严谨的学术译者和术语整理助手。"
        "你的任务是把英文技术或学术文本译成自然、准确、完整的中文，并抽取英中术语。"
        "不要遗漏内容，不要添加解释，不要输出思考过程。"
        "保留原文中的编号、年份、人名、引文、括号、章节号等关键信息。"
        "输出必须是 JSON 对象，格式如下："
        '{"translation":"中文译文","terms":[{"source":"英文术语","target":"中文术语"}]}。'
        "terms 最多保留 8 项，只保留真正有价值的术语。"
    )
    user_prompt = (
        f"块编号：{block.block_id}\n"
        f"页码：{block.page}\n"
        "请翻译下面的英文文本，并抽取术语：\n"
        f"{block.source_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
