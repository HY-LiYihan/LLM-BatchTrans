from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .models import Block, TranslationResult, TranslationTerm


class TranslationMemory:
    def __init__(self, cache_path: Path, logger: logging.Logger, glossary_hint_limit: int) -> None:
        self.cache_path = cache_path
        self.logger = logger
        self.glossary_hint_limit = glossary_hint_limit
        self.lock = threading.Lock()
        self.entries_by_hash: dict[str, TranslationResult] = {}
        self.term_frequencies: Counter[tuple[str, str]] = Counter()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def lookup(self, block: Block) -> TranslationResult | None:
        key = self._hash_source(block.source_text)
        with self.lock:
            cached = self.entries_by_hash.get(key)
            if cached is None or cached.source_text != block.source_text:
                return None
            return TranslationResult(
                page=block.page,
                block_id=block.block_id,
                source_text=block.source_text,
                translation=cached.translation,
                term_pairs=list(cached.term_pairs),
                attempts=0,
                elapsed_seconds=0.0,
                status="ok",
                error_message="",
                cache_hit=True,
            )

    def suggest_terms(self, block_text: str) -> list[TranslationTerm]:
        text_lower = block_text.lower()
        suggestions: list[tuple[int, int, TranslationTerm]] = []
        with self.lock:
            for (source, target), frequency in self.term_frequencies.items():
                if source.lower() not in text_lower:
                    continue
                suggestions.append(
                    (
                        frequency,
                        len(source),
                        TranslationTerm(source=source, target=target),
                    )
                )

        suggestions.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in suggestions[: self.glossary_hint_limit]]

    def record(self, result: TranslationResult) -> None:
        if result.status != "ok":
            return

        key = self._hash_source(result.source_text)
        with self.lock:
            existing = self.entries_by_hash.get(key)
            if existing and existing.source_text == result.source_text:
                if existing.translation == result.translation and existing.term_pairs == result.term_pairs:
                    return

            self.entries_by_hash[key] = result
            for pair in result.term_pairs:
                self.term_frequencies[(pair.source, pair.target)] += 1

            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "source_hash": key,
                            "result": asdict(result),
                        },
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")

    def _load(self) -> None:
        if not self.cache_path.exists():
            self.logger.info("Translation memory initialized: %s (new)", self.cache_path)
            return

        loaded_count = 0
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                result = self._result_from_json(payload["result"])
                self.entries_by_hash[payload["source_hash"]] = result
                loaded_count += 1

        self.term_frequencies.clear()
        for result in self.entries_by_hash.values():
            for pair in result.term_pairs:
                self.term_frequencies[(pair.source, pair.target)] += 1

        self.logger.info(
            "Translation memory loaded: %s entries=%s unique_terms=%s",
            self.cache_path,
            len(self.entries_by_hash),
            len(self.term_frequencies),
        )

    def _result_from_json(self, payload: dict) -> TranslationResult:
        return TranslationResult(
            page=int(payload["page"]),
            block_id=str(payload["block_id"]),
            source_text=str(payload["source_text"]),
            translation=str(payload["translation"]),
            term_pairs=[
                TranslationTerm(source=str(item["source"]), target=str(item["target"]))
                for item in payload.get("term_pairs", [])
            ],
            attempts=int(payload.get("attempts", 0)),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            status=str(payload.get("status", "ok")),
            error_message=str(payload.get("error_message", "")),
            cache_hit=bool(payload.get("cache_hit", False)),
        )

    def _hash_source(self, source_text: str) -> str:
        return hashlib.sha1(source_text.encode("utf-8")).hexdigest()
