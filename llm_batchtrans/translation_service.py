from __future__ import annotations

import concurrent.futures
import logging
import time

from .models import Block, TranslationResult
from .moonshot_client import MoonshotTranslatorClient
from .translation_memory import TranslationMemory


class TranslationOrchestrator:
    def __init__(self, client: MoonshotTranslatorClient, logger: logging.Logger, max_workers: int) -> None:
        self.client = client
        self.logger = logger
        self.max_workers = max_workers

    def translate(self, blocks: list[Block], translation_memory: TranslationMemory | None = None) -> list[TranslationResult]:
        if not blocks:
            return []

        worker_count = min(self.max_workers, len(blocks))
        pending_blocks: list[Block] = []
        cache_hits = 0
        results: list[TranslationResult] = []

        if translation_memory is not None:
            for block in blocks:
                cached = translation_memory.lookup(block)
                if cached is None:
                    pending_blocks.append(block)
                    continue
                results.append(cached)
                cache_hits += 1
                if cache_hits <= 5 or cache_hits % 10 == 0:
                    self.logger.info(
                        "Cache hit %s/%s | %s",
                        cache_hits,
                        len(blocks),
                        block.block_id,
                    )
        else:
            pending_blocks = list(blocks)

        self.logger.info(
            "Starting concurrent translation: blocks=%s, uncached=%s, cache_hits=%s, workers=%s, model=%s",
            len(blocks),
            len(pending_blocks),
            cache_hits,
            worker_count,
            self.client.config.model,
        )

        if not pending_blocks:
            return sorted(results, key=lambda item: (item.page, item.block_id))

        started_at = time.time()
        completed = len(results)
        max_in_flight = max(worker_count * 2, worker_count)
        block_iter = iter(pending_blocks)
        future_to_block: dict[concurrent.futures.Future[TranslationResult], Block] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            while len(future_to_block) < max_in_flight:
                try:
                    block = next(block_iter)
                except StopIteration:
                    break
                future_to_block[executor.submit(self._translate_block, block, translation_memory)] = block

            while future_to_block:
                done, _ = concurrent.futures.wait(
                    future_to_block,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    future_to_block.pop(future)

                    if completed <= 5 or completed % 10 == 0 or result.status != "ok" or result.cache_hit:
                        self.logger.info(
                            "Progress %s/%s | %s | status=%s | cache_hit=%s | attempts=%s | elapsed=%.2fs",
                            completed,
                            len(blocks),
                            result.block_id,
                            result.status,
                            result.cache_hit,
                            result.attempts,
                            result.elapsed_seconds,
                        )

                    if translation_memory is not None:
                        translation_memory.record(result)

                    try:
                        next_block = next(block_iter)
                    except StopIteration:
                        continue
                    future_to_block[
                        executor.submit(self._translate_block, next_block, translation_memory)
                    ] = next_block

        ordered = sorted(results, key=lambda item: (item.page, item.block_id))
        elapsed = time.time() - started_at
        ok_count = sum(1 for item in ordered if item.status == "ok")
        self.logger.info(
            "Translation finished: ok=%s, failed=%s, cache_hits=%s, total_elapsed=%.2fs, throughput=%.2f blocks/s",
            ok_count,
            len(ordered) - ok_count,
            cache_hits,
            elapsed,
            (len(ordered) / elapsed) if elapsed else 0.0,
        )
        return ordered

    def _translate_block(
        self,
        block: Block,
        translation_memory: TranslationMemory | None,
    ) -> TranslationResult:
        preferred_terms = []
        if translation_memory is not None:
            preferred_terms = translation_memory.suggest_terms(block.source_text)
        return self.client.translate_block(block, preferred_terms)
