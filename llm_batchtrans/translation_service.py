from __future__ import annotations

import concurrent.futures
import logging
import time

from .models import Block, TranslationResult
from .moonshot_client import MoonshotTranslatorClient


class TranslationOrchestrator:
    def __init__(self, client: MoonshotTranslatorClient, logger: logging.Logger, max_workers: int) -> None:
        self.client = client
        self.logger = logger
        self.max_workers = max_workers

    def translate(self, blocks: list[Block]) -> list[TranslationResult]:
        if not blocks:
            return []

        worker_count = min(self.max_workers, len(blocks))
        self.logger.info(
            "Starting concurrent translation: blocks=%s, workers=%s, model=%s",
            len(blocks),
            worker_count,
            self.client.config.model,
        )

        started_at = time.time()
        completed = 0
        results: list[TranslationResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_block = {
                executor.submit(self.client.translate_block, block): block
                for block in blocks
            }
            for future in concurrent.futures.as_completed(future_to_block):
                result = future.result()
                results.append(result)
                completed += 1
                if completed <= 5 or completed % 10 == 0 or result.status != "ok":
                    self.logger.info(
                        "Progress %s/%s | %s | status=%s | attempts=%s | elapsed=%.2fs",
                        completed,
                        len(blocks),
                        result.block_id,
                        result.status,
                        result.attempts,
                        result.elapsed_seconds,
                    )

        ordered = sorted(results, key=lambda item: (item.page, item.block_id))
        elapsed = time.time() - started_at
        ok_count = sum(1 for item in ordered if item.status == "ok")
        self.logger.info(
            "Translation finished: ok=%s, failed=%s, total_elapsed=%.2fs, throughput=%.2f blocks/s",
            ok_count,
            len(ordered) - ok_count,
            elapsed,
            (len(ordered) / elapsed) if elapsed else 0.0,
        )
        return ordered
