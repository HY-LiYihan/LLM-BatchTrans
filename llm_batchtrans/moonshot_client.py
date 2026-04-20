from __future__ import annotations

import json
import logging
import re
import threading
import time

import requests

from .config import PipelineConfig
from .models import Block, TranslationResult, TranslationTerm
from .prompting import build_translation_messages


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: float, message: str) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class FixedIntervalRateLimiter:
    def __init__(self, rpm_limit: int) -> None:
        self.interval_seconds = 60.0 / rpm_limit if rpm_limit > 0 else 0.0
        self.lock = threading.Lock()
        self.next_slot_at = time.monotonic()

    def acquire(self) -> None:
        if self.interval_seconds <= 0:
            return

        while True:
            with self.lock:
                now = time.monotonic()
                if now >= self.next_slot_at:
                    self.next_slot_at = now + self.interval_seconds
                    return
                wait_seconds = self.next_slot_at - now
            time.sleep(wait_seconds)


class MoonshotTranslatorClient:
    def __init__(self, config: PipelineConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.rate_limiter = FixedIntervalRateLimiter(config.rpm_limit)
        self.thread_local = threading.local()

    def translate_block(self, block: Block) -> TranslationResult:
        started_at = time.time()
        last_error = ""

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.rate_limiter.acquire()
                response = self._session().post(
                    f"{self.config.base_url}/chat/completions",
                    json=self._build_payload(block),
                    timeout=self.config.request_timeout,
                )

                if response.status_code == 429:
                    retry_after = self._resolve_retry_after(response, attempt)
                    raise RateLimitExceeded(
                        retry_after_seconds=retry_after,
                        message=f"429 Too Many Requests, retry_after={retry_after:.2f}s",
                    )

                response.raise_for_status()
                parsed = self._parse_response_json(response.json())
                translation = str(parsed.get("translation", "")).strip()
                if not translation:
                    raise ValueError("Empty translation in response JSON")

                return TranslationResult(
                    page=block.page,
                    block_id=block.block_id,
                    source_text=block.source_text,
                    translation=translation,
                    term_pairs=self._parse_terms(parsed.get("terms", [])),
                    attempts=attempt,
                    elapsed_seconds=time.time() - started_at,
                    status="ok",
                )
            except RateLimitExceeded as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                self.logger.warning(
                    "Rate limited for %s on attempt %s/%s: %s",
                    block.block_id,
                    attempt,
                    self.config.max_retries,
                    last_error,
                )
                time.sleep(exc.retry_after_seconds)
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                self.logger.warning(
                    "Translate failed for %s on attempt %s/%s: %s",
                    block.block_id,
                    attempt,
                    self.config.max_retries,
                    last_error,
                )
                time.sleep(min(self.config.retry_base_delay * attempt, 8.0))

        return TranslationResult(
            page=block.page,
            block_id=block.block_id,
            source_text=block.source_text,
            translation="",
            term_pairs=[],
            attempts=self.config.max_retries,
            elapsed_seconds=time.time() - started_at,
            status="failed",
            error_message=last_error,
        )

    def _session(self) -> requests.Session:
        session = getattr(self.thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                }
            )
            self.thread_local.session = session
        return session

    def _build_payload(self, block: Block) -> dict:
        return {
            "model": self.config.model,
            "messages": build_translation_messages(block),
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

    def _parse_response_json(self, payload: dict) -> dict:
        content = payload["choices"][0]["message"]["content"]
        stripped = content.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.S)
        if fenced:
            stripped = fenced.group(1).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start:end + 1]
        return json.loads(stripped)

    def _parse_terms(self, raw_terms: object) -> list[TranslationTerm]:
        term_pairs: list[TranslationTerm] = []
        if not isinstance(raw_terms, list):
            return term_pairs

        for item in raw_terms:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if source and target:
                term_pairs.append(TranslationTerm(source=source, target=target))
        return term_pairs

    def _resolve_retry_after(self, response: requests.Response, attempt: int) -> float:
        header_value = response.headers.get("Retry-After")
        if header_value:
            try:
                return max(float(header_value), self.config.retry_base_delay)
            except ValueError:
                pass
        return min(self.config.retry_base_delay * (2 ** (attempt - 1)), 15.0)
