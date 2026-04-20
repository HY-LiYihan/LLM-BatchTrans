from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

from .models import Block, ExtractionReport
from .text_utils import chunk_text, clean_page_text


class PDFBlockExtractor:
    def __init__(self, logger: logging.Logger, max_chars: int) -> None:
        self.logger = logger
        self.max_chars = max_chars

    def extract(self, pdf_path: Path) -> ExtractionReport:
        blocks: list[Block] = []
        skipped_pages: list[int] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            self.logger.info("PDF loaded: %s, pages=%s", pdf_path.name, len(pdf.pages))
            for page_index, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned_text = clean_page_text(raw_text)
                if not cleaned_text:
                    skipped_pages.append(page_index)
                    self.logger.warning("Page %s extracted no usable text", page_index)
                    continue

                page_units: list[str] = []
                for part in cleaned_text.split("\n\n"):
                    page_units.extend(chunk_text(part.strip(), self.max_chars))

                for unit_index, unit in enumerate(page_units, start=1):
                    normalized = unit.strip()
                    if not normalized:
                        continue
                    blocks.append(
                        Block(
                            page=page_index,
                            block_id=f"P{page_index:03d}-B{unit_index:03d}",
                            source_text=normalized,
                            source_word_count=len(normalized.split()),
                            source_char_count=len(normalized),
                        )
                    )

        self.logger.info("Extracted %s text blocks from PDF", len(blocks))
        return ExtractionReport(blocks=blocks, skipped_pages=skipped_pages)
