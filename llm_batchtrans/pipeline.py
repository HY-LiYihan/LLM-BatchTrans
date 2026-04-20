from __future__ import annotations

import logging
from pathlib import Path

from .config import PipelineConfig
from .exporters import ArtifactExporter
from .models import PipelineRunResult
from .moonshot_client import MoonshotTranslatorClient
from .pdf_extractor import PDFBlockExtractor
from .translation_service import TranslationOrchestrator


class AssignmentPipeline:
    def __init__(self, config: PipelineConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.extractor = PDFBlockExtractor(logger, config.chunk_size_chars)
        self.translator_client = MoonshotTranslatorClient(config, logger)
        self.translation_service = TranslationOrchestrator(
            self.translator_client,
            logger,
            config.max_workers,
        )
        self.exporter = ArtifactExporter(logger)

    def run(
        self,
        *,
        pdf_path: Path,
        output_dir: Path,
        log_path: Path,
        limit_blocks: int = 0,
    ) -> PipelineRunResult:
        extraction_report = self.extractor.extract(pdf_path)
        blocks = extraction_report.blocks
        if limit_blocks > 0:
            blocks = blocks[:limit_blocks]
            extraction_report.blocks = blocks
            self.logger.info("Fast test mode enabled: limit_blocks=%s", limit_blocks)

        translations = self.translation_service.translate(blocks)
        artifacts = self.exporter.export(
            extraction_report=extraction_report,
            translations=translations,
            output_dir=output_dir,
            pdf_path=pdf_path,
            config=self.config,
        )
        return PipelineRunResult(
            output_dir=output_dir,
            log_path=log_path,
            extraction_report=extraction_report,
            translations=translations,
            artifacts=artifacts,
        )
