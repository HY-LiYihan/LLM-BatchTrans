from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranslationTerm:
    source: str
    target: str


@dataclass
class Block:
    page: int
    block_id: str
    source_text: str
    source_word_count: int
    source_char_count: int
    extraction_backend: str = "pdfplumber"


@dataclass
class TranslationResult:
    page: int
    block_id: str
    source_text: str
    translation: str
    term_pairs: list[TranslationTerm] = field(default_factory=list)
    attempts: int = 0
    elapsed_seconds: float = 0.0
    status: str = "pending"
    error_message: str = ""
    cache_hit: bool = False


@dataclass
class ExtractionReport:
    blocks: list[Block]
    skipped_pages: list[int] = field(default_factory=list)
    page_backends: dict[int, str] = field(default_factory=dict)


@dataclass
class ArtifactManifest:
    extracted_csv: Path
    extracted_xlsx: Path
    bilingual_csv: Path
    bilingual_xlsx: Path
    glossary_xlsx: Path
    docx: Path
    review: Path
    raw_json: Path
    config_json: Path


@dataclass
class PipelineRunResult:
    output_dir: Path
    log_path: Path
    extraction_report: ExtractionReport
    translations: list[TranslationResult]
    artifacts: ArtifactManifest
