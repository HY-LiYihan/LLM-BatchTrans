from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import PipelineConfig
from .logging_utils import configure_logging
from .pipeline import AssignmentPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a PDF into Chinese and export DOCX/XLSX assignment deliverables."
    )
    parser.add_argument("--pdf", default="200-228.pdf", help="PDF file to process")
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional output root directory; defaults to OUTPUT_ROOT or ./outputs",
    )
    parser.add_argument(
        "--limit-blocks",
        type=int,
        default=0,
        help="Only process the first N blocks for a smoke test",
    )
    parser.add_argument("--max-workers", type=int, default=0, help="Override MAX_WORKERS from .env")
    parser.add_argument(
        "--chunk-size-chars",
        type=int,
        default=0,
        help="Override CHUNK_SIZE_CHARS from .env",
    )
    parser.add_argument("--rpm-limit", type=int, default=0, help="Override RPM_LIMIT from .env")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = PipelineConfig.from_env(PROJECT_ROOT).with_overrides(
        output_root=Path(args.output_root) if args.output_root else None,
        max_workers=args.max_workers or None,
        chunk_size_chars=args.chunk_size_chars or None,
        rpm_limit=args.rpm_limit or None,
    )

    pdf_path = (PROJECT_ROOT / args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    run_name = f"{pdf_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = config.output_root / run_name
    logger, log_path = configure_logging(output_dir)

    logger.info("Run started")
    logger.info("Output directory: %s", output_dir)
    logger.info("Log file: %s", log_path)
    logger.info(
        "Config summary: model=%s, max_workers=%s, rpm_limit=%s, chunk_size_chars=%s",
        config.model,
        config.max_workers,
        config.rpm_limit,
        config.chunk_size_chars,
    )

    pipeline = AssignmentPipeline(config, logger)
    run_result = pipeline.run(
        pdf_path=pdf_path,
        output_dir=output_dir,
        log_path=log_path,
        limit_blocks=args.limit_blocks,
    )

    failures = [item for item in run_result.translations if item.status != "ok"]
    if failures:
        logger.warning("Run finished with %s failed blocks", len(failures))
        return 2

    logger.info("Run finished successfully")
    return 0
