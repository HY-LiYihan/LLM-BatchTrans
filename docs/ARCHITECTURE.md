# Architecture

## Overview

The repository now has two tracks:

- Legacy CSV-based scripts for industrial material translation:
  - `data_process.py`
  - `validation.py`
- A layered PDF assignment pipeline for academic/technical documents:
  - `pdf_assignment_pipeline.py`
  - `llm_batchtrans/`

The PDF pipeline is organized by responsibility instead of keeping extraction, networking, export, and review logic inside one file.

## Layered Design

```mermaid
flowchart LR
    A["CLI Entry<br/>pdf_assignment_pipeline.py"] --> B["Config Layer<br/>config.py"]
    B --> C["Pipeline Orchestrator<br/>pipeline.py"]
    C --> D["PDF Extraction Layer<br/>pdf_extractor.py"]
    C --> E["Translation Orchestration Layer<br/>translation_service.py"]
    E --> F["Moonshot Client Layer<br/>moonshot_client.py"]
    E --> G["Translation Memory Layer<br/>translation_memory.py"]
    F --> H["Prompt Layer<br/>prompting.py"]
    C --> I["Export Layer<br/>exporters.py"]
    C --> J["Review Layer<br/>review.py"]
    D --> K["Domain Models<br/>models.py"]
    E --> K
    G --> K
    I --> K
    J --> K
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `llm_batchtrans/config.py` | Load `.env`, define runtime settings, mask secrets for exported config metadata |
| `llm_batchtrans/models.py` | Domain dataclasses for blocks, translation results, extraction reports, and artifacts |
| `llm_batchtrans/logging_utils.py` | Console + file logging bootstrap |
| `llm_batchtrans/text_utils.py` | Page cleanup, chunking, term normalization, simple heuristics |
| `llm_batchtrans/pdf_extractor.py` | Convert PDF pages into ordered text blocks |
| `llm_batchtrans/prompting.py` | Build the translation prompt contract |
| `llm_batchtrans/moonshot_client.py` | Session reuse, global request pacing, retries, JSON parsing |
| `llm_batchtrans/translation_memory.py` | Exact-match cache plus glossary-style term reuse across runs |
| `llm_batchtrans/translation_service.py` | Parallel block scheduling and progress logging |
| `llm_batchtrans/review.py` | Self-review heuristics and report generation |
| `llm_batchtrans/exporters.py` | Write DOCX/XLSX/CSV/JSON/Markdown artifacts |
| `llm_batchtrans/pipeline.py` | End-to-end orchestration |
| `llm_batchtrans/cli.py` | CLI argument parsing and top-level runtime wiring |

## Why This Structure

- Extraction, LLM I/O, export, and review have different failure modes and evolve independently.
- The Moonshot client is isolated, so changing providers later does not require touching the exporter or extractor.
- The orchestration layer stays readable because it only coordinates the layers below it.
- Logging stays centralized and every run produces a durable audit trail.
- Translation memory is isolated from the client, so cache, glossary hints, and future TMX-style export can evolve separately.

## Throughput and Stability

The pipeline supports high concurrency, but it now also includes a global request pacing layer.

- `MAX_WORKERS` controls thread-level parallelism.
- `RPM_LIMIT` controls request launch rate.
- Translation tasks are submitted through a bounded in-flight window instead of a full eager fan-out.
- Successful results are appended into a reusable translation memory cache.
- Cached term pairs can be fed back into later prompts as glossary hints.
- Retries back off automatically on transient failures and `429`.

This avoids the “burst all requests immediately” behavior that caused early throttling in the first version.

## External Inspirations

The current design borrows proven ideas from several established tools and projects:

- [DocuTranslate](https://github.com/xunbu/docutranslate): workflow-oriented document translation, glossary handling, and structured exports.
- [Aphra](https://github.com/DavidLMS/aphra): multi-stage translation workflow with explicit quality-oriented orchestration.
- [OmegaT](https://omegat.org/): translation memory, glossary assistance, and match propagation ideas from CAT tooling.
- [Weblate](https://docs.weblate.org/): project-oriented glossary and translation suggestion workflows.
- [Bitextor](https://github.com/bitextor/bitextor): configuration-driven pipelines and keeping intermediate artifacts visible.

The repository does not reuse code from those projects here. The value taken from them is architectural: isolate concerns, persist useful intermediate knowledge, and make repeated translation runs cheaper and more consistent.
