from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    output_root: Path
    api_key: str
    base_url: str
    model: str
    max_workers: int
    max_retries: int
    request_timeout: int
    chunk_size_chars: int
    rpm_limit: int
    retry_base_delay: float

    @classmethod
    def from_env(cls, project_root: Path) -> "PipelineConfig":
        load_dotenv(project_root / ".env")
        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("MOONSHOT_API_KEY 未配置，请在 .env 中填写。")

        return cls(
            project_root=project_root,
            output_root=Path(os.getenv("OUTPUT_ROOT", project_root / "outputs")).resolve(),
            api_key=api_key,
            base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1").rstrip("/"),
            model=os.getenv("MOONSHOT_MODEL", "kimi-k2-thinking-turbo").strip(),
            max_workers=int(os.getenv("MAX_WORKERS", "200")),
            max_retries=int(os.getenv("MAX_RETRIES", "4")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "180")),
            chunk_size_chars=int(os.getenv("CHUNK_SIZE_CHARS", "1800")),
            rpm_limit=int(os.getenv("RPM_LIMIT", "4800")),
            retry_base_delay=float(os.getenv("RETRY_BASE_DELAY", "2.0")),
        )

    def with_overrides(
        self,
        *,
        output_root: Path | None = None,
        max_workers: int | None = None,
        chunk_size_chars: int | None = None,
        rpm_limit: int | None = None,
    ) -> "PipelineConfig":
        updates = {}
        if output_root is not None:
            updates["output_root"] = output_root.resolve()
        if max_workers is not None and max_workers > 0:
            updates["max_workers"] = max_workers
        if chunk_size_chars is not None and chunk_size_chars > 0:
            updates["chunk_size_chars"] = chunk_size_chars
        if rpm_limit is not None and rpm_limit > 0:
            updates["rpm_limit"] = rpm_limit
        return replace(self, **updates)

    def to_safe_dict(self) -> dict:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["output_root"] = str(self.output_root)
        payload["api_key"] = "***masked***"
        return payload
