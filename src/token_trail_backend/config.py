"""Fixed model identity and bounded service configuration."""

import os
from dataclasses import dataclass

MODEL_ALIAS = "qwen2.5-0.5b"
MODEL_REPO = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MAX_PROMPT_CHARS = 1000
MAX_PROMPT_TOKENS = 256
MAX_NEW_TOKENS = 32
MAX_TOP_K = 10
MAX_BODY_BYTES = 4096


@dataclass(frozen=True)
class Settings:
    model_alias: str = MODEL_ALIAS
    model_repo: str = MODEL_REPO
    model_revision: str = MODEL_REVISION
    request_timeout_seconds: float = 30.0
    rate_limit_per_minute: int = 12
    origins: tuple[str, ...] = (
        "https://crunchycodes.net",
        "https://www.crunchycodes.net",
        "https://ozyjay.github.io",
    )

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            item.strip().rstrip("/")
            for item in os.getenv("TOKEN_TRAIL_ALLOWED_ORIGINS", ",".join(cls.origins)).split(",")
            if item.strip()
        )
        return cls(
            request_timeout_seconds=float(os.getenv("TOKEN_TRAIL_TIMEOUT_SECONDS", "30")),
            rate_limit_per_minute=int(os.getenv("TOKEN_TRAIL_RATE_LIMIT_PER_MINUTE", "12")),
            origins=origins,
        )
