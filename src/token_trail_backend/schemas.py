"""Public API schemas. Unknown inputs are rejected."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import MAX_NEW_TOKENS, MAX_PROMPT_CHARS, MAX_TOP_K, MODEL_ALIAS


class TraceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    model: str = MODEL_ALIAS
    max_new_tokens: int = Field(default=16, ge=1, le=MAX_NEW_TOKENS)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)

    @field_validator("prompt")
    @classmethod
    def valid_prompt(cls, value: str) -> str:
        if not value.strip() or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("prompt must contain text and no unpaired surrogates")
        return value

    @field_validator("model")
    @classmethod
    def approved_model(cls, value: str) -> str:
        if value != MODEL_ALIAS:
            raise ValueError("unknown model alias")
        return value
