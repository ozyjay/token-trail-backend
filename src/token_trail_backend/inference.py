"""Single-model CPU inference, with logits captured at each autoregressive step."""

import time
from threading import Event
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MAX_PROMPT_TOKENS, Settings
from .distribution import choose
from .schemas import TraceRequest


class PromptTooLong(ValueError):
    pass


class TraceTimedOut(TimeoutError):
    pass


class InferenceEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokenizer: Any = None
        self.model: Any = None
        self.tokenizer_load_ms: float | None = None
        self.model_load_ms: float | None = None

    def load(self) -> None:
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_repo,
            revision=self.settings.model_revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.tokenizer_load_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_repo,
            revision=self.settings.model_revision,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        self.model.eval()
        self.model_load_ms = (time.perf_counter() - start) * 1000

    def close(self) -> None:
        self.model = None
        self.tokenizer = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def trace(self, request: TraceRequest, cancel: Event | None = None) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("model is not ready")
        started = time.perf_counter()
        encoded = self.tokenizer(request.prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = encoded["input_ids"][0].tolist()
        tokenisation_ms = (time.perf_counter() - started) * 1000
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            raise PromptTooLong(f"prompt exceeds {MAX_PROMPT_TOKENS} model tokens")
        generator = torch.Generator(device="cpu")
        if request.seed is not None:
            generator.manual_seed(request.seed)
        else:
            generator.seed()
        context = encoded["input_ids"]
        mask = encoded["attention_mask"]
        past = None
        steps: list[dict[str, Any]] = []
        selected_ids: list[int] = []
        with torch.inference_mode():
            for index in range(request.max_new_tokens):
                if cancel is not None and cancel.is_set():
                    raise TraceTimedOut("trace cancelled")
                if time.perf_counter() - started > self.settings.request_timeout_seconds:
                    raise TraceTimedOut("trace exceeded time budget")
                step_start = time.perf_counter()
                output = self.model(
                    input_ids=context, attention_mask=mask, past_key_values=past, use_cache=True
                )
                logits = output.logits[0, -1]
                selected, alternatives = choose(
                    logits,
                    top_k=request.top_k,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    generator=generator,
                )
                past = output.past_key_values
                selected_ids.append(selected.token_id)
                step_ms = (time.perf_counter() - step_start) * 1000
                steps.append(
                    {
                        "step": index,
                        "selected_token": self.tokenizer.decode(
                            [selected.token_id], clean_up_tokenization_spaces=False
                        ),
                        "selected_token_id": selected.token_id,
                        "selected_probability": selected.probability,
                        "alternatives": [
                            {
                                "token": self.tokenizer.decode(
                                    [item.token_id], clean_up_tokenization_spaces=False
                                ),
                                "token_id": item.token_id,
                                "probability": item.probability,
                            }
                            for item in alternatives
                        ],
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "step_ms": step_ms,
                    }
                )
                if selected.token_id == self.tokenizer.eos_token_id:
                    break
                context = torch.tensor([[selected.token_id]], dtype=torch.long)
                mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype)), dim=1)
        return {
            "schema_version": "1.0",
            "model": {
                "alias": self.settings.model_alias,
                "repository": self.settings.model_repo,
                "revision": self.settings.model_revision,
            },
            "prompt": request.prompt,
            "prompt_tokens": [
                {
                    "token_id": int(i),
                    "token": self.tokenizer.decode([int(i)], clean_up_tokenization_spaces=False),
                }
                for i in prompt_ids
            ],
            "generated_text": self.tokenizer.decode(
                selected_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "steps": steps,
            "decoding": {
                "method": "greedy" if request.temperature == 0 else "sample",
                "temperature": request.temperature,
                "top_p": request.top_p,
                "seed": request.seed,
                "probability_basis": "raw_softmax_logits",
            },
            "timing": {
                "prompt_tokenisation_ms": tokenisation_ms,
                "first_step_ms": steps[0]["elapsed_ms"] if steps else None,
                "total_trace_ms": (time.perf_counter() - started) * 1000,
            },
            "finish_reason": "eos"
            if selected_ids and selected_ids[-1] == self.tokenizer.eos_token_id
            else "length",
        }
