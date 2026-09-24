"""Local CPU benchmark with a fixed prompt and separate startup timings."""

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_trail_backend.config import Settings  # noqa: E402
from token_trail_backend.inference import InferenceEngine  # noqa: E402
from token_trail_backend.schemas import TraceRequest  # noqa: E402


def main() -> None:
    engine = InferenceEngine(Settings())
    started = time.perf_counter()
    engine.load()
    trace = engine.trace(
        TraceRequest(prompt="The moon is", max_new_tokens=8, top_k=5, temperature=0)
    )
    steps = trace["steps"]
    print(
        json.dumps(
            {
                "model": trace["model"],
                "fixed_prompt": "The moon is",
                "tokenizer_load_ms": engine.tokenizer_load_ms,
                "model_load_ms": engine.model_load_ms,
                "prompt_tokenisation_ms": trace["timing"]["prompt_tokenisation_ms"],
                "first_step_ms": trace["timing"]["first_step_ms"],
                "per_step_ms": [step["step_ms"] for step in steps],
                "total_trace_ms": trace["timing"]["total_trace_ms"],
                "generated_token_count": len(steps),
                "peak_process_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024
                / 1024
                if sys.platform == "darwin"
                else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "whole_process_ms": (time.perf_counter() - started) * 1000,
            },
            indent=2,
        )
    )
    engine.close()


if __name__ == "__main__":
    main()
