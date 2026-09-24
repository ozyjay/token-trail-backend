---
title: Token Trail API
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
short_description: Next-token probability traces from an open model
---

# Token Trail backend

A small FastAPI service for educational next-token traces. It loads one pinned open-weight model, `Qwen/Qwen2.5-0.5B`, on CPU. It has no ModelDeck or Framework Desktop dependency, no browser secret, no database, no model fallback, and no prompt logging. The site can call the Space over HTTPS from an allowed CrunchyCodes origin.

## Quick start

Use Python 3.12 from pyenv. In this directory:

```text
python3 -m venv .venv
.venv/bin/python3 -m pip install -e '.[dev]'
.venv/bin/python3 scripts/download_model.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/uvicorn token_trail_backend.api:app --host 127.0.0.1 --port 7860 --workers 1 --no-access-log
```

`GET /health` reports readiness; the service loads its pinned model before accepting traces. Example request:

```text
curl -X POST http://127.0.0.1:7860/v1/traces -H 'Content-Type: application/json' -d '{"prompt":"The moon is","max_new_tokens":8,"top_k":5}'
```

Run unit tests and lint with `.venv/bin/python3 -m pytest` and `.venv/bin/ruff check .`. Run `.venv/bin/python3 scripts/benchmark.py` after downloading the model. The benchmark does not send a visitor prompt anywhere.

## Semantics

Each `selected_probability` and alternative `probability` is the **raw softmax of the causal model's logits** immediately before that token was selected. Temperature and top-p affect the sampling choice only; they do not change the displayed probabilities. Greedy decoding is the default (`temperature: 0`). Token pieces are decoded one ID at a time; concatenating those pieces may differ slightly from decoding the full sequence, so use `generated_text` for display. These values describe next-token likelihood under the model, not reasoning or confidence in factual correctness.

See [API contract](docs/API.md), [architecture and investigation](docs/ARCHITECTURE.md), [deployment](docs/DEPLOYMENT.md), and [local benchmark](docs/BENCHMARK.md).

## Privacy and limits

The service holds prompts in process memory only while handling a request. It does not persist prompts or responses and starts Uvicorn without access logs. Framework, reverse proxy, Hugging Face platform and browser network logs are outside this application-level promise. Request bodies are capped at 4096 bytes; prompts at 1000 characters and 256 model tokens; output at 32 tokens; alternatives at 10; one trace runs per process; and the application applies a 12 requests/minute per observed client-IP limit. Long forward passes cannot be interrupted mid-call, so the 30-second budget and disconnect cancellation are checked between steps. The in-process limiter is per worker and may group visitors if the Space proxy presents a shared address. Keep one worker and add an edge rate limit before substantial public traffic.
