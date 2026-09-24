# API contract v1

`GET /health` returns `{"status":"ready","ready":true}` once the pinned model is loaded, or `not_ready` and `false` otherwise. Model load failure stops startup.

`GET /v1/models` lists the single approved alias, repository, pinned revision and readiness. It never accepts a visitor-supplied model ID.

`POST /v1/traces` accepts JSON:

| Field | Type | Bounds | Default |
| --- | --- | --- | --- |
| `prompt` | string | 1–1000 characters, at most 256 model tokens | required |
| `model` | string | `qwen2.5-0.5b` only | `qwen2.5-0.5b` |
| `max_new_tokens` | integer | 1–32 | 16 |
| `top_k` | integer | 1–10 alternatives | 5 |
| `temperature` | number | 0–2; zero means greedy | 0 |
| `top_p` | number | greater than 0, at most 1 | 1 |
| `seed` | integer or null | 0–2⁶³−1 | null |

Unknown fields and malformed JSON return `422`. An invalid model alias also returns `422`; there is no fallback. Top-p has no effect in greedy mode. A seed controls sampling on this process and model revision, but exact outputs can differ across library or hardware versions.

A successful response contains `schema_version`, `model` (alias, repository, revision), `prompt`, `prompt_tokens` (ID and decoded piece), `generated_text`, `steps`, `decoding`, `timing`, and `finish_reason` (`eos` or `length`). Steps are numbered from zero and contain `selected_token`, `selected_token_id`, `selected_probability`, `alternatives` (each ID, decoded piece, probability, sorted high to low), `step_ms`, and `elapsed_ms`. `step_ms` includes model forward, distribution calculation and selection. `elapsed_ms` is measured from the start of the trace request. `timing` includes prompt tokenisation, first-step and total trace milliseconds. The selected token is separate from alternatives, even when selected by sampling outside the top-k.

All displayed probabilities use `decoding.probability_basis: "raw_softmax_logits"` from the model's next-token logits at that step. Sampling uses temperature-adjusted logits and optional nucleus filtering. Alternatives remain ranked by raw probability, which makes distributions comparable across decoding settings. They need not sum to one because the full vocabulary is much larger.

Errors have `{"error":{"code":"...","message":"..."}}`. Relevant statuses: `413 payload_too_large`, `422 invalid_request` or `prompt_too_long`, `429 busy` or `rate_limited`, `503 not_ready`, `504 timeout`, and `500 inference_failed`. A disconnected client may be represented as `499 client_disconnected` internally; browsers typically see a cancelled network request.

No streaming endpoint is provided in v1. The entire bounded trace is returned after generation.
