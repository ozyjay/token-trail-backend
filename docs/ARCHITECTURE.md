# Investigation and architecture

## Existing Token Trail

Inspected `ozyjay/TokenTrail` at commit `b4c3a40bf3df58183e43694c9efa7d4ad341a25e`, cloned on 24 September 2026. It has two trace families. Prepared traces in `src/token_trail/traces.py` are display scripts with `prompt_tokens`, `steps`, `selected_token`, `candidates` and human-authored `explanation`; the simple tokeniser and probabilities are illustrative, not model output. The live ModelDeck adapter in `src/token_trail/adapters/modeldeck.py` calls `/native/v1/autoregressive/traces`. It validates native prompt and user-prompt token ID/text arrays, then maps events into `step`, selected token ID/text, candidates ID/text/probability, cumulative token IDs, `text_so_far`, timestamps, elapsed seconds and completion. It carries a request ID and optional metrics. The ModelDeck adapter sorts candidates, discards generated control tokens and requires a sentence boundary after at least eight steps. It does not expose a named probability basis in its public conversion.

The existing local HF probe and server are useful evidence, but are not a suitable public boundary. The probe can obtain forward logits, yet its trace omits selected IDs, candidate IDs, step numbers and timings; it deduplicates decoded candidate strings, which can collapse distinct IDs. Its server offers warm-up and model discovery geared to a local operator, accepts an instruction field, and has no public request size, upper generation, rate or concurrency controls. Its startup assumes a local cache. The UI's scripted fallback is appropriate for the local demo, but a hosted endpoint must report a model failure visibly.

## New design

One FastAPI process loads one pinned Qwen base model at startup from a build-time snapshot. Request schemas, HTTP controls, inference, distribution maths and configuration are separate modules. Every inference step reads the actual final-position causal logits, calculates full-vocabulary raw softmax, chooses greedy or samples from a separately adjusted distribution, and returns the selected token plus ranked raw alternatives. Token IDs stay distinct. A single process lock limits model work to one trace; excess calls get `429`. Model failure fails startup or returns an explicit error. There is no persistence, hidden system prompt, arbitrary model selection or fallback service.

## Model choice

| Model | Weights | Licence | Trade-off |
| --- | ---: | --- | --- |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | about 988 MB safetensors | Apache 2.0 | Chosen: small, official, standard Transformers causal LM and broad tokeniser; base completion can be rough. |
| [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | about 3.09 GB safetensors | Apache 2.0 | Better capacity, substantially slower and more memory on CPU. |
| [Pythia-1B](https://huggingface.co/EleutherAI/pythia-1b) | around 2 GB in 16-bit weights | Apache 2.0 | Research-friendly English base with simple GPT-NeoX support, but its card says it is not intended for human-facing deployment. |

Qwen2.5-0.5B has roughly 494 million parameters. Float32 weights require about 2 GB before runtime buffers and Python overhead, while the download is about 988 MB. Expect several GB of RAM, then measure on the Space. Its BPE tokenisation can split leading spaces and Unicode into pieces; token IDs and text pieces are both returned to show this clearly. The pinned revision is `060db6499f32faf8b98477b0a26969ef7d8b9987`. The educational objective favours small, repeatable raw logits over answer quality. We have not benchmarked the other models locally.

## Limitations and next experiments

The API returns a complete trace, not live streamed steps. CPU forward passes are checked for timeout and cancellation between tokens, so an individual slow call can exceed the nominal budget. The in-memory rate limit cannot coordinate multiple replicas and observed IPs may be proxy addresses. Test live Space cold starts, load memory, public origins and proxy behaviour before connecting the website. Then consider streaming and a managed edge rate limit if use warrants it.

Model references: [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen2.5-0.5B weights](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/model.safetensors), [Qwen2.5-1.5B weights](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/model.safetensors), and [Pythia-1B model card](https://huggingface.co/EleutherAI/pythia-1b).
