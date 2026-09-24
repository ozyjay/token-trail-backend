# Project instructions

- Use Australian spelling in documentation.
- Use the active pyenv `python3` for Python work.
- Keep the public API narrow and the model pinned to one approved alias.
- Never add a fallback model, visitor-selected model ID, prompt storage, or prompt-bearing telemetry.
- A candidate probability is the raw softmax of next-token logits for its own step.
- Keep CPU operation working before considering other hardware.
