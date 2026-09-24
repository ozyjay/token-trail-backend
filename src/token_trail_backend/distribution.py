"""One raw next-token distribution and its optional sampling distribution."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Choice:
    token_id: int
    probability: float


def choose(
    logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> tuple[Choice, list[Choice]]:
    raw = torch.softmax(logits.float(), dim=-1)
    if not torch.isfinite(raw).all():
        raise ValueError("model returned non-finite probabilities")
    if temperature == 0:
        selected_id = int(torch.argmax(raw).item())
    else:
        adjusted = torch.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1:
            sorted_probs, sorted_ids = torch.sort(adjusted, descending=True)
            keep = torch.cumsum(sorted_probs, dim=0) - sorted_probs < top_p
            filtered = torch.zeros_like(adjusted)
            filtered[sorted_ids[keep]] = sorted_probs[keep]
            adjusted = filtered / filtered.sum()
        selected_id = int(torch.multinomial(adjusted, 1, generator=generator).item())
    ids = torch.topk(raw, min(top_k + 1, raw.numel())).indices.tolist()
    alternatives = [Choice(int(i), float(raw[i].item())) for i in ids if i != selected_id][:top_k]
    return Choice(selected_id, float(raw[selected_id].item())), alternatives
