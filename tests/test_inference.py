from types import SimpleNamespace

import torch

from token_trail_backend.config import Settings
from token_trail_backend.inference import InferenceEngine, PromptTooLong
from token_trail_backend.schemas import TraceRequest


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, return_tensors, add_special_tokens):
        length = 257 if text == "too long" else 2
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    def decode(self, ids, **kwargs):
        return "".join({0: " A", 1: "B", 2: "C"}.get(i, "?") for i in ids)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        scores = [0.0, 1.0, 2.0] if self.calls == 1 else [3.0, 1.0, 0.0]
        return SimpleNamespace(logits=torch.tensor([[scores]]), past_key_values=object())


def test_probabilities_match_each_generation_step():
    engine = InferenceEngine(Settings())
    engine.tokenizer = FakeTokenizer()
    engine.model = FakeModel()
    trace = engine.trace(TraceRequest(prompt="Hi", max_new_tokens=2, top_k=2))
    assert [step["selected_token_id"] for step in trace["steps"]] == [2, 0]
    for step, scores in zip(trace["steps"], ([0.0, 1.0, 2.0], [3.0, 1.0, 0.0]), strict=True):
        raw = torch.softmax(torch.tensor(scores), dim=0)
        assert abs(step["selected_probability"] - raw[step["selected_token_id"]].item()) < 1e-6
        assert (
            step["alternatives"][0]["probability"]
            == raw[step["alternatives"][0]["token_id"]].item()
        )
    assert [step["step"] for step in trace["steps"]] == [0, 1]


def test_prompt_token_limit():
    engine = InferenceEngine(Settings())
    engine.tokenizer = FakeTokenizer()
    engine.model = FakeModel()
    try:
        engine.trace(TraceRequest(prompt="too long"))
    except PromptTooLong:
        pass
    else:
        raise AssertionError("long tokenised prompt was accepted")
