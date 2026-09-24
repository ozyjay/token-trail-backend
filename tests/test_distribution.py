import math

import torch

from token_trail_backend.distribution import choose


def test_raw_probabilities_and_sorted_alternatives():
    logits = torch.tensor([0.0, 2.0, 1.0, -1.0])
    selected, alternatives = choose(logits, top_k=2, temperature=0, top_p=1)
    expected = torch.softmax(logits, dim=-1)
    assert selected.token_id == 1
    assert math.isclose(selected.probability, expected[1].item(), rel_tol=1e-6)
    assert [item.token_id for item in alternatives] == [2, 0]
    assert all(
        math.isclose(item.probability, expected[item.token_id].item(), rel_tol=1e-6)
        for item in alternatives
    )


def test_sampled_selection_still_uses_raw_probability():
    logits = torch.tensor([0.0, 1.0, 2.0])
    selected, alternatives = choose(
        logits, top_k=2, temperature=1.5, top_p=0.5, generator=torch.Generator().manual_seed(4)
    )
    raw = torch.softmax(logits, dim=-1)
    assert math.isclose(selected.probability, raw[selected.token_id].item(), rel_tol=1e-6)
    assert selected.token_id not in [item.token_id for item in alternatives]
    assert [x.probability for x in alternatives] == sorted(
        [x.probability for x in alternatives], reverse=True
    )


def test_probability_errors_are_visible():
    try:
        choose(torch.tensor([float("nan"), 1.0]), top_k=1, temperature=0, top_p=1)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite logits were accepted")
