"""Tests for the Python drills. Run via `python drills/run.py <NN>` (your exercise
file) or `python drills/run.py all --solutions` (reference solutions)."""
import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
IMPL = os.environ.get("DRILL_IMPL", "exercises")


def load(stem: str):
    path = HERE / IMPL / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"{IMPL}_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- 01
@pytest.mark.d01
def test_d01_attention_matches_torch():
    m = load("d01_attention")
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 4) for _ in range(3))
    for causal in (True, False):
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert torch.allclose(m.attention(q, k, v, causal), ref, atol=1e-5)


@pytest.mark.d01
def test_d01_mha_is_causal():
    m = load("d01_attention")
    torch.manual_seed(0)
    mha = m.MultiHeadAttention(16, 4)
    x = torch.randn(1, 6, 16)
    y1 = mha(x)
    x2 = x.clone()
    x2[:, -1] += 10.0  # changing the last token must not change earlier outputs
    y2 = mha(x2)
    assert y1.shape == (1, 6, 16)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)


# --------------------------------------------------------------------------- 02
@pytest.mark.d02
def test_d02_filters_and_sampling():
    m = load("d02_sampling")
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    tk = m.top_k_filter(logits, 2)
    assert torch.isinf(tk[0, 2:]).all() and torch.isfinite(tk[0, :2]).all()
    tp = m.top_p_filter(logits, 0.7)  # 0.5 < 0.7 -> need 0.3 too
    assert torch.isfinite(tp[0, :2]).all() and torch.isinf(tp[0, 2:]).all()
    tiny = m.top_p_filter(logits, 0.01)
    assert torch.isfinite(tiny[0, 0]) and torch.isinf(tiny[0, 1:]).all()
    assert m.sample(logits, temperature=0).item() == 0
    g = torch.Generator().manual_seed(0)
    draws = torch.stack([m.sample(logits.expand(1000, 4), 1.0, 0, 0.7, g) for _ in range(5)]).flatten()
    assert set(draws.tolist()) <= {0, 1}
    frac0 = (draws == 0).float().mean().item()
    assert abs(frac0 - 0.5 / 0.8) < 0.03


@pytest.mark.d02
def test_d02_unsorted_logits():
    m = load("d02_sampling")
    logits = torch.log(torch.tensor([[0.05, 0.3, 0.15, 0.5]]))
    tp = m.top_p_filter(logits, 0.7)
    assert torch.isfinite(tp[0, [1, 3]]).all() and torch.isinf(tp[0, [0, 2]]).all()


# --------------------------------------------------------------------------- 03
@pytest.mark.d03
def test_d03_response_logprobs():
    m = load("d03_logprobs")
    torch.manual_seed(0)
    B, T, V = 2, 6, 9
    logits, ids = torch.randn(B, T, V), torch.randint(0, V, (B, T))
    mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]])
    got = m.response_logprobs(logits, ids, mask)
    for b in range(B):
        exp = sum(torch.log_softmax(logits[b, t - 1], -1)[ids[b, t]].item() for t in range(T) if mask[b, t])
        assert math.isclose(got[b].item(), exp, rel_tol=1e-5)


# --------------------------------------------------------------------------- 04
@pytest.mark.d04
def test_d04_dpo():
    m = load("d04_dpo")
    x = torch.tensor([-3.0, -4.0])
    loss, acc = m.dpo_loss(x, x, x, x, 0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    pc, pr, rc, rr = torch.tensor([-2.0]), torch.tensor([-6.0]), torch.tensor([-3.0]), torch.tensor([-5.0])
    loss, acc = m.dpo_loss(pc, pr, rc, rr, 0.5)
    assert math.isclose(loss.item(), -math.log(1 / (1 + math.exp(-1.0))), rel_tol=1e-6)
    assert acc.item() == 1.0


# --------------------------------------------------------------------------- 05
@pytest.mark.d05
def test_d05_reward():
    m = load("d05_reward")
    rc, rr = torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0])
    assert math.isclose(m.bradley_terry_loss(rc, rr).item(),
                        (math.log(1 + math.exp(-1)) + math.log(1 + math.exp(1))) / 2, rel_tol=1e-6)
    assert m.pairwise_accuracy(rc, rr) == 0.5
    h = torch.arange(24.0).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert torch.equal(m.last_token_pool(h, mask), torch.stack([h[0, 1], h[1, 3]]))


# --------------------------------------------------------------------------- 06
@pytest.mark.d06
def test_d06_kl():
    m = load("d06_kl")
    torch.manual_seed(0)
    pl, ql = torch.randn(7), torch.randn(7)
    p, q = torch.softmax(pl, -1), torch.softmax(ql, -1)
    true = (p * (p.log() - q.log())).sum()
    assert torch.allclose(m.exact_kl(pl, ql), true, atol=1e-6)
    x = torch.multinomial(p, 300_000, replacement=True)
    lp, lq = p.log()[x], q.log()[x]
    assert abs(m.kl_k1(lp, lq).mean() - true) < 0.02
    assert abs(m.kl_k3(lp, lq).mean() - true) < 0.02
    assert (m.kl_k3(lp, lq) >= -1e-7).all() and (m.kl_k2(lp, lq) >= 0).all()


# --------------------------------------------------------------------------- 07
@pytest.mark.d07
def test_d07_gae():
    m = load("d07_gae")
    adv, ret = m.gae([0.0, 0.0, 1.0], [0.2, 0.5, 0.9], gamma=1.0, lam=1.0)
    assert all(math.isclose(r, 1.0) for r in ret)
    adv, _ = m.gae([0.5, 1.0], [0.2, 0.4], gamma=0.9, lam=0.0, last_value=0.3)
    assert math.isclose(adv[0], 0.5 + 0.9 * 0.4 - 0.2) and math.isclose(adv[1], 1.0 + 0.9 * 0.3 - 0.4)


# --------------------------------------------------------------------------- 08
@pytest.mark.d08
def test_d08_packing():
    m = load("d08_packing")
    rows, segs, pos = m.pack_sequences([[1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11]], seq_len=5, pad_id=0)
    assert rows == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 0, 0, 0, 0]]
    assert segs == [[1, 1, 1, 2, 2], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]
    assert pos == [[0, 1, 2, 0, 1], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]]
    mask = m.block_mask([1, 1, 2, 0])
    assert mask == [[True, False, False, False], [True, True, False, False],
                    [False, False, True, False], [False, False, False, False]]


# --------------------------------------------------------------------------- 09
@pytest.mark.d09
def test_d09_bpe():
    m = load("d09_bpe")
    text = "low low low lower lowest newer newer wider"
    merges = m.train_bpe(text, 10)
    assert merges[0] == ("l", "o")
    assert ("lo", "w") in merges
    enc = m.encode("lowest newer", merges)
    assert m.decode(enc) == "lowest newer"
    assert enc[0][0] == "low"
    assert m.train_bpe("a b c", 5) == []


# --------------------------------------------------------------------------- 10
@pytest.mark.d10
def test_d10_lru():
    m = load("d10_engineering")
    c = m.LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # evicts b (least recently used)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3


@pytest.mark.d10
def test_d10_first_bad_step():
    m = load("d10_engineering")
    lines = ['{"step": 1, "loss": 2.0}', "garbage", '{"step": 2, "lr": 1}', '{"step": 3, "loss": 1.9}',
             '{"step": 4, "loss": NaN}']
    assert m.first_bad_step(lines) == 4
    assert m.first_bad_step(['{"step": 1, "loss": 1.0}', '{"step": 2, "loss": 50.0}']) == 2
    assert m.first_bad_step(['{"step": 1, "loss": 1.0}']) is None


@pytest.mark.d10
def test_d10_parallel_map():
    m = load("d10_engineering")
    calls = {}

    def flaky(x):
        calls[x] = calls.get(x, 0) + 1
        if x == 3 and calls[x] < 2:
            raise ValueError("transient")
        if x == 5:
            raise RuntimeError("permanent")
        return x * x

    out = m.parallel_map(flaky, list(range(7)), workers=3, retries=2)
    assert out[:5] == [0, 1, 4, 9, 16] and out[6] == 36
    assert isinstance(out[5], RuntimeError) and calls[5] == 3
