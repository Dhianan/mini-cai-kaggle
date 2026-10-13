"""Hand-written losses vs independent references."""
import math

import torch
import torch.nn.functional as F

from minicai import losses as L

torch.manual_seed(0)


def _rand(B=3, T=7, V=11):
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = L.IGNORE  # prompt
    labels[0, -1] = L.IGNORE  # padding
    return logits, labels


def test_sft_loss_matches_manual():
    logits, labels = _rand()
    total, n = 0.0, 0
    for b in range(logits.size(0)):
        for t in range(logits.size(1) - 1):
            y = labels[b, t + 1].item()
            if y != L.IGNORE:
                total += -torch.log_softmax(logits[b, t], -1)[y].item()
                n += 1
    assert math.isclose(L.sft_loss(logits, labels).item(), total / n, rel_tol=1e-5)


def test_sequence_logprobs_brute_force():
    logits, labels = _rand()
    got = L.sequence_logprobs(logits, labels)
    for b in range(logits.size(0)):
        s = sum(torch.log_softmax(logits[b, t], -1)[labels[b, t + 1]].item()
                for t in range(logits.size(1) - 1) if labels[b, t + 1] != L.IGNORE)
        assert math.isclose(got[b].item(), s, rel_tol=1e-5)


def test_dpo_loss_formula_and_identity():
    pc, pr, rc, rr = torch.tensor([-5.0, -3.0]), torch.tensor([-6.0, -2.0]), torch.tensor([-5.5, -3.0]), torch.tensor([-5.5, -2.5])
    beta = 0.1
    loss, m = L.dpo_loss(pc, pr, rc, rr, beta)
    z = beta * ((pc - rc) - (pr - rr))
    assert math.isclose(loss.item(), (-F.logsigmoid(z)).mean().item(), rel_tol=1e-6)
    # policy == reference -> loss is log 2 regardless of beta
    l2, _ = L.dpo_loss(rc, rr, rc, rr, 0.5)
    assert math.isclose(l2.item(), math.log(2), rel_tol=1e-6)
    assert m["reward_acc"].item() == 0.5


def test_dpo_gradient_pushes_chosen_up_rejected_down():
    pc = torch.tensor([-4.0], requires_grad=True)
    pr = torch.tensor([-4.0], requires_grad=True)
    loss, _ = L.dpo_loss(pc, pr, torch.tensor([-4.0]), torch.tensor([-4.0]), 0.1)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()  # gradient descent raises chosen, lowers rejected


def test_bradley_terry():
    loss, m = L.bradley_terry_loss(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]))
    expect = (-math.log(1 / (1 + math.exp(-2))) - math.log(1 / (1 + math.exp(1)))) / 2
    assert math.isclose(loss.item(), expect, rel_tol=1e-6)
    assert m["acc"].item() == 0.5


def test_kl_estimators_unbiased():
    p = torch.softmax(torch.randn(6), -1)
    q = torch.softmax(torch.randn(6), -1)
    true_kl = (p * (p.log() - q.log())).sum().item()
    x = torch.multinomial(p, 200_000, replacement=True)
    lp, lq = p.log()[x], q.log()[x]
    assert abs(L.kl_k1(lp, lq).mean().item() - true_kl) < 0.02
    k3 = L.kl_k3(lp, lq)
    assert (k3 >= -1e-7).all()
    assert abs(k3.mean().item() - true_kl) < 0.02


def test_gae_lambda1_is_monte_carlo():
    r = torch.tensor([[0.0, 0.0, 1.0]])
    v = torch.tensor([[0.2, 0.5, 0.9]])
    adv, ret = L.gae(r, v, gamma=1.0, lam=1.0)
    assert torch.allclose(ret, torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.allclose(adv, ret - v)


def test_gae_lambda0_is_td():
    r = torch.tensor([[0.5, 1.0]])
    v = torch.tensor([[0.2, 0.4]])
    adv, _ = L.gae(r, v, gamma=0.9, lam=0.0)
    assert torch.allclose(adv, torch.tensor([[0.5 + 0.9 * 0.4 - 0.2, 1.0 - 0.4]]))


def test_group_advantages():
    a = L.group_advantages(torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert torch.allclose(a.mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(a[1], torch.zeros(3))  # identical rewards -> no signal


def test_clipped_pg_on_policy_equals_neg_advantage():
    lp = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    adv = torch.tensor([1.0, -2.0])
    loss, m = L.clipped_pg_loss(lp, lp.clone(), adv, mask)
    assert math.isclose(loss.item(), -adv.mean().item(), rel_tol=1e-6)
    assert m["clip_frac"].item() == 0.0


def test_clipping_limits_positive_advantage_gain():
    old = torch.zeros(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    adv = torch.tensor([1.0])
    far, _ = L.clipped_pg_loss(torch.full((1, 1), 1.0), old, adv, mask, clip=0.2)  # ratio e
    assert math.isclose(far.item(), -1.2, rel_tol=1e-6)
