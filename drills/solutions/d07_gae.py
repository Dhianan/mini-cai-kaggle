"""Drill 07 (30 min): Generalized Advantage Estimation, pure Python.

Interview follow-ups: what do lambda=0 and lambda=1 reduce to? Why does GRPO
drop the value function entirely? Where does the bias-variance trade-off sit?
"""


def gae(rewards: list[float], values: list[float], gamma: float = 1.0, lam: float = 0.95,
        last_value: float = 0.0) -> tuple[list[float], list[float]]:
    """One trajectory. values[t] = V(s_t); last_value = V(s_T) (0 if terminal).
    Returns (advantages, returns) with returns = advantages + values."""
    adv = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(rewards) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv, [a + v for a, v in zip(adv, values)]
