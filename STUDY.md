# Study guide

Morning reading (2h/day), then answer the day's questions out loud **without notes**.
Answers below are short keys to check yourself against, not scripts to memorise.

## Reading by day

| Day | Read | Focus on |
|---|---|---|
| 1 | Ouyang et al. 2022, *Training language models to follow instructions with human feedback* (InstructGPT) · Bai et al. 2022, *Constitutional AI*, §1–3 | the 3-stage pipeline; why SFT first |
| 2 | Constitutional AI, SL-CAI section + appendix constitution | critique→revision; why mix helpful data |
| 3 | Stiennon et al. 2020, *Learning to summarize from human feedback* · Lee et al. 2023, *RLAIF* | RM training, overoptimisation, AI vs human labels |
| 4 | Rafailov et al. 2023, *DPO* · Shao et al. 2024, *DeepSeekMath* (GRPO section) · Gao et al. 2022, *Scaling laws for reward model overoptimization* | implicit reward; group baselines; Goodhart |
| 5 | Rajbhandari et al. 2020, *ZeRO* §1–4 · PyTorch FSDP docs | what is sharded, what is communicated |
| 6 | Your own `metrics.jsonl` files | reading curves before reading code |
| 7 | Anthropic's recent research posts | pick one open problem to have an opinion on |

## Concept questions (answer keys)

**Why mask prompt tokens in SFT? What breaks if you don't?**
You want p(response | prompt), not a language model of user text. Without the mask the model spends capacity
imitating users, learns to continue *as the user*, and the loss looks better than it is (prompts are easy).
In this repo: fault `prompt_loss`; guardrail `check_loss_mask`.

**Why include the end-of-turn token in the loss?**
It's the only signal that teaches the model to stop. Drop it and generations run to max length (`eos_dropped`).

**DPO vs PPO/GRPO: what does each optimise, and when does online RL win?**
Both target max E[r] − β·KL(π‖π_ref). DPO solves it in closed form on a *fixed* preference dataset (offline,
no RM, no sampling, stable, cheap). Online RL samples from the current policy and scores with an RM, so it
gets feedback on its *own* distribution, can keep improving beyond the dataset, and can optimise any reward
(verifiers, rubrics). Online wins when the dataset's responses are far from what the policy now produces,
or when you have a reward you can query cheaply. It costs instability and reward-hacking risk.

**What does β control in DPO?**
The strength of the implicit KL to the reference. Small β: rewards grow fast, policy moves far, overfits
noisy labels. Large β: conservative, small changes. It's also the scale of the implicit reward β·log(π/π_ref).

**Why can log p(chosen) go down during DPO?**
The loss only cares about the *margin*. Both log-probs often fall, rejected faster. Mass moves to responses
in neither set ("likelihood displacement"); in excess the model degrades. Watch `logp_chosen`.

**What does the KL penalty do in RLHF?**
Keeps the policy near the reference so the RM stays in-distribution (the RM was trained on reference-like
samples) and preserves capabilities/fluency. Too low: reward hacking. Too high: no learning.

**k1 vs k3 KL estimators?**
For x ~ π: k1 = log π/π_ref (unbiased, high variance, negative per sample). k3 = (r−1) − log r with
r = π_ref/π (unbiased, always ≥ 0, lower variance). This repo uses k3 as the per-token penalty.

**Why does GRPO not need a value function?**
It samples G responses per prompt and uses the group mean (and std) as the baseline. That's a Monte-Carlo
baseline per prompt instead of a learned critic: saves a model and its instability, costs more samples.

**How does reward hacking show up in logs before samples?**
RM reward rises while the judge/eval metric is flat or falls; length (or refusal rate, or a format feature)
moves in lockstep with reward; KL rises steadily; reward-length correlation increases; entropy drops.

**Why does CAI mix helpful data into the harmlessness phase?**
Training only on revised harmful-prompt responses teaches "refuse/hedge" as the universal answer. Helpful data
anchors normal behaviour; the `harm-not-evasive` principle targets the same failure.

**Position bias and self-preference in AI labelers: how to measure and reduce?**
Run every comparison in both orders; the flip rate is the position-bias rate (`labeler_report.json`). Average
both orders, drop inconsistent pairs. Self-preference (a model favours its own style): use a different labeler
family, check agreement with human labels, and check length/style correlations of the chosen side.

**FSDP vs DDP vs tensor parallel?**
DDP: full replica per GPU; all-reduce gradients. Memory = whole model + optimizer per GPU.
FSDP (ZeRO-3): shard params, grads and optimizer state; all-gather params per layer for forward/backward,
reduce-scatter grads. Memory ~1/N, more communication. ZeRO-2 (`SHARD_GRAD_OP`) shards grads + optimizer only.
Tensor parallel splits individual matmuls across GPUs (needs fast interconnect, usually within a node).

**Why bf16 over fp16? Where do NaNs come from?**
bf16 has fp32's exponent range, so no loss scaling and far fewer overflows; less mantissa precision is fine
for training with fp32 master weights. NaNs: overflow in fp16 logits/softmax, log(0) in a loss, division by a
zero mask count, LR too high, a bad batch (empty labels), exploding grads without clipping.

**How do you tell if an eval gain is real?**
Multiple seeds and confidence intervals; same judge and reference across stages; both-order judging;
check contamination; check length (judges like long answers); read samples; check that held-out human
agreement of the judge is decent.

**Memory budget for full fine-tuning a 1.5B model with AdamW?**
fp32 weights 6 GB + grads 6 GB + Adam m and v 12 GB = ~24 GB before activations. Activations scale with
batch × sequence × layers × hidden; gradient checkpointing trades ~30% more compute for most of that memory.

## 10-minute research walkthrough (Day 7)

1. Question (30s): can a 1.5B base model be taken through the full CAI stack on one GPU in a week, and
   where does it break?
2. Setup (2 min): the diagram; what was hand-written; data; judge.
3. One result (2 min): the results table; the single most interesting number and its uncertainty.
4. One failure (3 min): the reward-hacking or incident you found; symptom → hypotheses in order → cause → guardrail.
5. Next (2 min): with 1000× compute, what you'd change first and what you'd measure.
6. Leave 30s for "what surprised you".
