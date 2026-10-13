# Mini-CAI: Constitutional AI post-training in 7 days

A complete, hand-written post-training stack on a 1.5B **base** model:

```
base ──SFT──> sft ──critique/revise──> slcai ──AI-labelled pairs──> RM ──GRPO──> rl
                                         └──────────────DPO──────────────> dpo
                          eval harness after every stage
```

Every loss (SFT mask, Bradley-Terry, DPO, k3 KL, GRPO/PPO clip, GAE) is written by hand
in `src/minicai/losses.py` and unit-tested against independent references. Library
trainers (TRL) are used only to cross-check. Ten realistic training bugs can be injected
blind for incident drills, and each has a guardrail that catches it.

## Layout

| Path | What |
|---|---|
| `src/minicai/` | `build_data` · `sft` · `cai_revise` · `label_prefs` · `train_rm` · `train_dpo` · `train_rl` · `evals` · `report` |
| `src/minicai/losses.py` | every loss, hand-written |
| `src/minicai/chat.py` | ChatML rendering with exact assistant-only loss masks |
| `src/minicai/checks.py` | guardrails (loss mask, EOS, template hash, frozen ref, contamination) |
| `src/minicai/faults.py` | Day 6 fault injector (10 faults) |
| `configs/` | one YAML per stage; override anything with `key.sub=value` |
| `constitution.yaml` | 12 principles; edit them, it's part of the exercise |
| `tests/` | 30 tests: losses, masks, every guardrail, overfit, exact resume, DPO/RM learning |
| `drills/` | 10 timed Python interview drills with tests and reference solutions |
| `scripts/` | `setup_gpu.sh`, `smoke.sh` (CPU end-to-end), `bench.sh` (Day 5), `drill.sh` (Day 6) |

## Before Day 1 (on your laptop, 10 minutes)

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest -q                 # 30 tests, ~1 min on CPU
bash scripts/smoke.sh               # whole pipeline on a 2-layer CPU model, ~2 min
python drills/run.py all --solutions
```

Then rent **1× H100 or A100 80GB** (hourly) for Days 1–4 and 6, and **2–4 GPUs** for Day 5.
On the box: `git clone … && bash scripts/setup_gpu.sh`.

### Free tier (no budget): Kaggle T4

Same code and drills, smaller scale: policy `Qwen2.5-0.5B`, judge `Qwen2.5-3B-Instruct`, fp16 (T4 has no
native bf16; the code switches automatically). Kaggle gives ~30 free GPU-hours/week, 12-hour sessions.

1. Kaggle → New Notebook → Settings: Accelerator **GPU T4 x2**, Internet **on**.
2. Add-ons → Secrets: `HF_TOKEN` (write token from huggingface.co/settings/tokens), `HF_REPO` = `<you>/mini-cai-runs`.
3. Cells:
   ```python
   from kaggle_secrets import UserSecretsClient; import os
   s = UserSecretsClient(); os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN"); os.environ["HF_REPO"] = s.get_secret("HF_REPO")
   !git clone https://github.com/<you>/mini-cai.git && cd mini-cai && bash scripts/kaggle.sh setup
   !cd mini-cai && bash scripts/kaggle.sh day1     # then day2, day3, day4, day5 (2 GPUs), day6
   !cd mini-cai && bash scripts/kaggle.sh save     # ALWAYS before the session ends
   ```
Present it as "0.5B on free T4s; mechanics identical to the production recipe."

Judges/critics default to a local `Qwen/Qwen2.5-7B-Instruct` (`kind: hf`). To use Claude instead,
`export ANTHROPIC_API_KEY=…` and set `kind: anthropic` in the config. `kind: self` reuses the
policy (pure CAI, weak at 1.5B; use it only to see why the paper needed a capable model).

---

## Day 1 — Eval harness first, then SFT

```bash
python -m minicai.build_data --source hf                 # ~10k SFT, 3k red-team, 3k helpful, eval sets, decontaminated
cat data/manifest.json                                   # counts + contamination report
python -m minicai.sft --config configs/sft.yaml          # hand-written loop, resumes if killed
python -m minicai.evals --config configs/eval.yaml model=runs/sft/final   # becomes the reference for all later stages
pip install trl && python -m minicai.trl_reference sft --config configs/sft.yaml train.max_steps=200 out_dir=runs/sft_trl
python drills/run.py 01 && python drills/run.py 02       # night: attention, sampling
```
**Done when:** `runs/sft/samples.jsonl` reads like a coherent assistant, `hit_max_rate` < 5%,
your loss curve and TRL's agree within noise, `python -m minicai.report` shows the SFT row.

## Day 2 — CAI supervised phase

```bash
python -m minicai.cai_revise --config configs/cai_revise.yaml
head -n 20 data/cai_trace.jsonl | python -m json.tool --json-lines | less    # READ the critiques. Bad critic = bad model.
python -m minicai.sft --config configs/slcai.yaml
python -m minicai.evals --config configs/eval.yaml model=runs/slcai/final
python drills/run.py 03 && python drills/run.py 04
```
**Done when:** `harmless_rate` up vs SFT and `overrefusal_rate` not sharply up. If over-refusal
jumps, raise `data.helpful_ratio` or edit `harm-not-evasive` in the constitution and rerun.

## Day 3 — RLAIF preferences and the reward model

```bash
python -m minicai.label_prefs --config configs/label_prefs.yaml
cat data/labeler_report.json      # sanity acc, human agreement, position-bias rate, chosen_is_longer_rate
python -m minicai.train_rm --config configs/rm.yaml
cat runs/rm/eval_rm.json          # acc, acc_chosen_longer vs acc_chosen_shorter, reward_length_corr, ece
python drills/run.py 05 && python drills/run.py 06
```
**Done when:** RM held-out accuracy clearly > 0.5, and you have written down its length correlation.
Write one sentence: "my labeler prefers longer answers X% of the time, so my RM will…"

## Day 4 — DPO and online RL

```bash
for b in 0.05 0.1 0.5; do
  python -m minicai.train_dpo --config configs/dpo.yaml beta=$b out_dir=runs/dpo_b$b run_name=dpo_b$b
  python -m minicai.evals --config configs/eval.yaml model=runs/dpo_b$b/final
done
python -m minicai.trl_reference dpo --config configs/dpo.yaml train.max_steps=100 out_dir=runs/dpo_trl   # cross-check
python -m minicai.train_rl --config configs/rl.yaml
python -m minicai.evals --config configs/eval.yaml model=runs/rl/final
python -m minicai.report
python drills/run.py 07
```
Then **find reward hacking yourself**: plot `reward`, `rm_raw`, `len`, `kl` from `runs/rl/metrics.jsonl`,
read `runs/rl/samples.jsonl` over steps. If reward rises with length while judge win rate is flat, fix
it (`train.length_penalty=0.002`, a higher `train.kl_coef`, or `length_normalize: true` for DPO) and rerun.
**Done when:** DPO + RL rows are in RESULTS.md and one hacking case is documented with before/after samples.

## Day 5 — Scale: distributed and reproducible

```bash
bash scripts/bench.sh                                     # single vs DDP vs FSDP, ± gradient checkpointing
accelerate launch --config_file configs/accelerate/fsdp.yaml -m minicai.sft --config configs/sft.yaml out_dir=runs/sft_fsdp
accelerate launch --config_file configs/accelerate/fsdp.yaml -m minicai.train_dpo --config configs/dpo.yaml out_dir=runs/dpo_fsdp
# Resume drill: kill the FSDP run with Ctrl-C after a checkpoint, rerun the same command, check the loss curve is continuous.
python drills/run.py 08 && python drills/run.py 09
```
Edit `num_processes` in `configs/accelerate/*.yaml` to your GPU count.
**Done when:** the bench table is in RESULTS.md, and a killed FSDP run resumed cleanly.
Explain each row: what is sharded, what is communicated, why memory/throughput moved.

## Day 6 — Incident drills

```bash
FAULT=none bash scripts/drill.sh          # clean baseline first: learn what "normal" looks like
bash scripts/drill.sh                     # random hidden fault, guardrails OFF; repeat 10x
python -m minicai.faults reveal           # only after writing the postmortem
python -m minicai.faults disarm
python drills/run.py 10
```
Time-box 45 min each. Write each postmortem in `POSTMORTEMS.md` **before** revealing.
`bash scripts/drill.sh tiny` runs the same workflow on CPU (useful to practise without a GPU).
Then one clean ablation (e.g. SL-CAI data on vs off before DPO) with evals.

## Day 7 — Write-up and rehearsal

```bash
python -m minicai.report                 # final results table into RESULTS.md
python drills/run.py reset all && python drills/run.py all    # timed re-run of every drill
```
Fill `WRITEUP.md`. Record a 10-minute walkthrough out loud (STUDY.md has the script).
**Done when:** someone else can clone the repo and reproduce SFT → DPO from this README.

---

## Where each failure shows up

| Fault (`python -m minicai.faults list`) | Symptom | Guardrail |
|---|---|---|
| `prompt_loss` | low SFT loss, model parrots/role-plays user turns | `check_loss_mask` |
| `template_mismatch` | DPO margin up, eval quality down | template hash vs parent `meta.json` |
| `ref_unfrozen` | `ref_param_drift` > 0, implicit reward drifts | `assert_frozen` |
| `eos_dropped` | `hit_max_rate` ≈ 100%, rambling | `check_loss_mask` + eval `hit_max_rate` |
| `rl_lr_10x` | `kl` spikes, entropy collapses, gibberish | `kl_stop`, `entropy_alert` |
| `rm_length_bias` | `reward_length_corr` high, RL length climbs | RM eval split by length |
| `eval_leak` | one stage's eval jumps unrealistically | n-gram decontamination |
| `padding_right` | garbage only at batch size > 1 | batch-equality test |
| `resume_no_optim` | loss spike right after resume | exact-resume test |
| `pref_swap_half` | DPO/RM accuracy stuck near 0.5 | labeler sanity set |

## Honest scope

This is a 1.5B model on one node, with an open 7B (or Claude) as critic and judge. It reproduces
the *mechanics* of production post-training and the failure modes that matter, not frontier results.
Say exactly that when you present it.
