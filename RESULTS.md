# Results

Model: Qwen2.5-1.5B (base). Judge: <!-- fill: Qwen2.5-7B-Instruct / Claude model -->. Reference for win rate: the SFT model.
Regenerate the table with `python -m minicai.report`.

## Stages

<!-- results:start -->
| Run | Win rate vs ref | Harmless | Over-refusal | Capability | Mean len (tok) | No EOS |
|---|---|---|---|---|---|---|
<!-- results:end -->

## Labeler and reward model (Day 3)

| Metric | Value |
|---|---|
| Labeler sanity accuracy | |
| Labeler agreement with human labels (hh-rlhf) | |
| Position-bias rate | |
| Chosen-is-longer rate | |
| RM held-out accuracy | |
| RM accuracy, chosen longer / chosen shorter | |
| RM reward-length correlation | |

## DPO beta sweep (Day 4)

| β | Eval reward acc | Win rate vs SFT | Harmless | Over-refusal | Mean len |
|---|---|---|---|---|---|
| 0.05 | | | | | |
| 0.1 | | | | | |
| 0.5 | | | | | |

## Reward hacking found (Day 4)

- Symptom:
- Evidence (metrics + 2 samples before/after):
- Fix and result:

## Scaling (Day 5)

Paste the table from `bash scripts/bench.sh`.

## Ablation (Day 6)

- Question:
- Result:
