#!/usr/bin/env bash
# Day 6: one blind incident drill.
#
#   bash scripts/drill.sh            # arms a random fault, runs a short pipeline UNPROTECTED
#   bash scripts/drill.sh tiny       # same on the CPU tiny model (practise the workflow offline)
#   FAULT=none bash scripts/drill.sh # clean baseline run to compare against (do this first)
#
# Then diagnose from runs/drill/*/metrics.jsonl, eval_results.json, samples.jsonl and
# data/manifest.json ONLY. Write your postmortem in POSTMORTEMS.md first, then:
#   python -m minicai.faults reveal
#   python -m minicai.faults disarm
#
# Needs Day 1-3 artifacts (data/, data/prefs_*.jsonl, runs/slcai/final, runs/rm/final)
# unless run in tiny mode, which builds everything from synthetic data.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
MODE=${1:-gpu}
R=runs/drill
rm -rf $R && mkdir -p $R
if [ "${FAULT:-}" = none ]; then python -m minicai.faults disarm; else python -m minicai.faults arm ${FAULT:+--name $FAULT}; fi
G="guardrails=false"

if [ "$MODE" = tiny ]; then
  export MINICAI_TINY_DIR=$R/tiny-base
  D=$R/data
  T="train.mixed_precision=no train.grad_accum=1 train.gradient_checkpointing=false train.peak_tflops=null train.log_every=5"
  python -m minicai.build_data --source synthetic --out $D --sft-n 300 --redteam-n 24 --helpful-n 24 --eval-n 8 --no-guardrails
  python -m minicai.sft --config configs/sft.yaml model.init=tiny out_dir=$R/sft data.train=$D/sft_train.jsonl data.eval=$D/sft_eval.jsonl data.max_len=96 train.batch_size=8 train.epochs=30 train.max_steps=120 train.lr=2e-3 train.save_every=60 train.stop_after=60 sample_tokens=32 $T $G
  python -m minicai.sft --config configs/sft.yaml model.init=tiny out_dir=$R/sft data.train=$D/sft_train.jsonl data.eval=$D/sft_eval.jsonl data.max_len=96 train.batch_size=8 train.epochs=30 train.max_steps=120 train.lr=2e-3 train.save_every=60 sample_tokens=32 $T $G
  python -m minicai.label_prefs --config configs/label_prefs.yaml model.init=$R/sft/final labeler.kind=self data.harmless_prompts=$D/redteam_train.jsonl data.helpful_prompts=$D/helpful_train.jsonl agreement.sanity=$D/labeler_sanity.jsonl agreement.human_pairs=$D/hh_pairs_eval.jsonl agreement.limit=8 sanity_min_acc=0 drop_inconsistent=false min_confidence=0 gen.max_new_tokens=24 out.dir=$D $G
  python -m minicai.train_rm  --config configs/rm.yaml  model.init=$R/sft/final out_dir=$R/rm data.train=$D/prefs_train.jsonl data.eval=$D/prefs_eval.jsonl data.max_len=96 train.batch_size=4 train.max_steps=20 train.lr=1e-4 $T $G
  python -m minicai.train_dpo --config configs/dpo.yaml model.init=$R/sft/final model.ref=$R/sft/final out_dir=$R/dpo data.train=$D/prefs_train.jsonl data.eval=$D/prefs_eval.jsonl data.max_len=96 train.batch_size=4 train.max_steps=20 train.lr=1e-4 train.eval_every=5 $T $G
  python -m minicai.train_rl  --config configs/rl.yaml  model.init=$R/sft/final model.ref=$R/sft/final reward_model=$R/rm/final out_dir=$R/rl "data.prompts=[$D/redteam_train.jsonl,$D/helpful_train.jsonl]" gen.max_new_tokens=24 train.steps=8 train.prompts_per_step=2 train.group_size=4 train.micro_batch=4 train.lr=1e-4 train.gradient_checkpointing=false $G
  for s in sft dpo rl; do
    python -m minicai.evals --config configs/eval.yaml model=$R/$s/final reference=$R/sft/final/eval_generations.jsonl judge.kind=self data.helpful=$D/helpful_eval.jsonl data.redteam=$D/redteam_eval.jsonl data.overrefusal=$D/overrefusal_eval.jsonl data.capability=$D/capability_eval.jsonl gen.max_new_tokens=32 "train_files=[$D/sft_train.jsonl]"
  done
else
  M=${DRILL_MODEL:-Qwen/Qwen2.5-0.5B}
  D=$R/data
  python -m minicai.build_data --source hf --out $D --sft-n 3000 --redteam-n 500 --helpful-n 500 --eval-n 100 --capability-n 200 --no-guardrails
  cp data/prefs_train.jsonl data/prefs_eval.jsonl $D/
  # SFT is "crashed" half way and resumed, so resume bugs have somewhere to show up.
  python -m minicai.sft --config configs/sft.yaml model.init=$M out_dir=$R/sft data.train=$D/sft_train.jsonl data.eval=$D/sft_eval.jsonl train.epochs=1 train.max_steps=200 train.save_every=100 train.stop_after=100 $G
  python -m minicai.sft --config configs/sft.yaml model.init=$M out_dir=$R/sft data.train=$D/sft_train.jsonl data.eval=$D/sft_eval.jsonl train.epochs=1 train.max_steps=200 train.save_every=100 $G
  python -m minicai.train_rm  --config configs/rm.yaml  model.init=$R/sft/final out_dir=$R/rm data.train=$D/prefs_train.jsonl data.eval=$D/prefs_eval.jsonl train.max_steps=150 $G
  python -m minicai.train_dpo --config configs/dpo.yaml model.init=$R/sft/final model.ref=$R/sft/final out_dir=$R/dpo data.train=$D/prefs_train.jsonl data.eval=$D/prefs_eval.jsonl train.max_steps=150 train.eval_every=25 $G
  python -m minicai.train_rl  --config configs/rl.yaml  model.init=$R/sft/final model.ref=$R/sft/final reward_model=$R/rm/final out_dir=$R/rl train.steps=40 $G
  for s in sft dpo rl; do
    python -m minicai.evals --config configs/eval.yaml model=$R/$s/final reference=$R/sft/final/eval_generations.jsonl data.helpful=$D/helpful_eval.jsonl data.redteam=$D/redteam_eval.jsonl data.overrefusal=$D/overrefusal_eval.jsonl data.capability=$D/capability_eval.jsonl data.limit=100 "train_files=[$D/sft_train.jsonl]"
  done
fi
python -m minicai.report --runs $R --results /dev/null
echo
echo "Drill run complete. Diagnose from logs only, write the postmortem, then: python -m minicai.faults reveal"
