#!/usr/bin/env bash
# End-to-end plumbing test on a 2-layer CPU model with synthetic data (~2-4 min).
# Run this before renting a GPU, and after every code change.
#   bash scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export MINICAI_TINY_DIR=smoke/runs/tiny-base
D=smoke/data
R=smoke/runs
rm -rf smoke && mkdir -p smoke
source configs/tiny/overrides.sh
NG="guardrails=true"

echo "== data";      python -m minicai.build_data --source synthetic --out $D --sft-n 200 --redteam-n 24 --helpful-n 24 --eval-n 8 --capability-n 16
echo "== sft";       python -m minicai.sft --config configs/sft.yaml model.init=tiny out_dir=$R/sft data.train=$D/sft_train.jsonl data.eval=$D/sft_eval.jsonl data.max_len=128 train.epochs=1 train.max_steps=20 train.lr=1e-3 train.save_every=10 sample_tokens=16 $TINY_TRAIN $NG
echo "== eval sft";  python -m minicai.evals --config configs/eval.yaml model=$R/sft/final reference=$R/sft/final/eval_generations.jsonl judge.kind=self data.helpful=$D/helpful_eval.jsonl data.redteam=$D/redteam_eval.jsonl data.overrefusal=$D/overrefusal_eval.jsonl data.capability=$D/capability_eval.jsonl gen.max_new_tokens=16 "train_files=[$D/sft_train.jsonl]"
echo "== cai";       python -m minicai.cai_revise --config configs/cai_revise.yaml model.init=$R/sft/final critic.kind=self critic.max_new_tokens=16 data.prompts=$D/redteam_train.jsonl data.helpful_mix=$D/sft_train.jsonl gen.max_new_tokens=16 out.dir=$D
echo "== slcai";     python -m minicai.sft --config configs/slcai.yaml model.init=$R/sft/final out_dir=$R/slcai data.train=$D/slcai_train.jsonl data.eval=$D/sft_eval.jsonl data.max_len=128 train.epochs=1 train.max_steps=8 train.lr=1e-3 sample_tokens=16 $TINY_TRAIN $NG
echo "== prefs";     python -m minicai.label_prefs --config configs/label_prefs.yaml model.init=$R/slcai/final labeler.kind=self data.harmless_prompts=$D/redteam_train.jsonl data.helpful_prompts=$D/helpful_train.jsonl agreement.sanity=$D/labeler_sanity.jsonl agreement.human_pairs=$D/hh_pairs_eval.jsonl agreement.limit=8 sanity_min_acc=0 drop_inconsistent=false min_confidence=0 gen.max_new_tokens=16 out.dir=$D
echo "== rm";        python -m minicai.train_rm --config configs/rm.yaml model.init=$R/slcai/final out_dir=$R/rm data.train=$D/prefs_train.jsonl data.eval=$D/prefs_eval.jsonl data.max_len=128 train.max_steps=6 train.lr=1e-3 $TINY_TRAIN $NG
echo "== dpo";       python -m minicai.train_dpo --config configs/dpo.yaml model.init=$R/slcai/final model.ref=$R/slcai/final out_dir=$R/dpo data.train=$D/prefs_train.jsonl data.eval=$D/prefs_eval.jsonl data.max_len=128 train.max_steps=6 train.lr=1e-4 $TINY_TRAIN $NG
echo "== rl";        python -m minicai.train_rl --config configs/rl.yaml model.init=$R/slcai/final model.ref=$R/slcai/final reward_model=$R/rm/final out_dir=$R/rl "data.prompts=[$D/redteam_train.jsonl,$D/helpful_train.jsonl]" gen.max_new_tokens=16 train.steps=3 train.prompts_per_step=2 train.group_size=4 train.micro_batch=4 train.lr=1e-4 train.gradient_checkpointing=false train.sample_every=1 $NG
for s in slcai dpo rl; do
  echo "== eval $s"; python -m minicai.evals --config configs/eval.yaml model=$R/$s/final reference=$R/sft/final/eval_generations.jsonl judge.kind=self data.helpful=$D/helpful_eval.jsonl data.redteam=$D/redteam_eval.jsonl data.overrefusal=$D/overrefusal_eval.jsonl data.capability=$D/capability_eval.jsonl gen.max_new_tokens=16 "train_files=[$D/sft_train.jsonl]"
done
echo "== report";    python -m minicai.report --runs $R --results /dev/null
echo "SMOKE OK"
