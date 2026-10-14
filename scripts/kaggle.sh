#!/usr/bin/env bash
# Free-tier runner (Kaggle T4/P100, or Colab T4). One command per day; resumable.
#   bash scripts/kaggle.sh setup        # every new session (installs, pulls saved data/runs)
#   bash scripts/kaggle.sh day1|day2|day3|day4|day6
#   bash scripts/kaggle.sh save         # end of every session (pushes data/runs to your private HF repo)
# Kaggle sessions die after 12 h: run `save` before that. Training resumes from the last checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/kaggle/overrides.sh
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation on 15 GB T4s
EVAL="python -m minicai.evals --config configs/eval.yaml $K_EVAL"

case "${1:-}" in
  setup)
    nvidia-smi
    pip install -q -r requirements.txt && pip install -q -U "transformers>=4.56" trl && pip install -q -e .
    python -c "import torch;print('GPUs:',torch.cuda.device_count(),torch.cuda.get_device_name(0),'cc',torch.cuda.get_device_capability())"
    python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ['$POLICY','$JUDGE']]"
    if [[ -n "${HF_REPO:-}" ]]; then python scripts/hf_sync.py pull || echo "[setup] nothing to pull yet"; fi
    python -m pytest -q
    ;;
  day1)
    [[ -f data/manifest.json ]] || python -m minicai.build_data --source hf
    cat data/manifest.json
    if grep -q '"overrefusal_eval": 0' data/manifest.json; then
      echo "[day1] STOP: overrefusal_eval is empty. Accept the terms at huggingface.co/datasets/walledai/XSTest,"
      echo "       then: rm data/manifest.json and rerun day1."; exit 1
    fi
    python -m minicai.sft --config configs/sft.yaml $K_SFT
    $EVAL model=runs/sft/final
    python -m minicai.report
    ;;
  judge-check)
    # Day 1 follow-up: test the harm judge against human labels (3B vs 7B in 4-bit). ~20 min GPU.
    pip install -q bitsandbytes
    python scripts/judge_check.py $JUDGE
    python scripts/judge_check.py Qwen/Qwen2.5-7B-Instruct --4bit
    ;;
  eval-sft)
    # Re-run only the SFT evaluation (no retraining), e.g. after changing eval decoding or the judge.
    $EVAL model=runs/sft/final
    python -m minicai.report
    ;;
  day1-trl)
    # Library cross-check: same data/LR/batch, 200 steps. Compare with runs/sft/metrics.jsonl.
    python -m minicai.trl_reference sft --config configs/sft.yaml $K_SFT train.max_steps=200 out_dir=runs/sft_trl
    python scripts/compare_trl.py runs/sft runs/sft_trl --grad-accum 16
    ;;
  day2)
    python -m minicai.cai_revise --config configs/cai_revise.yaml $K_CRITIC
    head -n 5 data/cai_trace.jsonl
    python -m minicai.sft --config configs/slcai.yaml $K_SLCAI
    $EVAL model=runs/slcai/final
    ;;
  day3)
    python -m minicai.label_prefs --config configs/label_prefs.yaml $K_LABEL
    cat data/labeler_report.json
    python -m minicai.train_rm --config configs/rm.yaml $K_RM
    cat runs/rm/eval_rm.json
    ;;
  day4)
    for b in 0.05 0.1 0.5; do
      python -m minicai.train_dpo --config configs/dpo.yaml $K_DPO beta=$b out_dir=runs/dpo_b$b run_name=dpo_b$b
      $EVAL model=runs/dpo_b$b/final
    done
    python -m minicai.train_rl --config configs/rl.yaml $K_RL
    $EVAL model=runs/rl/final
    python -m minicai.report
    ;;
  day5)
    # Needs 2 GPUs: Kaggle "GPU T4 x2". DDP only (FSDP on T4 works but teaches the same lesson slower).
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 -m minicai.sft --config configs/sft.yaml \
      $K_SFT out_dir=runs/sft_ddp run_name=sft_ddp train.max_steps=150
    ;;
  day6)
    FAULT=none bash scripts/drill.sh tiny
    bash scripts/drill.sh tiny
    ;;
  save)
    python scripts/hf_sync.py push
    ;;
  *)
    sed -n 2,7p "$0"; exit 1 ;;
esac
