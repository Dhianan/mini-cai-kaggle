#!/usr/bin/env bash
# One-time setup on a rented GPU box (Ubuntu + CUDA drivers, e.g. Lambda / RunPod / any H100 or A100 80GB).
#   bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0), "bf16:", torch.cuda.is_bf16_supported())
EOF
# Optional: huggingface-cli login (for gated datasets), wandb login, export ANTHROPIC_API_KEY=...
# Pre-download the models so the first run doesn't stall:
python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ['Qwen/Qwen2.5-1.5B','Qwen/Qwen2.5-0.5B','Qwen/Qwen2.5-7B-Instruct']]"
python -m pytest -q
echo "Ready. Start Day 1: see README.md"
