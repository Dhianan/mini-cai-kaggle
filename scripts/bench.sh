#!/usr/bin/env bash
# Day 5: throughput and memory across parallelism strategies, same model and data.
#   bash scripts/bench.sh            # uses all visible GPUs for ddp/fsdp
# Prints a markdown table; copy it into RESULTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
STEPS=${STEPS:-60}
COMMON="train.max_steps=$STEPS train.save_every=100000 train.eval_every=100000 train.log_every=10 data.pack=true data.eval=null"

run () {  # name accel_config nproc extra...
  local name=$1 cfg=$2 n=$3; shift 3
  echo "== $name"
  accelerate launch --config_file configs/accelerate/$cfg.yaml --num_processes $n \
    -m minicai.sft --config configs/sft.yaml out_dir=runs/bench/$name $COMMON "$@"
}

run single_bs8          single 1 train.batch_size=8
run single_bs8_gc       single 1 train.batch_size=8 train.gradient_checkpointing=true
run single_bs16_gc      single 1 train.batch_size=16 train.gradient_checkpointing=true
if [ "$NGPU" -ge 2 ]; then
  run ddp_bs8           ddp  "$NGPU" train.batch_size=8
  run fsdp_bs8          fsdp "$NGPU" train.batch_size=8
  run fsdp_bs16_gc      fsdp "$NGPU" train.batch_size=16 train.gradient_checkpointing=true
fi

python - <<'EOF'
import json, pathlib
rows = []
for d in sorted(pathlib.Path("runs/bench").iterdir()):
    s = json.loads((d / "summary.json").read_text())
    m = [json.loads(l) for l in open(d / "metrics.jsonl") if '"mfu"' in l or '"tok_per_s"' in l]
    mfu = [r["mfu"] for r in m[1:] if "mfu" in r]  # skip warm-up window
    mem = max((r.get("peak_mem_gb") or 0) for r in m) if m else 0
    rows.append((d.name, s["world"], s["mean_tok_per_s"], sum(mfu) / len(mfu) if mfu else float("nan"), mem))
print("| Run | GPUs | Tokens/s (total) | MFU | Peak mem/GPU (GB) |\n|---|---|---|---|---|")
for n, w, t, f, m in rows:
    print(f"| {n} | {w} | {t:,.0f} | {f:.1%} | {m:.1f} |")
EOF
