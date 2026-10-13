# Shared overrides that shrink every stage to a 2-layer CPU model for smoke tests.
# Sourced by scripts/smoke.sh.
TINY_TRAIN="train.batch_size=4 train.grad_accum=1 train.mixed_precision=no train.gradient_checkpointing=false train.log_every=2 train.eval_every=1000 train.save_every=1000 train.peak_tflops=null"
TINY_JUDGE="kind=self"
