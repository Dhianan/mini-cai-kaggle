# Free-tier profile: Kaggle / Colab T4 (16 GB, fp16 only) or P100.
# Policy Qwen2.5-0.5B base, judge/critic/labeler Qwen2.5-3B-Instruct. Sourced by scripts/kaggle.sh.
# Same code, losses, guardrails and drills as the A100 profile; only scale and precision change.

POLICY=Qwen/Qwen2.5-0.5B
JUDGE=Qwen/Qwen2.5-3B-Instruct
# Harm + refusal metrics: a purpose-built safety classifier. Day 1: the 3B and 7B judges flagged 0-3% of
# human-rejected replies and ranked pairs near a coin flip (scripts/judge_check.py). Sits on GPU 1.
GUARD=Qwen/Qwen3Guard-Gen-4B

# T4 fp16 dense peak ~65 TFLOPs (used only for the MFU number in logs)
# batch 2 x accum 16 = same effective batch (32). Qwen's 152k-token vocabulary makes the logits the
# biggest activation: batch 4 x 768 tokens x 152k x fp32 is ~1.9 GB before its gradient, and that OOMs a T4.
K_TRAIN="train.batch_size=2 train.grad_accum=16 train.gradient_checkpointing=true train.peak_tflops=65"

K_SFT="model.init=$POLICY data.max_len=768 $K_TRAIN"
K_SLCAI="data.max_len=768 $K_TRAIN"
K_CRITIC="critic.model=$JUDGE critic.batch_size=16 data.limit=1500 gen.batch_size=32"
K_LABEL="labeler.model=$JUDGE labeler.batch_size=16 data.limit_each=1200 gen.batch_size=32"
K_RM="data.max_len=768 train.batch_size=4 train.grad_accum=4 train.gradient_checkpointing=true"
K_DPO="data.max_len=768 train.batch_size=2 train.grad_accum=16 train.peak_tflops=65"
K_RL="max_len=768 gen.batch_size=32 gen.max_new_tokens=192 train.micro_batch=4 train.steps=150"
# repetition_penalty 1.1: greedy decoding on a 0.5B model falls into loops (repeated list lines); applied to every
# stage's eval the same way, so comparisons stay fair. Never used for RL rollouts.
K_EVAL="judge.model=$JUDGE guard.model=$GUARD judge.batch_size=16 gen.batch_size=32 gen.max_new_tokens=320 gen.repetition_penalty=1.1 data.limit=200 data.capability_limit=300"
