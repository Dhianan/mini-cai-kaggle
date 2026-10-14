"""Model loading, the reward model, checkpoint saving, and batched generation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from . import faults
from .chat import CHATML_JINJA, IM_END, IM_START, prompt_text

PAD_FALLBACK = "<|endoftext|>"


# --------------------------------------------------------------------------- tiny model (CPU smoke tests)

_TINY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "how do i bake bread at home with simple ingredients",
    "i can not help with that request because it could cause harm",
    "here is a short answer to your question about python lists",
    "please explain the difference between a list and a tuple",
    "write a polite email asking for a meeting next week",
    "the capital of switzerland is bern and zurich is its largest city",
    "user assistant system response critique revision principle harmless helpful honest",
]


def build_tiny(out_dir: str | Path) -> str:
    """A 2-layer Llama with a freshly trained BPE tokenizer. Lets every stage run
    end-to-end on a laptop CPU in seconds. Not for results, only for plumbing."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        return str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = [PAD_FALLBACK, IM_START, IM_END]
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(_TINY_CORPUS * 20, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token=PAD_FALLBACK, eos_token=PAD_FALLBACK)
    tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids(IM_END),
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return str(out_dir)


def resolve_path(name_or_path: str) -> str:
    if name_or_path == "tiny":
        return build_tiny(os.environ.get("MINICAI_TINY_DIR", "runs/tiny-base"))
    return name_or_path


# --------------------------------------------------------------------------- loading

def native_bf16() -> bool:
    """True on Ampere+ (A100/H100/L4/RTX 30xx+). T4/P100/V100 emulate bf16 slowly or not at all."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def half_dtype() -> torch.dtype:
    """The 16-bit dtype this GPU actually runs fast: bf16 on Ampere+, fp16 on T4/P100 (free Kaggle/Colab)."""
    return torch.bfloat16 if native_bf16() else torch.float16


def _dtype(name: str | None) -> torch.dtype:
    if name in (None, "auto"):
        return half_dtype() if torch.cuda.is_available() else torch.float32
    if name == "bf16" and torch.cuda.is_available() and not native_bf16():
        return torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = PAD_FALLBACK if PAD_FALLBACK in tok.get_vocab() else tok.eos_token
    for t in (IM_START, IM_END):
        if t not in tok.get_vocab():
            raise ValueError(f"Tokenizer at {path} has no {t}; use a ChatML-vocab model (Qwen2.5).")
    tok.chat_template = CHATML_JINJA
    return tok


def load_policy(name_or_path: str, dtype: str | None = "fp32", device: str | None = None, load_in_4bit: bool = False):
    """Causal LM + tokenizer. Train in fp32 master weights with bf16 autocast
    (accelerate mixed_precision=bf16); load frozen models (ref, judge) in bf16."""
    from transformers import AutoModelForCausalLM

    path = resolve_path(name_or_path)
    tok = load_tokenizer(path)
    kwargs = {"dtype": _dtype(dtype)}
    if torch.cuda.is_available():
        kwargs["attn_implementation"] = "sdpa"
    if load_in_4bit:  # frozen judges only: a 7B judge in ~5.5 GB on a 16 GB T4 (pip install bitsandbytes)
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=_dtype(dtype))
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    if device and not load_in_4bit:
        model.to(device)
    return model, tok


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def assert_frozen(model: nn.Module, name: str = "reference") -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable:
        raise RuntimeError(f"Guardrail: {name} model has {trainable:,} trainable params; it must be frozen.")


def param_norm(model: nn.Module) -> float:
    with torch.no_grad():
        return float(torch.sqrt(sum((p.float() ** 2).sum() for p in model.parameters())))


def end_token_ids(tok) -> list[int]:
    ids = [tok.convert_tokens_to_ids(IM_END)]
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


# --------------------------------------------------------------------------- saving

def save_checkpoint(model, tok, out_dir: str | Path, accelerator=None, meta: dict | None = None) -> None:
    """Works for single GPU, DDP and FSDP (gathers a full state dict on rank 0)."""
    out_dir = Path(out_dir)
    if accelerator is not None:
        accelerator.wait_for_everyone()
        state = accelerator.get_state_dict(model)
        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped.save_pretrained(out_dir, state_dict=state, safe_serialization=True)
            tok.save_pretrained(out_dir)
        accelerator.wait_for_everyone()
    else:
        model.save_pretrained(out_dir, safe_serialization=True)
        tok.save_pretrained(out_dir)
    if meta is not None and (accelerator is None or accelerator.is_main_process):
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


# --------------------------------------------------------------------------- reward model

class RewardModel(nn.Module):
    """Transformer backbone + scalar head read at the last non-pad token."""

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.head.weight, std=1 / (hidden_size + 1) ** 0.5)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_policy(cls, name_or_path: str, dtype: str | None = "fp32"):
        from transformers import AutoModel

        path = resolve_path(name_or_path)
        tok = load_tokenizer(path)
        backbone = AutoModel.from_pretrained(path, dtype=_dtype(dtype))
        rm = cls(backbone, backbone.config.hidden_size)
        rm.head.to(_dtype(dtype))
        head_path = Path(path) / "value_head.pt"
        if head_path.exists():
            rm.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        return rm, tok

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Right padding: last real token is at sum(mask) - 1.
        last = attention_mask.long().sum(1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last]
        return self.head(pooled.to(self.head.weight.dtype)).squeeze(-1).float()

    def save(self, out_dir: str | Path, tok, meta: dict | None = None) -> None:
        out_dir = Path(out_dir)
        self.backbone.save_pretrained(out_dir, safe_serialization=True)
        torch.save(self.head.state_dict(), out_dir / "value_head.pt")
        tok.save_pretrained(out_dir)
        if meta is not None:
            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)


# --------------------------------------------------------------------------- generation

@torch.no_grad()
def generate(model, tok, prompts: list[list[dict]], max_new_tokens: int = 256, temperature: float = 0.7,
             top_p: float = 0.95, top_k: int = 0, batch_size: int = 16, greedy: bool = False,
             num_return_sequences: int = 1, repetition_penalty: float = 1.0) -> list[dict]:
    """Batched sampling. Returns one dict per (prompt, sample):
    {prompt_idx, text, token_ids, n_tokens, hit_max}. Left padding is required
    for decoder-only batched generation; fault `padding_right` breaks that."""
    was_training = model.training
    model.eval()
    side = "right" if faults.on("padding_right") else "left"
    old_side = tok.padding_side
    tok.padding_side = side
    device = next(model.parameters()).device
    stops = end_token_ids(tok)
    out: list[dict] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = [prompt_text(p) for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen_kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=stops, pad_token_id=tok.pad_token_id,
                              num_return_sequences=num_return_sequences)
            if repetition_penalty != 1.0:  # eval-time decoding only; never used for RL rollouts
                gen_kwargs.update(repetition_penalty=repetition_penalty)
            if greedy:
                gen_kwargs.update(do_sample=False)
            else:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k or None)
            seqs = model.generate(**enc, **gen_kwargs)
            new = seqs[:, enc["input_ids"].shape[1]:]
            for j in range(new.size(0)):
                ids = new[j].tolist()
                cut = len(ids)
                for k, t in enumerate(ids):
                    if t in stops:
                        cut = k
                        break
                hit_max = cut == len(ids) and not any(t in stops for t in ids)
                ids = [t for t in ids[:cut] if t != tok.pad_token_id]
                out.append({
                    "prompt_idx": i + j // num_return_sequences,
                    "text": tok.decode(ids, skip_special_tokens=True).strip(),
                    "token_ids": ids,
                    "n_tokens": len(ids),
                    "hit_max": hit_max,
                })
    finally:
        tok.padding_side = old_side
        if was_training:
            model.train()
    return out
