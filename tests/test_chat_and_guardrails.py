"""Chat template, loss masks, and every guardrail tied to a Day 6 drill."""
import json

import pytest
import torch

from minicai import chat, checks
from minicai.data import contamination, load_pairs, write_jsonl
from minicai.models import assert_frozen, freeze, generate

CONV = [{"role": "user", "content": "what is 2 plus 2?"}, {"role": "assistant", "content": "2 plus 2 is 4."}]


def _labelled_text(tok, e):
    return tok.decode([l for l in e["labels"] if l != chat.IGNORE])


def test_mask_covers_only_assistant_and_end(tiny):
    _, tok = tiny
    e = chat.encode_chat(tok, CONV)
    assert _labelled_text(tok, e) == "2 plus 2 is 4." + chat.IM_END
    assert len(e["input_ids"]) == len(e["labels"])
    checks.check_loss_mask(tok, [e])


def test_generation_prompt_is_prefix_of_training_tokens(tiny):
    """What the model sees at sampling time must match training, token for token."""
    _, tok = tiny
    full = chat.encode_chat(tok, CONV)["input_ids"]
    prompt = tok(chat.prompt_text(CONV[:1]), add_special_tokens=False)["input_ids"]
    assert full[: len(prompt)] == prompt


def test_fault_prompt_loss_trips_guardrail(tiny, monkeypatch):
    _, tok = tiny
    monkeypatch.setenv("MINICAI_FAULT", "prompt_loss")
    with pytest.raises(checks.GuardrailError, match="prompt"):
        checks.check_loss_mask(tok, [chat.encode_chat(tok, CONV)])


def test_fault_eos_dropped_trips_guardrail(tiny, monkeypatch):
    _, tok = tiny
    monkeypatch.setenv("MINICAI_FAULT", "eos_dropped")
    with pytest.raises(checks.GuardrailError, match="stop"):
        checks.check_loss_mask(tok, [chat.encode_chat(tok, CONV)])


def test_fault_template_mismatch_trips_guardrail(tiny, tmp_path, monkeypatch):
    _, tok = tiny
    chat.set_stage("sft")
    (tmp_path / "meta.json").write_text(json.dumps({"template_hash": chat.template_hash(tok)}))
    chat.set_stage("dpo")
    checks.check_template(str(tmp_path), tok)  # clean: passes
    monkeypatch.setenv("MINICAI_FAULT", "template_mismatch")
    with pytest.raises(checks.GuardrailError, match="template"):
        checks.check_template(str(tmp_path), tok)
    chat.set_stage("sft")


def test_contamination_detects_leak():
    ev = ["please explain how photosynthesis works in plants step by step"]
    clean = contamination(["a totally different sentence about cooking pasta at home tonight"], ev)
    leaked = contamination(["user asked: please explain how photosynthesis works in plants step by step"], ev)
    assert clean["rate"] == 0 and leaked["rate"] == 1


def test_pref_swap_fault(tmp_path, monkeypatch):
    rows = [{"prompt": str(i), "chosen": "good", "rejected": "bad"} for i in range(200)]
    write_jsonl(tmp_path / "p.jsonl", rows)
    assert all(r["chosen"] == "good" for r in load_pairs(tmp_path / "p.jsonl", "dpo"))
    monkeypatch.setenv("MINICAI_FAULT", "pref_swap_half")
    swapped = sum(r["chosen"] == "bad" for r in load_pairs(tmp_path / "p.jsonl", "dpo"))
    assert 70 < swapped < 130


def test_rm_length_bias_fault(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "p.jsonl", [{"prompt": "x", "chosen": "short", "rejected": "much much longer"}])
    monkeypatch.setenv("MINICAI_FAULT", "rm_length_bias")
    assert load_pairs(tmp_path / "p.jsonl", "rm")[0]["chosen"] == "much much longer"
    assert load_pairs(tmp_path / "p.jsonl", "dpo")[0]["chosen"] == "short"


def test_reference_must_be_frozen(tiny):
    model, _ = tiny
    import copy
    ref = copy.deepcopy(model)
    with pytest.raises(RuntimeError):
        assert_frozen(ref)
    assert_frozen(freeze(ref))


def test_batched_greedy_equals_single(tiny):
    """Left padding: a prompt's greedy output must not depend on its batch-mates."""
    model, tok = tiny
    prompts = [[{"role": "user", "content": p}] for p in ["hi", "what is 3 plus 4 and why is it so", "bread", "zurich"]]
    single = [generate(model, tok, [p], max_new_tokens=8, greedy=True)[0]["token_ids"] for p in prompts]
    batched = [o["token_ids"] for o in generate(model, tok, prompts, max_new_tokens=8, greedy=True, batch_size=4)]
    assert single == batched


def test_padding_right_fault_breaks_batch_equality(tiny, monkeypatch):
    model, tok = tiny
    prompts = [[{"role": "user", "content": p}] for p in ["hi", "what is 3 plus 4 and why is it so long a question"]]
    single = [generate(model, tok, [p], max_new_tokens=8, greedy=True)[0]["token_ids"] for p in prompts]
    monkeypatch.setenv("MINICAI_FAULT", "padding_right")
    batched = [o["token_ids"] for o in generate(model, tok, prompts, max_new_tokens=8, greedy=True, batch_size=2)]
    assert single != batched


def test_refusal_heuristic():
    assert checks.is_refusal("I'm sorry, but I can't help with that.")
    assert not checks.is_refusal("Sure! Here's how to make bread: ...")


def test_eval_leak_fault_trips_build_data_guardrail(tmp_path, monkeypatch):
    import subprocess, sys, os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), MINICAI_FAULT_FILE=str(tmp_path / ".f"))
    base = [sys.executable, "-m", "minicai.build_data", "--source", "synthetic", "--out", str(tmp_path / "d"),
            "--sft-n", "60", "--eval-n", "8"]
    assert subprocess.run(base, env=env, capture_output=True).returncode == 0
    env["MINICAI_FAULT"] = "eval_leak"
    r = subprocess.run(base, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "Guardrail" in r.stderr


def test_fault_cli_hides_name(tmp_path, monkeypatch, capsys):
    from minicai import faults
    faults.FAULT_FILE = tmp_path / ".fault"
    faults.FAULT_FILE.write_text(faults._encode("eos_dropped"))
    assert "eos_dropped" not in faults.FAULT_FILE.read_text()
    assert faults.active() == "eos_dropped" and faults.on("eos_dropped")


def test_parse_hh_rlhf_transcript():
    from minicai.build_data import parse_hh
    t = "\n\nHuman: How do I bake bread?\n\nAssistant: Mix flour, water, salt, yeast.\n\nHuman: Thanks!\n\nAssistant: Enjoy."
    m = parse_hh(t)
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant"]
    assert m[0]["content"] == "How do I bake bread?" and m[-1]["content"] == "Enjoy."
