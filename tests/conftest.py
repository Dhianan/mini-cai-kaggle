import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _no_fault(monkeypatch, tmp_path):
    """Each test starts clean; tests that need a fault set MINICAI_FAULT themselves."""
    monkeypatch.delenv("MINICAI_FAULT", raising=False)
    monkeypatch.setenv("MINICAI_FAULT_FILE", str(tmp_path / ".fault"))
    import minicai.faults as f
    f.FAULT_FILE = tmp_path / ".fault"


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """(model, tokenizer) for the 2-layer test model, built once per session."""
    d = tmp_path_factory.mktemp("tiny")
    os.environ["MINICAI_TINY_DIR"] = str(d / "tiny-base")
    from minicai.models import load_policy
    return load_policy("tiny", dtype="fp32")


@pytest.fixture(scope="session")
def tiny_dir(tiny):
    return os.environ["MINICAI_TINY_DIR"]
