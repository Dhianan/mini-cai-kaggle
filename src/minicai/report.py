"""Build the results table from every eval_results.json under runs/.

    python -m minicai.report            # prints and rewrites the table in RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [("win_rate_vs_ref", "Win rate vs ref", "{:.1%}"), ("harmless_rate", "Harmless", "{:.1%}"),
        ("overrefusal_rate", "Over-refusal", "{:.1%}"), ("capability_acc", "Capability", "{:.1%}"),
        ("mean_len", "Mean len (tok)", "{:.0f}"), ("hit_max_rate", "No EOS", "{:.1%}")]
ORDER = ["sft", "slcai", "dpo", "rl"]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def stage_key(path: Path) -> tuple:
    name = path.parent.parent.name if path.parent.name == "final" else path.parent.name
    rank = next((i for i, s in enumerate(ORDER) if name.startswith(s)), len(ORDER))
    return rank, name


def table(runs_dir: Path) -> str:
    files = sorted(runs_dir.glob("**/eval_results.json"), key=stage_key)
    head = "| Run | " + " | ".join(c[1] for c in COLS) + " |\n|---|" + "---|" * len(COLS) + "\n"
    lines = []
    for f in files:
        r = json.loads(f.read_text())
        name = f.parent.parent.name if f.parent.name == "final" else f.parent.name
        cells = []
        for key, _, fmt in COLS:
            v = r.get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) and v == v else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="RESULTS.md")
    a = ap.parse_args()
    t = table(Path(a.runs))
    print(t)
    p = Path(a.results)
    if p.exists() and START in p.read_text():
        s = p.read_text()
        s = s[: s.index(START) + len(START)] + "\n" + t + "\n" + s[s.index(END):]
        p.write_text(s)


if __name__ == "__main__":
    main()
