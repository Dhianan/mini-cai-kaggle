"""Nightly drill runner.

    python drills/run.py 04               # test YOUR drills/exercises/d04_*.py
    python drills/run.py all              # all your exercises
    python drills/run.py 04 --solutions   # check the reference (after you're done!)
    python drills/run.py reset 04         # restore a fresh stub to practise again

Rules: no autocomplete, no AI, timer on (see the minutes in each file's
docstring), talk out loud. Log each attempt in drills/LOG.md.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "reset":
        pat = "d*.py" if args[1] == "all" else f"d{int(args[1]):02d}_*.py"
        for f in sorted((HERE / "stubs").glob(pat)):
            shutil.copy(f, HERE / "exercises" / f.name)
            print(f"reset {f.name}")
        return
    impl = "solutions" if "--solutions" in args else "exercises"
    target = args[0]
    cmd = [sys.executable, "-m", "pytest", "-q", str(HERE / "test_drills.py"), "-p", "no:cacheprovider",
           "-o", "addopts=", "-o", "testpaths=", "-o", "markers=" + " ".join(f"d{i:02d}" for i in range(1, 11))]
    if target != "all":
        cmd += ["-k", f"d{int(target):02d}"]
    raise SystemExit(subprocess.call(cmd, env=dict(os.environ, DRILL_IMPL=impl)))


if __name__ == "__main__":
    main()
