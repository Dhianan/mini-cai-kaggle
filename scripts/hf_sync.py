"""Persist data/ and runs/ across free-tier sessions (Kaggle/Colab wipe the disk after 12 h).

    python scripts/hf_sync.py push      # end of a session
    python scripts/hf_sync.py pull      # start of a session

Uses a PRIVATE Hugging Face model repo (free). Needs HF_TOKEN (write) and HF_REPO=<you>/mini-cai-runs.
On Kaggle: Add-ons -> Secrets -> HF_TOKEN, HF_REPO.
"""
import os
import sys

from huggingface_hub import HfApi, snapshot_download

FOLDERS = ["data", "runs"]
# Mid-run optimizer states are large; keep them only so an unfinished stage can resume.
IGNORE = ["**/__pycache__/**", "runs/tiny-base/**", "**/*.tmp"]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("push", "pull"):
        raise SystemExit(__doc__)
    repo, token = os.environ.get("HF_REPO"), os.environ.get("HF_TOKEN")
    if not repo or not token:
        raise SystemExit("Set HF_REPO and HF_TOKEN first (see docstring).")
    api = HfApi(token=token)
    if sys.argv[1] == "push":
        api.create_repo(repo, private=True, exist_ok=True)
        for f in FOLDERS:
            if os.path.isdir(f):
                api.upload_folder(repo_id=repo, folder_path=f, path_in_repo=f, ignore_patterns=IGNORE,
                                  commit_message=f"sync {f}")
                print(f"[sync] pushed {f}/ -> {repo}")
    else:
        snapshot_download(repo, token=token, local_dir=".", allow_patterns=[f"{f}/**" for f in FOLDERS])
        print(f"[sync] pulled {FOLDERS} from {repo}")


if __name__ == "__main__":
    main()
