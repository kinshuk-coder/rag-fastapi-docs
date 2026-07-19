"""
ingestion/fetch_docs.py

Clones the FastAPI repo (shallow) and gives you the paths to the
tutorial/ and advanced/ docs subfolders we care about.
"""

import os
import shutil
from git import Repo

# Anchor every path to the project root (one level up from this file's
# own folder), NOT the current working directory. Without this, running
# `python fetch_docs.py` from a different folder than intended silently
# writes/reads data in the wrong place - this bit us in Milestone 3
# when embed.py couldn't find chunks.jsonl because chunk_docs.py had
# written it relative to a different working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO_URL = "https://github.com/fastapi/fastapi.git"
CLONE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fastapi_repo")

# The subfolders inside the cloned repo we actually want to index
SUBFOLDERS_OF_INTEREST = [
    os.path.join("docs", "en", "docs", "tutorial"),
    os.path.join("docs", "en", "docs", "advanced"),
]


def fetch_docs(force: bool = False) -> list[str]:
    """
    Clones the FastAPI repo shallowly into CLONE_PATH.
    Returns a list of absolute paths to the subfolders of interest.

    If force=True, deletes any existing clone first (useful if the
    repo was updated upstream and you want a fresh copy).
    """
    if force and os.path.exists(CLONE_PATH):
        shutil.rmtree(CLONE_PATH)

    if not os.path.exists(CLONE_PATH):
        print(f"Cloning {REPO_URL} into {CLONE_PATH} (depth=1)...")
        Repo.clone_from(REPO_URL, CLONE_PATH, depth=1)
        print("Clone complete.")
    else:
        print(f"Repo already exists at {CLONE_PATH}, skipping clone.")

    resolved_paths = []
    for subfolder in SUBFOLDERS_OF_INTEREST:
        full_path = os.path.join(CLONE_PATH, subfolder)
        if not os.path.isdir(full_path):
            raise FileNotFoundError(
                f"Expected subfolder not found: {full_path}. "
                f"The FastAPI repo structure may have changed."
            )
        resolved_paths.append(full_path)

    return resolved_paths


if __name__ == "__main__":
    paths = fetch_docs()
    print("Docs subfolders ready at:")
    for p in paths:
        num_md_files = len([f for f in os.listdir(p) if f.endswith(".md")])
        print(f"  {p}  ({num_md_files} top-level .md files)")