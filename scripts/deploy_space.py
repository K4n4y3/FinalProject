"""Deploy the service to a Hugging Face Docker Space (used by .github/workflows/deploy.yml).

Uploads only what the image needs (Dockerfile, lock file, package source, Space README) and sets the Space
variables that pin the model bundle version. Environment: HF_TOKEN, HF_SPACE, HF_MODEL_REPO, HF_MODEL_REVISION.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "src"]


def main() -> None:
    token, space = os.environ["HF_TOKEN"], os.environ["HF_SPACE"]
    model_repo = os.environ["HF_MODEL_REPO"]
    revision = os.environ.get("HF_MODEL_REVISION") or None
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        for name in FILES:
            src = ROOT / name
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst / name)
        readme = (ROOT / "deploy" / "space_README.md").read_text(encoding="utf-8")
        (dst / "README.md").write_text(readme.replace("{{MODEL_REPO}}", model_repo), encoding="utf-8")
        commit = api.upload_folder(
            folder_path=tmp,
            repo_id=space,
            repo_type="space",
            commit_message=f"deploy {os.environ.get('GITHUB_SHA', 'manual')[:7]}",
            delete_patterns=["src/**"],
        )
    api.add_space_variable(space, "MENU_HF_REPO", model_repo)
    if revision:
        api.add_space_variable(space, "MENU_HF_REVISION", revision)
    api.add_space_variable(space, "MENU_LLM_THREADS", "2")
    print(f"deployed {commit.oid[:7]} -> https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
