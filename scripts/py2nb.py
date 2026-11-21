"""Convert a percent-format script (``# %%`` / ``# %% [markdown]`` cells) into a Jupyter notebook.

python scripts/py2nb.py notebooks/final_training.py notebooks/final_training.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(src: Path) -> dict:
    cells: list[dict] = []
    kind, buf = None, []

    def flush() -> None:
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": text.splitlines(keepends=True),
                }
            )

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(json.dumps(convert(src), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{dst}: {len(convert(src)['cells'])} cells")
