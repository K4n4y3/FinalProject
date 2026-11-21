"""Build a small bundle from synthetic data (no downloads) — used by CI to smoke-test the Docker image.

python scripts/make_demo_bundle.py ci-artifacts
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.synthetic import make_split

from menu.bundle import build_bundle

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "ci-artifacts")
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "split.csv"
        make_split().to_csv(csv, index=False)
        m = build_bundle(csv, out, "ci-synthetic")
    print(out, m["bundle_version"], len(m["files"]), "files")
