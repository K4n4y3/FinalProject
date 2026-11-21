"""Model bundle: every artifact the service needs, plus a manifest that pins versions and checksums.

artifacts/
  manifest.json          bundle version, data SHA-256, file SHA-256s, metrics, final generator variant
  classifier.joblib      TF-IDF + LinearSVC + temperature
  retriever.joblib       BM25 (+ optional dense embeddings) over the TRAIN split
  generator/lora/        LoRA adapter from Assignment 4
  generator/lora_rag/    LoRA adapter trained with retrieved context (final project)
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from menu import __version__
from menu.data import LABELS, check_no_leakage, load_split, sha256
from menu.metrics import cls_metrics
from menu.models.classifier import DishClassifier
from menu.rag.retriever import DENSE_MODEL, Encoder, RecipeRetriever

MANIFEST = "manifest.json"
DEFAULT_GENERATOR = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "base_revision": None,
    "final_variant": {"name": "Qwen + LoRA (A4)", "adapter": "lora", "rag_k": 0, "decoding": "beam4"},
    "max_new_tokens": 24,
}


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST and not p.name.startswith(".")
    }


def build_bundle(
    data_csv: str | Path,
    out_dir: str | Path,
    bundle_version: str,
    adapters: dict[str, str | Path] | None = None,
    generator: dict | None = None,
    generator_eval: dict | None = None,
    encoder: Encoder | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_split(data_csv)
    leaks = check_no_leakage(df)
    if any(leaks.values()):
        raise ValueError(f"split leakage detected: {leaks}")
    lab = df[df.course.notna()]
    tr, va, te = (lab[lab.split == s] for s in ("train", "validation", "test"))

    clf = DishClassifier().fit(tr, augment=True)
    temperature = clf.calibrate(va)
    clf.save(out / "classifier.joblib")
    cls_report = {
        "test_raw_lines": cls_metrics(te.course, clf.predict(te.ing_raw)),
        "test_ingredient_names": cls_metrics(te.course, clf.predict(te.ing_clean)),
        "validation_raw_lines": cls_metrics(va.course, clf.predict(va.ing_raw)),
        "temperature": temperature,
        "labels": LABELS,
    }
    oov_val = float(np.mean([clf.oov_rate(x) for x in df[df.split == "validation"].ing_clean]))

    retriever = RecipeRetriever().fit(df[df.split == "train"], encoder=encoder)
    retriever.save(out / "retriever.joblib")

    gen_dir = out / "generator"
    for name, src in (adapters or {}).items():
        dst = gen_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("README.md", "*.bin.tmp", "train_report.json"))
    available = (
        sorted(p.name for p in gen_dir.iterdir() if (p / "adapter_config.json").exists()) if gen_dir.exists() else []
    )
    gen_cfg = {**DEFAULT_GENERATOR, **(generator or {})}
    if not available and not (generator or {}).get("final_variant"):
        gen_cfg["final_variant"] = {"name": "Qwen zero-shot", "adapter": None, "rag_k": 0, "decoding": "beam4"}
    if gen_cfg["final_variant"].get("adapter") not in [*available, None]:
        raise ValueError(f"final adapter {gen_cfg['final_variant']['adapter']!r} not in bundle {available}")

    manifest = {
        "bundle_version": bundle_version,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "file": Path(data_csv).name,
            "sha256": sha256(data_csv),
            "rows": {k: int(v) for k, v in df.split.value_counts().items()},
            "leakage_check": leaks,
        },
        "classifier": {
            "type": "TF-IDF(1-2) + LinearSVC(C=0.3, balanced), trained on raw+clean",
            **cls_report,
        },
        "retriever": {
            "type": "BM25(ingredients) + BM25(title)" + (f" + dense {DENSE_MODEL} (RRF)" if encoder else ""),
            "kb": "train split only",
            "documents": len(retriever),
            "dense": encoder is not None,
            "dense_model": DENSE_MODEL if encoder is not None else None,
        },
        "generator": {**gen_cfg, "adapters": available, "evaluation": generator_eval or {}},
        "monitoring_baseline": {"oov_rate_validation": oov_val},
    }
    manifest["files"] = _hash_tree(out)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def read_manifest(root: str | Path) -> dict:
    return json.loads((Path(root) / MANIFEST).read_text(encoding="utf-8"))


def verify_bundle(root: str | Path) -> list[str]:
    """Return the list of files whose checksum differs from the manifest (empty list = bundle intact)."""
    root = Path(root)
    expected = read_manifest(root)["files"]
    actual = _hash_tree(root)
    return sorted(k for k in expected if actual.get(k) != expected[k])


def ensure_bundle(root: Path, hf_repo: str | None, revision: str | None, repo_type: str = "model") -> Path:
    if (root / MANIFEST).exists():
        return root
    if not hf_repo:
        raise FileNotFoundError(f"{root}/{MANIFEST} not found and MENU_HF_REPO is not set")
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=hf_repo, revision=revision, repo_type=repo_type, local_dir=str(root))
    return Path(path)


def generation_table_to_dict(table: pd.DataFrame) -> list[dict]:
    return json.loads(table.to_json(orient="records"))
