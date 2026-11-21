"""Shared fixtures: a synthetic recipe split, a bundle built from it and (optionally) a tiny random Qwen2 model.

Nothing here touches the network, so the suite runs in CI and offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from menu.bundle import build_bundle
from tests.synthetic import SIGNATURES, make_split  # noqa: F401  (re-exported for tests)

HAS_LLM = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))


@pytest.fixture(scope="session")
def split_csv(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "split.csv"
    make_split().to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def tiny_llm(tmp_path_factory) -> Path | None:
    """A 2-layer random Qwen2 + tokenizer with a chat template, plus two LoRA adapters ('lora', 'lora_rag')."""
    if not HAS_LLM:
        return None
    from tests.tiny_model import build_tiny

    return build_tiny(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory, split_csv, tiny_llm) -> Path:
    out = tmp_path_factory.mktemp("bundle")
    adapters = {n: tiny_llm / n for n in ("lora", "lora_rag")} if tiny_llm else {}
    gen = {"base_model": str(tiny_llm / "base")} if tiny_llm else {}
    if tiny_llm:
        gen["final_variant"] = {
            "name": "Qwen + LoRA-RAG",
            "adapter": "lora_rag",
            "rag_k": 3,
            "decoding": "beam4",
        }
    build_bundle(split_csv, out, "test-0.0.1", adapters=adapters, generator=gen or None)
    return out
