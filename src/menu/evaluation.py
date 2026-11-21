"""Offline evaluation of generator variants on identical prompts + selection of the final variant on validation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from menu.metrics import GenerationScorer, decoding_score, summarise
from menu.models.generator import TextGenerator
from menu.rag.prompts import clean_title, title_messages
from menu.rag.retriever import RecipeRetriever


@dataclass(frozen=True)
class Variant:
    name: str
    adapter: str | None  # None = zero-shot base model
    rag_k: int  # 0 = no retrieval
    decoding: str = "beam4"


VARIANTS = [
    Variant("Qwen zero-shot", None, 0),
    Variant("Qwen zero-shot + RAG", None, 3),
    Variant("Qwen + LoRA (A4)", "lora", 0),
    Variant("Qwen + LoRA (A4) + RAG", "lora", 3),
    Variant("Qwen + LoRA-RAG", "lora_rag", 3),
]


def run_variant(
    gen: TextGenerator,
    retriever: RecipeRetriever,
    scorer: GenerationScorer,
    frame: pd.DataFrame,
    v: Variant,
    batch_size: int = 32,
) -> tuple[dict, pd.DataFrame]:
    frame = frame.reset_index(drop=True)
    hits = (
        retriever.search_batch(frame.ing_clean.tolist(), k=v.rag_k, mode="sparse")
        if v.rag_k
        else [[] for _ in range(len(frame))]
    )
    batch = [title_messages(ing, h) for ing, h in zip(frame.ing_clean, hits, strict=True)]
    t0 = time.perf_counter()
    raw = gen.generate(batch, adapter=v.adapter, decoding=v.decoding, batch_size=batch_size)
    secs = time.perf_counter() - t0
    parsed = [clean_title(x) for x in raw]
    pred = scorer.score(
        frame, [p[0] for p in parsed], [p[1] for p in parsed], contexts=[[x.title for x in h] for h in hits]
    )
    s = summarise(pred)
    row = {
        "model": v.name,
        "adapter": v.adapter,
        "rag_k": v.rag_k,
        "decoding": v.decoding,
        **s,
        "score": decoding_score(s),
        "sec_per_item": secs / max(1, len(frame)),
    }
    return row, pred


def select_final(val_table: pd.DataFrame) -> pd.Series:
    """Highest pre-registered validation score; ties broken by fewer context copies."""
    t = val_table.copy()
    t["context_copy"] = t.get("context_copy", 0.0)
    return t.sort_values(["score", "context_copy"], ascending=[False, True]).iloc[0]
