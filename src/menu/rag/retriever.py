"""Recipe retriever for RAG: BM25 over ingredients/titles, optional dense embeddings, fused with RRF.

The knowledge base is the TRAIN split only (no validation/test recipes), and ``exclude_groups`` removes
near-duplicates of the query recipe when building RAG training prompts — otherwise the model would learn to copy.
At 14.5 k documents exact search (a sparse/dense matrix product, ~ms) is faster and simpler than an ANN index.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

Encoder = Callable[[Sequence[str]], np.ndarray]
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, ngram_range: tuple[int, int] = (1, 1)):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b[a-zéñ][a-zéñ']+\b")

    def fit(self, docs: Sequence[str]) -> BM25:
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        denom_doc = self.k1 * (1 - self.b + self.b * dl / max(dl.mean(), 1e-9))
        tf.data = tf.data * (self.k1 + 1) / (tf.data + np.repeat(denom_doc, np.diff(tf.indptr)))
        self.weights = (tf @ sp.diags(self.idf)).tocsc()
        return self

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        q = self.vec.transform(queries)
        q.data[:] = 1.0  # each query term counted once
        return np.asarray((q @ self.weights.T).todense())


@dataclass
class Hit:
    recipe_id: int
    title: str
    ingredients: str
    course: str | None
    score: float

    def as_dict(self) -> dict:
        return {
            "recipe_id": self.recipe_id,
            "title": self.title,
            "ingredients": self.ingredients,
            "course": self.course,
            "score": round(self.score, 4),
        }


class RecipeRetriever:
    def __init__(self, title_weight: float = 0.0, rrf_k: int = 60):
        self.title_weight = title_weight
        self.rrf_k = rrf_k
        self.encoder: Encoder | None = None
        self.dense: np.ndarray | None = None

    # ---------------- build ----------------
    def fit(self, kb: pd.DataFrame, encoder: Encoder | None = None) -> RecipeRetriever:
        kb = kb.reset_index(drop=True)
        self.recipe_id = kb.recipe_id.to_numpy()
        self.titles = kb.title.tolist()
        self.ingredients = kb.ing_clean.tolist()
        self.courses = [None if pd.isna(c) else str(c) for c in kb.get("course", [None] * len(kb))]
        self.groups = kb["group"].to_numpy() if "group" in kb else np.arange(len(kb))
        self.bm25_ing = BM25().fit(self.ingredients)
        self.bm25_title = BM25().fit(self.titles)
        if encoder is not None:
            self.attach_encoder(encoder, build=True)
        return self

    def attach_encoder(self, encoder: Encoder, build: bool = False) -> None:
        self.encoder = encoder
        if build or self.dense is None:
            self.dense = encoder([self.doc_text(i) for i in range(len(self.titles))]).astype(np.float32)

    def doc_text(self, i: int) -> str:
        return f"{self.titles[i]}. Ingredients: {self.ingredients[i]}"

    def __len__(self) -> int:
        return len(self.titles)

    # ---------------- search ----------------
    def _sparse_scores(self, queries: Sequence[str], title_weight: float) -> np.ndarray:
        s = self.bm25_ing.scores(queries)
        if title_weight:
            s = s + title_weight * self.bm25_title.scores(queries)
        return s

    def search_batch(
        self,
        queries: Sequence[str],
        k: int = 3,
        mode: str = "hybrid",
        title_weight: float | None = None,
        exclude_groups: Sequence[int | None] | None = None,
        exclude_ids: Sequence[int | None] | None = None,
    ) -> list[list[Hit]]:
        tw = self.title_weight if title_weight is None else title_weight
        sparse = self._sparse_scores(queries, tw)
        dense = None
        if mode in ("dense", "hybrid") and self.encoder is not None and self.dense is not None:
            dense = self.encoder(list(queries)) @ self.dense.T
        if mode == "dense" and dense is not None:
            final = dense
        elif dense is not None:  # reciprocal rank fusion of the two rankings
            final = self._rrf(sparse) + self._rrf(dense)
        else:
            final = sparse
        out: list[list[Hit]] = []
        for qi in range(len(queries)):
            row = final[qi].astype(np.float64).copy()
            if exclude_groups is not None and exclude_groups[qi] is not None:
                row[self.groups == exclude_groups[qi]] = -np.inf
            if exclude_ids is not None and exclude_ids[qi] is not None:
                row[self.recipe_id == exclude_ids[qi]] = -np.inf
            top = np.argpartition(-row, min(k, len(row) - 1))[: k * 4]
            top = top[np.argsort(-row[top], kind="stable")][:k]
            out.append(
                [
                    Hit(
                        int(self.recipe_id[i]),
                        self.titles[i],
                        self.ingredients[i],
                        self.courses[i],
                        float(row[i]),
                    )
                    for i in top
                    if np.isfinite(row[i]) and row[i] > 0
                ]
            )
        return out

    def search(self, query: str, k: int = 3, **kw) -> list[Hit]:
        return self.search_batch([query], k=k, **kw)[0]

    def _rrf(self, scores: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
        return 1.0 / (self.rrf_k + ranks + 1)

    # ---------------- persistence ----------------
    def save(self, path: str | Path) -> None:
        enc, self.encoder = self.encoder, None
        try:
            joblib.dump(self, path, compress=3)
        finally:
            self.encoder = enc

    @staticmethod
    def load(path: str | Path) -> RecipeRetriever:
        obj = joblib.load(path)
        obj.encoder = None
        return obj


def sentence_encoder(model_name: str = DENSE_MODEL, revision: str | None = None, device: str = "cpu") -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts), batch_size=128, normalize_embeddings=True))

    return encode
