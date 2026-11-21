"""Evaluation metrics (ported from the Assignment 4 notebook, ``p3.py``).

Classification: accuracy, macro-F1, weighted-F1.
Generation: ROUGE-L, anchored, unsupported, novelty, exact_copy, valid — plus ``context_copy`` for RAG
(the generated title is a verbatim copy of one of the retrieved neighbours' titles).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from menu.text import input_words, norm_w, toks

GEN_KEYS = ("rougeL", "anchored", "unsupported", "novelty", "exact_copy", "valid")


def cls_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def rouge_l(pred: str, ref: str) -> float:
    a, b = toks(pred), toks(ref)
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    if dp[-1] == 0:
        return 0.0
    p, r = dp[-1] / len(a), dp[-1] / len(b)
    return 2 * p * r / (p + r)


def norm_title(t: str) -> str:
    return " ".join(toks(t))


def decoding_score(s: dict[str, float]) -> float:
    """Model-selection criterion fixed in advance in A4 (used to pick decoding / final generator on validation)."""
    return 0.5 * s["rougeL"] + 0.3 * s["anchored"] + 0.2 * s["novelty"] - s["unsupported"] - (1 - s["valid"])


class GenerationScorer:
    """Reference statistics are fitted on the TRAIN split only."""

    def fit(self, train: pd.DataFrame) -> GenerationScorer:
        ing_df = Counter(norm_w(w) for s in train.ing_clean for w in set(s.replace(";", " ").split()))
        title_df = Counter(norm_w(w) for t in train.title for w in set(toks(t)))
        self.food_vocab = {w for w, c in ing_df.items() if c >= 20 and len(w) > 2 and title_df[w] / c < 2.0} - {
            "and",
            "with",
            "the",
        }
        self.train_titles = {norm_title(t) for t in train.title}
        self.nov_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train.title)
        self.nov_mat = self.nov_vec.transform(train.title)
        return self

    def score(
        self,
        frame: pd.DataFrame,
        titles: Sequence[str],
        valid: Sequence[bool] | None = None,
        contexts: Sequence[Sequence[str]] | None = None,
    ) -> pd.DataFrame:
        titles = [str(t).strip() for t in titles]
        sims = np.asarray((self.nov_vec.transform(titles) @ self.nov_mat.T).max(axis=1).toarray()).ravel()
        rows = []
        for i, row in enumerate(frame.itertuples()):
            t = titles[i]
            words = {norm_w(w) for w in toks(t)}
            inp = input_words(row.ing_clean)
            food = [w for w in words if w in self.food_vocab]
            rec = {
                "recipe_id": row.recipe_id,
                "ingredients": row.ing_clean,
                "title": t,
                "ref_title": row.title,
                "rougeL": rouge_l(t, row.title),
                "anchored": bool(words & inp),
                "unsupported": (sum(w not in inp for w in food) / len(food)) if food else 0.0,
                "novelty": float(1 - sims[i]) if t else 0.0,
                "exact_copy": norm_title(t) in self.train_titles,
                "valid": (bool(t) and len(t.split()) <= 15) if valid is None else bool(valid[i]),
            }
            if contexts is not None:
                rec["context_copy"] = norm_title(t) in {norm_title(c) for c in contexts[i]}
            rows.append(rec)
        return pd.DataFrame(rows)


def summarise(pred: pd.DataFrame) -> dict[str, float]:
    keys = [*GEN_KEYS, *(["context_copy"] if "context_copy" in pred else [])]
    return {k: float(pred[k].mean()) for k in keys}
