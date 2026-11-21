"""Dish-type classifier: TF-IDF (1-2 grams) + LinearSVC with temperature-scaled probabilities.

Final model selection (see docs/MODEL_CARD.md):
* configuration = best GridSearch config of A4 (LinearSVC, C=0.3, ngram (1,2), min_df 3, balanced);
* trained on BOTH input formats (raw lines with quantities + cleaned names) so that free-form user input
  ("rum; lime; mint") is handled as well as editorial ingredient lines — chosen on validation;
* probabilities = softmax(decision / T); T fitted on validation by NLL (argmax is unchanged).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PARAMS = {"C": 0.3, "ngram_range": (1, 2), "min_df": 3, "class_weight": "balanced"}


@dataclass
class Prediction:
    label: str
    confidence: float
    top: list[tuple[str, float]]


class DishClassifier:
    def __init__(self, params: dict | None = None):
        self.params = {**PARAMS, **(params or {})}
        p = self.params
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=p["ngram_range"], min_df=p["min_df"], sublinear_tf=True),
                ),
                ("clf", LinearSVC(C=p["C"], class_weight=p["class_weight"], random_state=42)),
            ]
        )
        self.temperature = 1.0

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    @property
    def vocabulary(self) -> dict[str, int]:
        return self.pipeline.named_steps["tfidf"].vocabulary_

    def fit(self, train: pd.DataFrame, augment: bool = True) -> DishClassifier:
        X = pd.concat([train.ing_raw, train.ing_clean]) if augment else train.ing_raw
        y = pd.concat([train.course, train.course]) if augment else train.course
        self.pipeline.fit(X, y)
        return self

    def calibrate(self, val: pd.DataFrame) -> float:
        X = pd.concat([val.ing_raw, val.ing_clean])
        y = pd.concat([val.course, val.course]).to_numpy()
        S = self.pipeline.decision_function(X)
        idx = np.searchsorted(self.pipeline.classes_, y)

        def nll(t: float) -> float:
            return float(-log_softmax(S / t, axis=1)[np.arange(len(y)), idx].mean())

        self.temperature = float(minimize_scalar(nll, bounds=(0.01, 10), method="bounded").x)
        return self.temperature

    def predict(self, texts: Sequence[str]) -> list[str]:
        return list(self.pipeline.predict(list(texts)))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return softmax(self.pipeline.decision_function(list(texts)) / self.temperature, axis=1)

    def predict_one(self, text: str, top_k: int = 3) -> Prediction:
        p = self.predict_proba([text])[0]
        order = np.argsort(-p)[:top_k]
        top = [(self.classes[i], float(p[i])) for i in order]
        return Prediction(label=top[0][0], confidence=top[0][1], top=top)

    def oov_rate(self, text: str) -> float:
        """Share of input unigrams never seen in training — a cheap data-drift signal for monitoring."""
        analyzer = self.pipeline.named_steps["tfidf"].build_analyzer()
        words = [w for w in analyzer(text) if " " not in w]
        if not words:
            return 1.0
        vocab = self.vocabulary
        return sum(w not in vocab for w in words) / len(words)

    def save(self, path: str | Path) -> None:
        joblib.dump({"pipeline": self.pipeline, "temperature": self.temperature, "params": self.params}, path)

    @classmethod
    def load(cls, path: str | Path) -> DishClassifier:
        blob = joblib.load(path)
        obj = cls(blob["params"])
        obj.pipeline, obj.temperature = blob["pipeline"], blob["temperature"]
        return obj
