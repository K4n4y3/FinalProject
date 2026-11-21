"""Epicurious data pipeline: raw JSON -> cleaned, labelled, group-split table (Assignment 2, ported as code).

The frozen output of this pipeline (``epicurious_clean_split.csv``) is what every model in the system is trained
and evaluated on. ``build_split`` reproduces it from ``full_format_recipes.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from menu.text import clean_ingredient

RAW_SHA256 = "268811bcb6bc65621b29b8f0745e1ff51c04a222eb3c72bd055b030c1741d3bc"
SEED = 42

TAG_FIX = {
    "Drinks": "Drink",
    "Cookies": "Cookie",
    "Condiment": "Condiment/Spread",
    "turkey": "Turkey",
    "snack": "Snack",
}

# Order matters: the first matching rule wins (most specific first).
COURSE_RULES: list[tuple[str, frozenset[str]]] = [
    (
        "Drink",
        frozenset({"Drink", "Hot Drink", "Cocktail", "Alcoholic", "Non-Alcoholic", "Punch", "Smoothie"}),
    ),
    ("Dessert", frozenset({"Dessert", "Frozen Dessert", "Cake", "Cookie", "Candy", "Pie"})),
    ("Sauce & Condiment", frozenset({"Sauce", "Condiment/Spread", "Salad Dressing", "Dip"})),
    ("Soup & Stew", frozenset({"Soup/Stew", "Stew"})),
    ("Salad", frozenset({"Salad"})),
    ("Breakfast & Bread", frozenset({"Breakfast", "Bread"})),
    ("Appetizer", frozenset({"Appetizer", "Hors D'Oeuvre"})),
    ("Side", frozenset({"Side"})),
    ("Main", frozenset({"Dinner", "Sandwich"})),
]
LABELS: list[str] = sorted(name for name, _ in COURSE_RULES)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def course_of(tags: list[str]) -> tuple[str | None, int]:
    hits = [name for name, rule in COURSE_RULES if rule & set(tags)]
    return (hits[0] if hits else None), len(hits)


@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, name: str, before: int, after: int, why: str) -> None:
        self.steps.append({"step": name, "rows_before": before, "rows_after": after, "why": why})


def clean_recipes(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """The nine documented cleaning steps of Assignment 2."""
    log = CleaningLog()
    df = raw.copy()
    n0 = len(df)
    df = df[df.title.notna() & df.ingredients.notna()].copy()
    log.add("drop empty records", n0, len(df), "no title and no ingredients")
    n0 = len(df)
    df = df[df.ingredients.map(len) > 0].copy()
    log.add("drop recipes without ingredients", n0, len(df), "no model input")
    df["title"] = df.title.str.strip().str.replace(r"\s+", " ", regex=True)
    df["ing_raw"] = df.ingredients.map(" ; ".join)
    df["ing_clean"] = df.ingredients.map(
        lambda lines: "; ".join(dict.fromkeys(x for x in map(clean_ingredient, lines) if x))
    )
    n0 = len(df)
    df = df[df.ing_clean.str.len() > 0].copy()
    log.add("ingredient lines -> names", n0, len(df), "generator needs products, not quantities")
    df["title_key"] = df.title.str.lower()
    df["ingset_key"] = df.ing_clean.map(lambda s: " | ".join(sorted(set(s.split("; ")))))
    n0 = len(df)
    df = df.drop_duplicates(["title_key", "ingset_key"], keep="first").copy()
    log.add("drop duplicates (title + ingredient set)", n0, len(df), "re-publications leak between splits")
    if "rating" in df:
        df["rating"] = df.rating.where(df.rating > 0)
    if {"calories", "fat"} <= set(df.columns):
        bad = (df.calories > 5000) | (df.calories == 0) | ((9 * df.fat > df.calories * 1.05) & (df.calories > 0))
        for c in ["calories", "protein", "fat", "sodium"]:
            if c in df:
                df[c] = df[c].where(~bad)
    df["categories"] = df.categories.map(lambda tags: sorted({TAG_FIX.get(t, t) for t in tags}))
    tmp = df.categories.map(course_of)
    df["course"] = tmp.str[0]
    df["n_course_rules"] = tmp.str[1]
    df = df.reset_index(drop=True)
    df["recipe_id"] = np.arange(len(df))
    return df, log


def group_ids(df: pd.DataFrame, keys: tuple[str, ...] = ("title_key", "ingset_key")) -> np.ndarray:
    """Union-find: recipes sharing a title OR an ingredient set end up in the same group."""
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        first: dict[str, int] = {}
        for i, k in enumerate(df[key]):
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    return np.array([find(i) for i in range(len(df))])


def assign_split(df: pd.DataFrame, n_folds: int = 10, test_fold: int = 0, val_fold: int = 1) -> pd.Series:
    from sklearn.model_selection import StratifiedGroupKFold

    strata = df.course.fillna("Unlabelled")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, strata, df.group)):
        fold[idx] = k
    return pd.Series(
        np.where(fold == test_fold, "test", np.where(fold == val_fold, "validation", "train")), index=df.index
    )


def build_split(raw_json: str | Path, verify_sha: bool = True) -> tuple[pd.DataFrame, CleaningLog]:
    if verify_sha and sha256(raw_json) != RAW_SHA256:
        raise ValueError(f"{raw_json}: SHA-256 differs from the frozen Epicurious release")
    raw = pd.DataFrame(json.loads(Path(raw_json).read_text(encoding="utf-8")))
    df, log = clean_recipes(raw)
    df["group"] = group_ids(df)
    df["split"] = assign_split(df)
    return df, log


def load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"title", "ing_raw", "ing_clean", "course", "split", "recipe_id"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for c in ("ing_raw", "ing_clean", "title"):
        df[c] = df[c].fillna("").astype(str)
    return df


def check_no_leakage(df: pd.DataFrame) -> dict[str, int]:
    """Number of groups / titles / ingredient sets that appear in more than one split (must be all zeros)."""
    keys = [k for k in ("group", "title_key", "ingset_key") if k in df]
    return {k: int(df.groupby(k).split.nunique().gt(1).sum()) for k in keys}
