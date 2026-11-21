"""Text normalisation shared by training, evaluation and the online service.

Everything here is a 1:1 port of the Assignment 4 notebook (``p2.py`` / ``p3.py``) so that the served models see
exactly the same preprocessing they were trained with (no training/serving skew).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

DROP_WORDS: frozenset[str] = frozenset(
    [
        "cup",
        "cups",
        "c",
        "tablespoon",
        "tablespoons",
        "tbsp",
        "teaspoon",
        "teaspoons",
        "tsp",
        "pound",
        "pounds",
        "lb",
        "lbs",
        "ounce",
        "ounces",
        "oz",
        "gram",
        "grams",
        "g",
        "kg",
        "ml",
        "liter",
        "liters",
        "quart",
        "quarts",
        "pint",
        "pints",
        "gallon",
        "can",
        "cans",
        "package",
        "packages",
        "pkg",
        "stick",
        "sticks",
        "clove",
        "cloves",
        "large",
        "medium",
        "small",
        "whole",
        "pinch",
        "dash",
        "bunch",
        "head",
        "heads",
        "slice",
        "slices",
        "piece",
        "pieces",
        "sprig",
        "sprigs",
        "stalk",
        "stalks",
        "rib",
        "ribs",
        "inch",
        "inches",
        "jar",
        "jars",
        "bottle",
        "bottles",
        "envelope",
        "container",
        "bag",
        "box",
        "handful",
        "scant",
        "heaping",
        "generous",
        "about",
        "plus",
        "more",
        "to",
        "taste",
        "fresh",
        "freshly",
        "chopped",
        "minced",
        "sliced",
        "diced",
        "grated",
        "finely",
        "coarsely",
        "thinly",
        "peeled",
        "cut",
        "into",
        "for",
        "serving",
        "garnish",
        "optional",
        "room",
        "temperature",
        "melted",
        "softened",
        "divided",
        "packed",
        "firmly",
        "lightly",
        "ground",
        "dried",
        "frozen",
        "thawed",
        "cooked",
        "uncooked",
        "of",
        "and",
        "or",
        "a",
        "an",
        "the",
        "such",
        "as",
        "thick",
        "thin",
        "ripe",
        "low",
        "sodium",
        "reduced",
        "fat",
        "unsalted",
        "kosher",
        "extra",
        "virgin",
        "boneless",
        "skinless",
        "trimmed",
        "rinsed",
        "drained",
        "halved",
        "quartered",
        "crushed",
        "toasted",
        "shredded",
        "beaten",
        "well",
        "very",
        "style",
        "lengthwise",
        "crosswise",
        "plus",
        "additional",
        "storebought",
        "homemade",
        "good",
        "quality",
        "seeded",
        "available",
        "chilled",
        "such",
        "preferably",
    ]
)

TOKEN = re.compile(r"[a-zéñ]+(?:'[a-z]+)?")
_SPLIT_INPUT = re.compile(r"\s*(?:;|\n|\r)\s*")

MAX_INGREDIENTS = 40
MAX_LINE_CHARS = 160


def clean_ingredient(line: str) -> str:
    """'1 1/2 cups finely chopped red onion (about 1 large)' -> 'red onion'."""
    s = str(line).lower()
    s = re.sub(r"\(.*?\)", " ", s).split(",")[0].split(";")[0]
    s = re.sub(r"[^a-zéñ' ]+", " ", s)
    toks_ = [t for t in s.split() if t not in DROP_WORDS and len(t) > 1]
    return " ".join(toks_[:4])


def toks(text: object) -> list[str]:
    return TOKEN.findall(str(text).lower())


def norm_w(w: str) -> str:
    """Crude singularisation used by every generation metric."""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def input_words(ing_clean: str) -> set[str]:
    return {norm_w(w) for part in str(ing_clean).split("; ") for w in part.split() if len(w) > 2}


def split_ingredients(text: str | Sequence[str]) -> list[str]:
    """Accept either a list or free text separated by ';' / new lines; drop blanks and exact duplicates."""
    parts: Iterable[str] = _SPLIT_INPUT.split(text) if isinstance(text, str) else text
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        p = re.sub(r"\s+", " ", str(part)).strip(" ,.-")
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p[:MAX_LINE_CHARS])
    return out[:MAX_INGREDIENTS]


def to_raw(lines: Sequence[str]) -> str:
    """Representation used by the classifier (``ing_raw`` in the notebook)."""
    return " ; ".join(lines)


def to_clean(lines: Sequence[str]) -> str:
    """Representation used by the generator and the retriever (``ing_clean``)."""
    return "; ".join(dict.fromkeys(x for x in map(clean_ingredient, lines) if x))
