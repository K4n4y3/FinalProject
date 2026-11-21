"""Synthetic recipes with class-signature ingredients (no network, no real data)."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from menu.text import clean_ingredient

SIGNATURES = {
    "Drink": ["2 oz gin", "1 oz lime juice", "ice cubes", "club soda", "2 oz white rum", "fresh mint leaves"],
    "Dessert": [
        "1 cup sugar",
        "2 cups all-purpose flour",
        "1/2 cup unsalted butter",
        "1 tsp vanilla extract",
        "cocoa",
    ],
    "Soup & Stew": ["4 cups chicken stock", "1 onion, chopped", "2 carrots", "1 bay leaf", "celery stalks"],
    "Salad": ["romaine lettuce", "cucumber", "red wine vinegar", "cherry tomatoes", "feta cheese"],
    "Main": ["1 lb chicken thighs", "2 lb pork shoulder", "olive oil", "garlic cloves", "black pepper"],
    "Sauce & Condiment": [
        "mayonnaise",
        "dijon mustard",
        "1 tbsp honey",
        "apple cider vinegar",
        "chili flakes",
    ],
    "Breakfast & Bread": ["4 large eggs", "1 packet active dry yeast", "whole milk", "maple syrup", "bacon"],
    "Appetizer": ["crackers", "goat cheese", "smoked salmon", "capers", "baguette slices"],
    "Side": ["2 lb potatoes", "green beans", "brussels sprouts", "butter", "parsley"],
}
TITLE_WORDS = {
    "Drink": ["Cooler", "Fizz", "Mojito"],
    "Dessert": ["Cake", "Cookies", "Tart"],
    "Soup & Stew": ["Soup", "Stew"],
    "Salad": ["Salad"],
    "Main": ["Roast", "Braise"],
    "Sauce & Condiment": ["Sauce", "Dressing"],
    "Breakfast & Bread": ["Pancakes", "Bread"],
    "Appetizer": ["Crostini", "Bites"],
    "Side": ["Gratin", "Mash"],
}


def make_split(n_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for course, sig in SIGNATURES.items():
        for i in range(n_per_class):
            lines = rng.sample(sig, 3) + rng.sample(["salt", "water", "1 lemon", "2 tbsp olive oil"], 1)
            names = "; ".join(dict.fromkeys(x for x in map(clean_ingredient, lines) if x))
            main = clean_ingredient(lines[0]).title()
            rows.append(
                {
                    "title": f"{main} {rng.choice(TITLE_WORDS[course])} {i}",
                    "ing_raw": " ; ".join(lines),
                    "ing_clean": names,
                    "course": course,
                }
            )
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["recipe_id"] = np.arange(len(df))
    df["group"] = np.arange(len(df))
    df["title_key"] = df.title.str.lower()
    df["ingset_key"] = df.ing_clean + "#" + df.recipe_id.astype(str)
    df["split"] = np.where(df.index % 10 == 0, "test", np.where(df.index % 10 == 1, "validation", "train"))
    return df
