import pandas as pd
import pytest

from menu.data import LABELS, check_no_leakage, clean_recipes, course_of, group_ids
from menu.text import clean_ingredient, input_words, norm_w, split_ingredients, to_clean, to_raw


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1 1/2 cups finely chopped red onion (about 1 large)", "red onion"),
        ("2 oz. silver tequila", "silver tequila"),
        ("Kosher salt and freshly ground black pepper", "salt black pepper"),
        ("3 tablespoons extra-virgin olive oil, divided", "olive oil"),
        ("", ""),
    ],
)
def test_clean_ingredient(line, expected):
    assert clean_ingredient(line) == expected


def test_split_ingredients_accepts_text_and_lists():
    assert split_ingredients("rum; lime\n mint ;; ") == ["rum", "lime", "mint"]
    assert split_ingredients(["  Rum ", "rum", "", "lime."]) == ["Rum", "lime"]
    assert len(split_ingredients(["x"] * 3 + [f"i{i}" for i in range(100)])) == 40


def test_representations():
    lines = ["2 oz white rum", "1 lime, cut into wedges"]
    assert to_raw(lines) == "2 oz white rum ; 1 lime, cut into wedges"
    assert to_clean(lines) == "white rum; lime"


def test_metric_helpers():
    assert norm_w("tomatoes") == "tomatoe" and norm_w("glass") == "glass"
    assert input_words("white rum; lime") == {"white", "rum", "lime"}


def test_course_rules_first_match_wins():
    assert course_of(["Cocktail", "Dessert"]) == ("Drink", 2)
    assert course_of(["Dinner"]) == ("Main", 1)
    assert course_of(["Vegan"]) == (None, 0)
    assert len(LABELS) == 9


def test_cleaning_and_grouping():
    raw = pd.DataFrame(
        {
            "title": ["A  Cake", "a cake", "Soup", None, "Soup"],
            "ingredients": [["1 cup sugar"], ["1 cup sugar"], ["2 cups stock"], ["x"], ["1 onion"]],
            "categories": [["Cake"], ["Cake"], ["Soup/Stew"], [], ["Drinks"]],
            "rating": [0.0, 4.0, 3.0, 1.0, 5.0],
            "calories": [10.0, 9e6, 100.0, 1.0, 50.0],
            "fat": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    df, log = clean_recipes(raw)
    assert len(df) == 3  # empty title dropped, duplicate cake dropped
    assert df.rating.isna().sum() == 1  # rating 0 -> NaN
    assert df.course.tolist() == ["Dessert", "Soup & Stew", "Drink"]  # TAG_FIX Drinks -> Drink
    df["group"] = group_ids(df)
    assert df.group.iloc[1] == df.group.iloc[2]  # same title "Soup" -> same group
    df["split"] = ["train", "train", "test"]
    assert check_no_leakage(df)["group"] == 1
    assert len(log.steps) == 4
