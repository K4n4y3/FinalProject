"""Regression tests on the real Epicurious split (skipped in CI when data/ is absent).

They guarantee that the refactor into a package did not change any reported number of Assignment 4.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from menu.data import check_no_leakage, load_split
from menu.metrics import GenerationScorer, cls_metrics, summarise
from menu.models.classifier import DishClassifier

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "epicurious_clean_split.csv"
A4 = ROOT / "reports" / "a4"
pytestmark = [pytest.mark.data, pytest.mark.skipif(not SPLIT.exists(), reason="real data not present")]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_split(SPLIT)


def test_split_sizes_and_no_leakage(df):
    assert df.split.value_counts().to_dict() == {"train": 14543, "validation": 1824, "test": 1802}
    assert all(v == 0 for v in check_no_leakage(df).values())


def test_a4_classifier_reproduces_reported_macro_f1(df):
    lab = df[df.course.notna()]
    tr, te = lab[lab.split == "train"], lab[lab.split == "test"]
    a4 = DishClassifier().fit(tr, augment=False)
    assert round(cls_metrics(te.course, a4.predict(te.ing_raw))["macro_f1"], 3) == 0.710


def test_final_classifier_is_better_on_both_input_formats(df):
    lab = df[df.course.notna()]
    tr, va = lab[lab.split == "train"], lab[lab.split == "validation"]
    a4, final = DishClassifier().fit(tr, augment=False), DishClassifier().fit(tr, augment=True)
    for col in ("ing_raw", "ing_clean"):
        assert (
            cls_metrics(va.course, final.predict(va[col]))["macro_f1"]
            >= cls_metrics(va.course, a4.predict(va[col]))["macro_f1"]
        )


@pytest.mark.skipif(not A4.exists(), reason="A4 prediction files not present")
@pytest.mark.parametrize(
    ("file", "rouge"),
    [
        ("gen_pred_qwen2_5_0_5b_lora_beam_4_no_repeat_3.csv", 0.386),
        ("gen_pred_flan_t5_base_full_fine_tuning.csv", 0.374),
        ("gen_pred_trigram_lm_baseline.csv", 0.146),
    ],
)
def test_generation_metrics_match_a4(df, file, rouge):
    pred = pd.read_csv(A4 / file).fillna({"title": ""})
    te = df[df.split == "test"].set_index("recipe_id").loc[pred.recipe_id].reset_index()
    s = summarise(GenerationScorer().fit(df[df.split == "train"]).score(te, pred.title, pred.valid))
    assert round(s["rougeL"], 3) == rouge
