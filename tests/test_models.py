import numpy as np
import pandas as pd
import pytest

from menu.bundle import read_manifest, verify_bundle
from menu.metrics import GenerationScorer, decoding_score, rouge_l, summarise
from menu.models.classifier import DishClassifier
from menu.models.generator import RetrievalFallbackGenerator
from menu.rag.prompts import clean_title, title_messages
from menu.rag.retriever import BM25, RecipeRetriever
from menu.training import rag_examples
from tests.conftest import make_split


@pytest.fixture(scope="module")
def split():
    return make_split()


def test_rouge_l():
    assert rouge_l("Blackberry Poblano Margarita", "Blackberry-Poblano Margarita") == 1.0
    assert rouge_l("", "x") == 0.0
    assert 0 < rouge_l("Mint Lime Cooler", "Lime Cooler With Mint") < 1


def test_scorer_flags_copies_and_context(split):
    tr, te = split[split.split == "train"], split[split.split == "test"].head(3)
    sc = GenerationScorer().fit(tr)
    pred = sc.score(te, [tr.title.iloc[0], "", "Brand New Dish"], contexts=[["x"], [], ["Brand New Dish"]])
    assert pred.exact_copy.tolist() == [True, False, False]
    assert pred.valid.tolist() == [True, False, True]
    assert pred.context_copy.tolist() == [False, False, True]
    s = summarise(pred)
    assert set(s) >= {"rougeL", "novelty", "context_copy"} and isinstance(decoding_score(s), float)


def test_classifier_learns_and_calibrates(split, tmp_path):
    tr, va, te = (split[split.split == s] for s in ("train", "validation", "test"))
    clf = DishClassifier().fit(tr)
    t = clf.calibrate(va)
    assert 0.01 <= t <= 10
    assert np.mean(np.array(clf.predict(te.ing_raw)) == te.course.to_numpy()) > 0.8
    p = clf.predict_proba(["2 oz gin ; club soda"])
    assert p.shape == (1, 9) and abs(p.sum() - 1) < 1e-6
    pred = clf.predict_one("2 oz gin ; club soda ; ice cubes")
    assert pred.label == "Drink" and pred.top[0][1] >= pred.top[1][1]
    assert clf.oov_rate("zzzz qqqq") == 1.0 and clf.oov_rate("gin") == 0.0
    clf.save(tmp_path / "c.joblib")
    assert DishClassifier.load(tmp_path / "c.joblib").predict_one("gin").label == pred.label


def test_bm25_prefers_rare_matching_terms():
    bm = BM25().fit(["salt pepper chicken", "salt pepper", "saffron rice"])
    s = bm.scores(["saffron salt"])[0]
    assert s.argmax() == 2


def test_retriever_excludes_groups_and_ids(split):
    tr = split[split.split == "train"]
    r = RecipeRetriever().fit(tr)
    q = tr.iloc[0]
    assert q.recipe_id in [h.recipe_id for h in r.search(q.ing_clean, k=20)]
    hits = r.search_batch([q.ing_clean], k=5, exclude_groups=[q.group], exclude_ids=[q.recipe_id])[0]
    assert q.recipe_id not in [h.recipe_id for h in hits] and len(hits) == 5


def test_retriever_dense_hybrid_with_fake_encoder(split, tmp_path):
    tr = split[split.split == "train"]

    def enc(texts):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
        v = rng.normal(size=(len(texts), 8))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    r = RecipeRetriever().fit(tr, encoder=enc)
    assert r.dense is not None and r.dense.shape == (len(tr), 8)
    assert len(r.search("gin lime", k=3, mode="hybrid")) == 3
    r.save(tmp_path / "r.joblib")
    loaded = RecipeRetriever.load(tmp_path / "r.joblib")
    assert loaded.encoder is None and loaded.dense is not None
    assert len(loaded.search("gin lime", k=3, mode="hybrid")) == 3  # falls back to sparse without encoder


def test_rag_training_examples_never_contain_the_target(split):
    tr = split[split.split == "train"].reset_index(drop=True)
    ex = rag_examples(tr.head(30), RecipeRetriever().fit(tr), k=3)
    for (messages, target), row in zip(ex, tr.head(30).itertuples(), strict=True):
        assert target == row.title
        assert f"-> Title: {row.title}\n" not in messages[-1]["content"] + "\n"


def test_prompts_and_title_parsing():
    assert len(title_messages("rum; lime")) == 2
    assert clean_title('Title: "Mint Lime Cooler".\nmore') == ("Mint Lime Cooler", True)
    assert clean_title("") == ("", False)
    assert clean_title(" ".join(["w"] * 20))[1] is False


def test_fallback_generator_echoes_nearest_title(split):
    tr = split[split.split == "train"]
    hits = RecipeRetriever().fit(tr).search("gin lime soda", k=2)
    out = RetrievalFallbackGenerator().generate([title_messages("gin; lime", hits)])
    assert out == [hits[0].title]


def test_bundle_manifest_and_integrity(bundle_dir):
    m = read_manifest(bundle_dir)
    assert m["bundle_version"] == "test-0.0.1"
    assert m["data"]["leakage_check"]["group"] == 0
    assert "classifier.joblib" in m["files"] and verify_bundle(bundle_dir) == []
    (bundle_dir / "retriever.joblib").write_bytes((bundle_dir / "retriever.joblib").read_bytes())  # unchanged
    assert verify_bundle(bundle_dir) == []


def test_bundle_rejects_unknown_final_adapter(split_csv, tmp_path):
    from menu.bundle import build_bundle

    with pytest.raises(ValueError, match="final adapter"):
        build_bundle(split_csv, tmp_path, "x", generator={"final_variant": {"adapter": "missing"}})


def test_make_split_is_balanced():
    df = make_split()
    assert isinstance(df, pd.DataFrame) and df.course.value_counts().nunique() == 1
