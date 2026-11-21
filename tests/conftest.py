"""Shared fixtures: a synthetic recipe split, a bundle built from it and (optionally) a tiny random Qwen2 model.

Nothing here touches the network, so the suite runs in CI and offline.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from menu.bundle import build_bundle
from menu.text import clean_ingredient

HAS_LLM = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))
REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "epicurious_clean_split.csv"
REAL_A4 = Path(__file__).resolve().parents[1] / "data" / "a4_reference"

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


@pytest.fixture(scope="session")
def split_csv(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "split.csv"
    make_split().to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def tiny_llm(tmp_path_factory) -> Path | None:
    """A 2-layer random Qwen2 + tokenizer with a chat template, plus two LoRA adapters ('lora', 'lora_rag')."""
    if not HAS_LLM:
        return None
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    root = tmp_path_factory.mktemp("tiny")
    base = root / "base"
    corpus = [" ".join(s) for s in SIGNATURES.values()] + ["system user assistant Ingredients Title Similar recipes"]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk.train_from_iterator(
        corpus * 20,
        trainers.BpeTrainer(
            vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    from tokenizers import decoders

    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tok.save_pretrained(base)
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(base)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for name in ("lora", "lora_rag"):
        m = get_peft_model(
            Qwen2ForCausalLM.from_pretrained(base), LoraConfig(r=4, lora_alpha=8, target_modules=targets)
        )
        m.save_pretrained(root / name)
    (root / "info.json").write_text(json.dumps({"base": str(base)}))
    return root


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory, split_csv, tiny_llm) -> Path:
    out = tmp_path_factory.mktemp("bundle")
    adapters = {n: tiny_llm / n for n in ("lora", "lora_rag")} if tiny_llm else {}
    gen = {"base_model": str(tiny_llm / "base")} if tiny_llm else {}
    if tiny_llm:
        gen["final_variant"] = {
            "name": "Qwen + LoRA-RAG",
            "adapter": "lora_rag",
            "rag_k": 3,
            "decoding": "beam4",
        }
    build_bundle(split_csv, out, "test-0.0.1", adapters=adapters, generator=gen or None)
    return out
