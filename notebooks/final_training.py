# %% [markdown]
# # Меню от нейросетей — финальная модель (Kaggle T4)
#
# **Biloshchytskyi Yevhenii · AAI-2602 · AI Essentials: Theory and Applications — Final Project**
#
# Этот ноутбук — единственное место, где нужен GPU. Он использует **тот же пакет `menu`**, что и онлайн-сервис
# (одинаковые предобработка, промпты и метрики — без расхождений между обучением и inference), и делает:
#
# 1. воспроизводит очищенный датасет Epicurious и разбиение из сырого JSON (проверка: 14 543 / 1 824 / 1 802);
# 2. строит RAG-индекс (BM25 + all-MiniLM-L6-v2) **только по train**;
# 3. дообучает два LoRA-адаптера Qwen2.5-0.5B-Instruct: `lora` (как в A4) и `lora_rag` (с найденными рецептами в промпте);
# 4. сравнивает 5 вариантов генератора на validation, **выбирает финальный по заранее заданному критерию** и
#    оценивает все варианты на test;
# 5. собирает версионированный бандл модели (manifest + SHA-256) и публикует его в Hugging Face Hub.
#
# **Перед запуском:** *Add Input* → `hugodarwood/epirecipes`; *Accelerator* → GPU T4; *Internet* → On;
# *Add-ons → Secrets* → `HF_TOKEN` (токен Hugging Face с правом write, вводите сами). Запуск: *Save Version → Save & Run All*
# (~1.5–2 ч). Без `HF_TOKEN` бандл просто сохраняется в Output.

# %%
import os
import subprocess
import sys

REPO = os.environ.get("MENU_REPO", "git+https://github.com/K4n4y3/FinalProject@main")
if os.environ.get("MENU_SKIP_INSTALL") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"menu-from-nn[llm] @ {REPO}"], check=True)
    # always take the latest commit of the package code (pip skips a reinstall when the version string is unchanged)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", f"menu-from-nn @ {REPO}"],
        check=True,
    )

# %%
import json
import time
from pathlib import Path

import pandas as pd

import menu
from menu.bundle import build_bundle, verify_bundle
from menu.data import build_split, check_no_leakage, sha256
from menu.evaluation import VARIANTS, run_variant, select_final
from menu.metrics import GenerationScorer
from menu.models.generator import GeneratorSpec, QwenGenerator
from menu.rag.retriever import RecipeRetriever, sentence_encoder
from menu.training import LoraConfigSpec, plain_examples, rag_examples, set_seed, train_lora

CONFIG = {
    "bundle_version": "1.0.0",
    "hf_repo": "K4n4y3/menu-from-nn",  # model repo for the bundle (created if missing)
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "rag_k": 3,
    "val_n": 300,  # validation prompts for model selection
    "test_n": None,  # None = whole test split
    "lora": {"epochs": 2, "learning_rate": 2e-4, "batch_size": 16, "max_len": 192},  # = A4 configuration
    "lora_rag": {"epochs": 2, "learning_rate": 2e-4, "batch_size": 16, "max_len": 384},
    "dense": True,
    "train_n": None,  # smoke tests only
}
if os.environ.get("FINAL_OVERRIDES"):  # e.g. tiny model + few rows for a CPU smoke test
    for k, v in json.loads(os.environ["FINAL_OVERRIDES"]).items():
        CONFIG[k] = {**CONFIG[k], **v} if isinstance(v, dict) and isinstance(CONFIG.get(k), dict) else v
OUT = Path(os.environ.get("MENU_OUT", "/kaggle/working" if Path("/kaggle").exists() else "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
set_seed(42)
print("menu", menu.__version__, "| out:", OUT, "| config:", json.dumps(CONFIG))

# %% [markdown]
# ## 1. Данные: воспроизводим очистку и разбиение Assignment 2 из сырого файла


# %%
def locate(name: str) -> Path:
    for root in [Path(os.environ.get("MENU_DATA_DIR", ".")), Path("/kaggle/input")]:
        hits = sorted(root.rglob(name)) if root.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found: Add Input -> hugodarwood/epirecipes")


raw_path = locate("full_format_recipes.json")
df, clean_log = build_split(raw_path)
sizes = df.split.value_counts().to_dict()
print("split:", sizes, "| leakage:", check_no_leakage(df))
assert sizes == {"train": 14543, "validation": 1824, "test": 1802} or CONFIG["train_n"], "split differs from A4"
split_csv = OUT / "epicurious_clean_split.csv"
df.drop(columns=["ingredients", "directions", "categories"], errors="ignore").to_csv(split_csv, index=False)
df = pd.read_csv(split_csv).fillna({"ing_clean": "", "ing_raw": "", "title": ""})
train = df[df.split == "train"].reset_index(drop=True)
if CONFIG["train_n"]:
    train = train.sample(CONFIG["train_n"], random_state=42).reset_index(drop=True)
val = df[df.split == "validation"].sample(min(CONFIG["val_n"], (df.split == "validation").sum()), random_state=42)
test = df[df.split == "test"]
test = test if not CONFIG["test_n"] else test.sample(CONFIG["test_n"], random_state=43)
val, test = val.reset_index(drop=True), test.reset_index(drop=True)
print(
    f"train {len(train)} | validation prompts {len(val)} | test prompts {len(test)} | csv sha256 {sha256(split_csv)[:12]}"
)

# %% [markdown]
# ## 2. RAG-индекс по train (без validation/test) и качество поиска

# %%
encoder = None
if CONFIG["dense"]:
    try:
        encoder = sentence_encoder(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    except Exception as exc:  # offline smoke test
        print("dense encoder unavailable:", exc)
retriever = RecipeRetriever().fit(df[df.split == "train"], encoder=encoder)
scorer = GenerationScorer().fit(df[df.split == "train"])
print("KB documents:", len(retriever), "| dense:", retriever.dense is not None)

# %% [markdown]
# ## 3. Дообучение: `lora` (повтор A4) и `lora_rag` (контекст из 3 похожих рецептов)
#
# Для обучающих примеров `lora_rag` соседи ищутся с **исключением группы самого рецепта** (тот же title или набор
# ингредиентов), иначе модель научилась бы просто копировать ответ из контекста.

# %%
reports = {}
for name in ("lora", "lora_rag"):
    cfg = LoraConfigSpec(base_model=CONFIG["base_model"], **CONFIG[name])
    examples = plain_examples(train) if name == "lora" else rag_examples(train, retriever, k=CONFIG["rag_k"])
    print(
        f"\n== {name}: {len(examples)} examples; prompt example:\n{examples[0][0][-1]['content'][:400]}\n-> {examples[0][1]}"
    )
    reports[name] = train_lora(examples, cfg, OUT / "adapters" / name)
    print({k: v for k, v in reports[name].items() if k != "history"})
pd.DataFrame(
    [
        {
            "adapter": k,
            **{
                kk: v[kk]
                for kk in ("examples", "truncated_examples", "trainable_params", "trainable_share_pct", "train_minutes")
            },
        }
        for k, v in reports.items()
    ]
)

# %% [markdown]
# ## 4. Выбор финального генератора на validation, затем оценка на test
#
# Критерий зафиксирован заранее (A4): `score = 0.5·ROUGE-L + 0.3·anchored + 0.2·novelty − unsupported − (1 − valid)`.
# Дополнительно считается `context_copy` — доля названий, дословно скопированных из найденных рецептов.

# %%
gen = QwenGenerator(
    GeneratorSpec(
        base_model=CONFIG["base_model"], adapters={n: str(OUT / "adapters" / n) for n in ("lora", "lora_rag")}
    )
)
try:
    from huggingface_hub import model_info

    base_revision = model_info(CONFIG["base_model"]).sha
except Exception:
    base_revision = None
print("base model revision pinned:", base_revision)

val_rows = []
for v in VARIANTS:
    row, _ = run_variant(gen, retriever, scorer, val, v)
    val_rows.append(row)
    print(f"{v.name:28s} score {row['score']:.3f}  ROUGE-L {row['rougeL']:.3f}  copy {row.get('context_copy', 0):.3f}")
val_table = pd.DataFrame(val_rows)
final = select_final(val_table)
print("\nFINAL (chosen on validation):", final["model"])
val_table.round(4)

# %%
test_rows, examples = [], {}
for v in VARIANTS:
    row, pred = run_variant(gen, retriever, scorer, test, v)
    test_rows.append(row)
    examples[v.name] = pred.title.tolist()
    pred.to_csv(OUT / f"test_pred_{v.adapter or 'base'}_k{v.rag_k}.csv", index=False)
test_table = pd.DataFrame(test_rows)
test_table.round(4)

# %%
ex = test[["ing_clean", "title"]].head(10).rename(columns={"ing_clean": "ingredients", "title": "reference"})
for name, titles in examples.items():
    ex[name] = titles[:10]
ex

# %%
from menu.rag.prompts import title_messages

probe = "white rum; lime; mint leaves; sugar; club soda"
for v in VARIANTS:
    hits = retriever.search(probe, k=v.rag_k, mode="sparse") if v.rag_k else []
    t0 = time.perf_counter()
    for _ in range(3):
        out = gen.generate([title_messages(probe, hits)], adapter=v.adapter, decoding=v.decoding)
    print(f"{v.name:28s} {(time.perf_counter() - t0) / 3 * 1000:6.0f} ms/request ({gen.device}, batch 1) -> {out[0]!r}")

# %% [markdown]
# ## 5. Бандл модели: версия, SHA-256, метрики → Hugging Face Hub

# %%
fv = {"name": final["model"], "adapter": final["adapter"], "rag_k": int(final["rag_k"]), "decoding": final["decoding"]}
evaluation = {
    "validation": json.loads(val_table.to_json(orient="records")),
    "test": json.loads(test_table.to_json(orient="records")),
    "training": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in reports.items()},
    "selection_rule": "max validation score (A4 criterion); ties -> fewer context copies",
}
bundle = OUT / "bundle"
manifest = build_bundle(
    split_csv,
    bundle,
    CONFIG["bundle_version"],
    adapters={n: OUT / "adapters" / n for n in ("lora", "lora_rag")},
    generator={"base_model": CONFIG["base_model"], "base_revision": base_revision, "final_variant": fv},
    generator_eval=evaluation,
    encoder=encoder,
)
assert verify_bundle(bundle) == []
(OUT / "final_eval.json").write_text(json.dumps({"final_variant": fv, "evaluation": evaluation}, indent=2))
print(json.dumps({k: manifest[k] for k in ("bundle_version", "classifier", "retriever")}, indent=1, default=str)[:2500])

# %%
token = os.environ.get("HF_TOKEN")
if not token and Path("/kaggle").exists():
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
if token and CONFIG["hf_repo"]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(CONFIG["hf_repo"], repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(bundle), repo_id=CONFIG["hf_repo"], commit_message=f"bundle {CONFIG['bundle_version']}"
    )
    tag = f"v{CONFIG['bundle_version']}"
    try:
        api.create_tag(CONFIG["hf_repo"], tag=tag, revision=commit.oid)
    except Exception as exc:
        print("tag exists?", exc)
    print(f"published https://huggingface.co/{CONFIG['hf_repo']}  revision {commit.oid}  tag {tag}")
else:
    print("HF_TOKEN not set: download Output -> bundle/ and upload it with `hf upload` (see docs/DEPLOYMENT.md)")
