"""Gradio user interface. It calls the same MenuService as the REST API (one code path, one set of metrics)."""

from __future__ import annotations

import json
from collections.abc import Callable

import gradio as gr
import pandas as pd

from menu.service import MODES, MenuService
from menu.text import split_ingredients

EXAMPLES = [
    "2 oz white rum\n1 lime, cut into wedges\n10 fresh mint leaves\n2 tsp sugar\nclub soda",
    "blackberries\npoblano peppers\nsilver tequila\ncointreau",
    "1 lb spaghetti\n4 oz guanciale\n3 large eggs\n1 cup grated pecorino romano\nblack pepper",
    "2 cups all-purpose flour\n1 cup sugar\n2 eggs\n1 cup buttermilk\n1 tsp baking soda\n2 ripe bananas",
    "1 lb flank steak\nkimchi\ncorn tortillas\nscallions\ngochujang\nlime",
    "baguette\ntomatoes\ngarlic\nbasil\nextra-virgin olive oil",
    "butternut squash\nonion\nvegetable stock\nheavy cream\nnutmeg",
    "romaine lettuce\nparmesan\ncroutons\nanchovies\nlemon juice\negg yolk",
]
MODE_LABELS = {
    "auto": "Финальная модель (auto)",
    "lora": "Qwen + LoRA",
    "lora_rag": "Qwen + LoRA-RAG",
    "zero_shot_rag": "Qwen zero-shot + RAG",
    "zero_shot": "Qwen zero-shot",
}
THEME = gr.themes.Soft(primary_hue="rose", secondary_hue="amber", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])
CSS = """
#title-out {font-size: 1.6rem; font-weight: 700; padding: .6rem 0;}
.small-note {opacity: .75; font-size: .85rem;}
"""


def _hits_df(hits) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"рецепт": h.title, "тип": h.course or "—", "ингредиенты": h.ingredients[:140], "score": h.score}
            for h in hits
        ]
    )


def build_ui(get_service: Callable[[], MenuService | None]) -> gr.Blocks:
    def service() -> MenuService:
        s = get_service()
        if s is None or s.classifier is None:
            raise gr.Error("Модели ещё загружаются, попробуйте через несколько секунд")
        return s

    def run_menu(text: str, mode: str):
        items = split_ingredients(text)
        if not items:
            raise gr.Error("Введите хотя бы один ингредиент (по одному на строку или через ';')")
        s = service()
        c = s.classify(items, top_k=5, _endpoint="ui")
        g = s.generate(items, mode=mode, _endpoint="ui")
        notes = [*c.meta.warnings, *g.meta.warnings]
        meta = (
            f"режим: **{g.mode}** · модель: `{c.meta.model_version}` · "
            f"классификация {c.meta.latency_ms:.0f} мс · генерация {g.meta.latency_ms:.0f} мс"
        )
        if g.context_copy:
            notes.append("название совпало с одним из найденных рецептов (копирование контекста)")
        return (
            {x.label: x.probability for x in c.top},
            f"<div id='title-out'>🍽️ {g.title or '—'}</div>",
            meta + ("\n\n⚠️ " + "\n\n⚠️ ".join(notes) if notes else ""),
            _hits_df(s.retriever.search("; ".join(items), k=3) if s.retriever else [])
            if not g.context
            else pd.DataFrame(
                [
                    {
                        "рецепт": h.title,
                        "тип": h.course or "—",
                        "ингредиенты": h.ingredients[:140],
                        "score": h.score,
                    }
                    for h in g.context
                ]
            ),
        )

    def run_compare(text: str):
        items = split_ingredients(text)
        if not items:
            raise gr.Error("Введите ингредиенты")
        s = service()
        rows = []
        for mode in MODES:
            g = s.generate(items, mode=mode, _endpoint="ui")
            rows.append(
                {
                    "модель": MODE_LABELS[mode],
                    "название": g.title,
                    "мс": round(g.meta.latency_ms),
                    "копия контекста": ("да" if g.context_copy else "нет") if g.context else "—",
                    "режим": g.mode,
                }
            )
        return pd.DataFrame(rows)

    def run_chat(message: str, history: list):
        s = service()
        out = s.chat(message)
        src = "\n".join(f"{i + 1}. **{h.title}** ({h.course or '—'})" for i, h in enumerate(out.sources))
        tail = f"\n\n*Источники:*\n{src}" if src else ""
        if out.suggested_dish_type:
            tail += f"\n\n*Похоже на тип блюда:* {out.suggested_dish_type}"
        return out.answer + tail

    def monitoring():
        s = get_service()
        if s is None:
            return {}, {}
        return s.monitor.summary(), s.health()

    def about() -> str:
        s = get_service()
        if s is None:
            return "Загрузка…"
        m = s.manifest
        cl = m.get("classifier", {})
        ev = m.get("generator", {}).get("evaluation", {})
        test = ev.get("test", []) if isinstance(ev, dict) else []
        gen_rows = "\n".join(
            f"| {r['model']} | {r['rougeL']:.3f} | {r['unsupported']:.3f} | {r['novelty']:.3f} | {r.get('context_copy', 0):.3f} |"
            for r in test
        )
        gen_table = (
            "| вариант | ROUGE-L | unsupported ↓ | novelty | копия контекста |\n|---|---|---|---|---|\n" + gen_rows
            if gen_rows
            else "_оценка генератора появится после прогона notebooks/final_training.ipynb_"
        )
        raw = cl.get("test_raw_lines", {})
        clean = cl.get("test_ingredient_names", {})
        return f"""
### Версия модели: `{s.version}`  (пакет {m.get("package_version")}, собран {m.get("created_at")})
**Данные:** `{m.get("data", {}).get("file")}` · SHA-256 `{m.get("data", {}).get("sha256", "")[:16]}…` · {m.get("data", {}).get("rows")}

**Классификатор:** {cl.get("type")} · test macro-F1 **{raw.get("macro_f1", 0):.3f}** (строки с количествами),
**{clean.get("macro_f1", 0):.3f}** (просто названия) · температура калибровки {cl.get("temperature", 0):.3f}

**Retriever:** {m.get("retriever", {}).get("type")} · база знаний: {m.get("retriever", {}).get("documents")} рецептов (только train)

**Генератор:** {m.get("generator", {}).get("base_model")} + адаптеры {m.get("generator", {}).get("adapters")} ·
финальный вариант: `{json.dumps(m.get("generator", {}).get("final_variant"), ensure_ascii=False)}`

{gen_table}

**Целостность бандла:** {"✅ все SHA-256 совпадают" if not s.integrity_errors else "⚠️ " + ", ".join(s.integrity_errors)}
"""

    with gr.Blocks(title="Меню от нейросетей") as demo:
        gr.Markdown(
            "# 🍽️ Меню от нейросетей\nПо списку ингредиентов система определяет **тип блюда**, придумывает "
            "**название** и подсказывает, **что приготовить**. Ингредиенты — на английском, по одному на строку."
        )
        with gr.Tab("Меню"):
            with gr.Row():
                with gr.Column(scale=5):
                    ing = gr.Textbox(label="Ингредиенты", lines=7, value=EXAMPLES[0])
                    mode = gr.Dropdown([(v, k) for k, v in MODE_LABELS.items()], value="auto", label="Генератор")
                    btn = gr.Button("Придумать блюдо", variant="primary")
                    gr.Examples(EXAMPLES, inputs=ing, label="Примеры")
                with gr.Column(scale=6):
                    title = gr.HTML()
                    label = gr.Label(label="Тип блюда (вероятности)", num_top_classes=5)
                    meta = gr.Markdown(elem_classes="small-note")
                    ctx = gr.Dataframe(label="Похожие рецепты из базы (контекст RAG)", wrap=True)
            btn.click(run_menu, [ing, mode], [label, title, meta, ctx], api_name="menu")
        with gr.Tab("Сравнение моделей"):
            ing2 = gr.Textbox(label="Ингредиенты", lines=6, value=EXAMPLES[1])
            btn2 = gr.Button("Сравнить все генераторы", variant="primary")
            table = gr.Dataframe(wrap=True)
            gr.Markdown(
                "Один и тот же вход проходит через zero-shot, RAG, LoRA и LoRA-RAG.",
                elem_classes="small-note",
            )
            btn2.click(run_compare, ing2, table, api_name="compare")
        with gr.Tab("Чат-ассистент"):
            gr.ChatInterface(
                run_chat,
                examples=[
                    "I have chicken, lemon and rice. What can I cook?",
                    "Something sweet with apples and cinnamon",
                    "A cocktail with gin and grapefruit",
                ],
                description="Ответы опираются только на найденные рецепты Epicurious (RAG); источники указаны под ответом.",
            )
        with gr.Tab("Мониторинг"):
            refresh = gr.Button("Обновить")
            with gr.Row():
                summary = gr.JSON(label="Скользящее окно запросов")
                health = gr.JSON(label="Состояние компонентов")
            gr.Markdown(
                "Полные метрики Prometheus: [`/metrics`](/metrics) · JSON: [`/v1/monitoring`](/v1/monitoring)",
                elem_classes="small-note",
            )
            refresh.click(monitoring, None, [summary, health])
            gr.Timer(10).tick(monitoring, None, [summary, health])
        with gr.Tab("О модели"):
            info = gr.Markdown()
            gr.Markdown("REST API и схемы входа/выхода: [`/docs`](/docs)", elem_classes="small-note")
        demo.load(about, None, info)
        demo.load(monitoring, None, [summary, health])
    return demo
