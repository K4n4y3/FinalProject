"""Chat prompts. One module, used by training AND serving, so the formats can never drift apart."""

from __future__ import annotations

import re
from collections.abc import Sequence

from menu.rag.retriever import Hit

SYSTEM_PROMPT = "You name recipes. Given a list of ingredients, answer with one short recipe title only."
SYSTEM_PROMPT_RAG = (
    "You name recipes. Given a list of ingredients, answer with one short recipe title only. "
    "Similar published recipes are shown for style; invent a new title for the given ingredients, do not copy them."
)
CHAT_SYSTEM = (
    "You are a cooking assistant. Choose the ONE recipe from the numbered list that best answers the question. "
    "Reply in exactly this format: <number>: <one short sentence why it fits>."
)

MAX_CONTEXT_INGREDIENTS = 12


def _short(ing: str, n: int = MAX_CONTEXT_INGREDIENTS) -> str:
    parts = ing.split("; ")
    return "; ".join(parts[:n]) + ("; ..." if len(parts) > n else "")


def title_messages(ing_clean: str, context: Sequence[Hit] | None = None) -> list[dict[str, str]]:
    if not context:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Ingredients: {ing_clean}"},
        ]
    examples = "\n".join(f"- Ingredients: {_short(h.ingredients)} -> Title: {h.title}" for h in context)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": f"Similar recipes:\n{examples}\n\nIngredients: {ing_clean}"},
    ]


def chat_messages(question: str, context: Sequence[Hit]) -> list[dict[str, str]]:
    ctx = "\n".join(
        f"[{i + 1}] {h.title} ({h.course or 'unlabelled'}): {_short(h.ingredients)}" for i, h in enumerate(context)
    )
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": f"Recipes:\n{ctx}\n\nQuestion: {question}"},
    ]


def parse_choice(text: str, n: int) -> tuple[int | None, str]:
    """'2: Uses the lemon and the rice.' -> (1, 'Uses the lemon and the rice.'). Index is 0-based; None if unusable."""
    m = re.match(r"\s*\[?(\d+)\]?\s*[:.)\-]?\s*(.*)", str(text).strip(), flags=re.S)
    if not m or not 1 <= int(m.group(1)) <= n:
        return None, ""
    reason = m.group(2).strip().split("\n")[0]
    reason = re.split(r"(?<=[.!?])\s", reason)[0].strip()  # one sentence
    if reason and not reason.endswith((".", "!", "?")):
        reason += "."
    return int(m.group(1)) - 1, reason[:240]


def clean_title(text: str) -> tuple[str, bool]:
    """Same post-processing as A4: first line, strip 'Title:' prefixes and quotes; valid = 1..15 words."""
    t = str(text).strip().split("\n")[0]
    t = re.sub(r"^\s*(title|recipe title|name)\s*:\s*", "", t, flags=re.I).strip(" \"*'.")
    return t, bool(t) and len(t.split()) <= 15
