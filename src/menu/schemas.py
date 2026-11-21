"""API contract: model inputs and outputs (also rendered as OpenAPI at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from menu.text import MAX_INGREDIENTS, split_ingredients

GenMode = Literal["auto", "zero_shot", "lora", "lora_rag", "zero_shot_rag"]

EXAMPLE = ["2 oz white rum", "1 lime, cut into wedges", "10 fresh mint leaves", "2 tsp sugar", "club soda"]


class IngredientsIn(BaseModel):
    ingredients: list[str] | str = Field(
        ...,
        examples=[EXAMPLE],
        description="Ingredient lines (with or without quantities). A single string separated by ';' or new "
        f"lines is accepted too. 1-{MAX_INGREDIENTS} items, English.",
    )

    @field_validator("ingredients")
    @classmethod
    def _normalise(cls, v: list[str] | str) -> list[str]:
        items = split_ingredients(v)
        if not items:
            raise ValueError("at least one non-empty ingredient is required")
        if not any(any(ch.isalpha() for ch in x) for x in items):
            raise ValueError("ingredients must contain letters")
        return items


class ClassifyIn(IngredientsIn):
    top_k: int = Field(3, ge=1, le=9)


class GenerateIn(IngredientsIn):
    mode: GenMode = Field("auto", description="'auto' = the final model chosen on validation (see /v1/model)")
    rag_k: int | None = Field(None, ge=0, le=8, description="neighbours shown to the model (RAG modes)")


class ClassScore(BaseModel):
    label: str
    probability: float


class RecipeHit(BaseModel):
    recipe_id: int
    title: str
    ingredients: str
    course: str | None
    score: float


class Meta(BaseModel):
    model_version: str
    latency_ms: float
    warnings: list[str] = []


class ClassifyOut(BaseModel):
    dish_type: str
    confidence: float
    top: list[ClassScore]
    low_confidence: bool
    normalized_ingredients: list[str]
    meta: Meta


class GenerateOut(BaseModel):
    title: str
    valid: bool
    mode: str
    degraded: bool = Field(False, description="True if the LLM was unavailable and a retrieval fallback answered")
    context: list[RecipeHit] = []
    context_copy: bool = False
    meta: Meta


class MenuOut(BaseModel):
    classification: ClassifyOut
    generation: GenerateOut


class SearchIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, examples=["chicken, lemon and rice"])
    k: int = Field(5, ge=1, le=20)


class SearchOut(BaseModel):
    hits: list[RecipeHit]
    meta: Meta


class ChatIn(BaseModel):
    message: str = Field(
        ..., min_length=2, max_length=1000, examples=["I have chicken, lemon and rice. What can I cook?"]
    )
    k: int = Field(4, ge=1, le=8)


class ChatOut(BaseModel):
    answer: str
    sources: list[RecipeHit]
    suggested_dish_type: str | None
    grounded: bool = Field(..., description="the answer mentions at least one retrieved recipe")
    degraded: bool = False
    meta: Meta


class ModelInfo(BaseModel):
    bundle_version: str
    package_version: str
    components: dict[str, dict]
    final_generator: dict
    input_schema: dict
    output_schema: dict


class Health(BaseModel):
    status: Literal["ok", "degraded", "loading"]
    components: dict[str, bool]
    bundle_version: str | None
