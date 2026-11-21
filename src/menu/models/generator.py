"""Dish-name generator: Qwen2.5-0.5B-Instruct + LoRA adapters (plain and RAG-trained).

One base model is loaded once; adapters are switched per request (PEFT multi-adapter), and ``zero_shot`` runs the
base model with adapters disabled. torch/transformers are imported lazily so that the API, the classifier and the
tests work without the ``llm`` extra.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

Messages = list[dict[str, str]]

DECODING: dict[str, dict[str, Any]] = {
    "greedy": {"do_sample": False, "num_beams": 1, "no_repeat_ngram_size": 3},
    "beam4": {"do_sample": False, "num_beams": 4, "no_repeat_ngram_size": 3},
}


def dtype_kwarg(dtype) -> dict:
    """transformers>=4.56 renamed ``torch_dtype`` to ``dtype``; support both (Kaggle images lag behind)."""
    import transformers
    from packaging.version import Version

    return {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}


class TextGenerator(Protocol):
    adapters: list[str]

    def generate(self, batch: Sequence[Messages], adapter: str | None = None, **kwargs: Any) -> list[str]: ...


@dataclass
class GeneratorSpec:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    base_revision: str | None = None
    adapters: dict[str, str] = field(default_factory=dict)  # name -> local adapter dir
    device: str = "auto"
    dtype: str = "auto"
    num_threads: int | None = None


class QwenGenerator:
    def __init__(self, spec: GeneratorSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self._lock = threading.Lock()  # generate() is not re-entrant on a shared PEFT model
        if spec.num_threads:
            torch.set_num_threads(spec.num_threads)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if spec.device == "auto" else spec.device
        dtype = torch.float16 if (spec.dtype == "auto" and self.device == "cuda") else torch.float32
        if spec.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError(spec.dtype)
        if spec.dtype != "auto":
            dtype = getattr(torch, spec.dtype)
        self.tok = AutoTokenizer.from_pretrained(spec.base_model, revision=spec.base_revision)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(
            spec.base_model, revision=spec.base_revision, **dtype_kwarg(dtype)
        )
        self.adapters: list[str] = []
        if spec.adapters:
            from peft import PeftModel

            from menu.compat import patch_old_torchao

            patch_old_torchao()

            items = list(spec.adapters.items())
            name, path = items[0]
            model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
            self.adapters.append(name)
            for name, path in items[1:]:
                model.load_adapter(str(path), adapter_name=name)
                self.adapters.append(name)
        self.model = model.to(self.device).eval()

    def _ctx(self, adapter: str | None):
        if adapter is None:
            return self.model.disable_adapter() if self.adapters else contextlib.nullcontext()
        if adapter not in self.adapters:
            raise KeyError(f"unknown adapter {adapter!r}; loaded: {self.adapters}")
        self.model.set_adapter(adapter)
        return contextlib.nullcontext()

    def prompts(self, batch: Sequence[Messages]) -> list[str]:
        return [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]

    def generate(
        self,
        batch: Sequence[Messages],
        adapter: str | None = None,
        decoding: str = "beam4",
        max_new_tokens: int = 24,
        batch_size: int = 32,
        **_: Any,
    ) -> list[str]:
        import torch

        prompts = self.prompts(batch)
        outs: list[str] = []
        with self._lock, torch.inference_mode(), self._ctx(adapter):
            self.tok.padding_side = "left"
            for i in range(0, len(prompts), batch_size):
                enc = self.tok(
                    prompts[i : i + batch_size],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                    return_token_type_ids=False,
                ).to(self.device)
                ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tok.pad_token_id,
                    **DECODING[decoding],
                )
                outs += self.tok.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            self.tok.padding_side = "right"
        return outs


class RetrievalFallbackGenerator:
    """Degraded mode when the LLM cannot be loaded: echoes the nearest neighbour title from the prompt.

    Keeps the API contract alive (responses are flagged ``degraded=True``) instead of failing the whole service.
    """

    def __init__(self) -> None:
        self.adapters: list[str] = []

    def generate(self, batch: Sequence[Messages], adapter: str | None = None, **_: Any) -> list[str]:
        out = []
        for messages in batch:
            user = messages[-1]["content"]
            line = next((ln for ln in user.splitlines() if "-> Title:" in ln), "")
            out.append(line.split("-> Title:", 1)[1].strip() if line else "")
        return out


def adapter_dirs(root: str | Path, names: Sequence[str]) -> dict[str, str]:
    root = Path(root)
    return {n: str(root / n) for n in names if (root / n / "adapter_config.json").exists()}
