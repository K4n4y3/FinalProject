"""LoRA fine-tuning of the generator (port of the A4 training loop) + RAG training-set construction.

Run on a GPU (Kaggle T4): see notebooks/final_training.ipynb. Loss is computed on answer tokens only (-100 on the
prompt); fp32 weights + fp16 autocast + GradScaler; AdamW, linear warm-up/decay, grad-clip 1.0.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from menu.models.generator import dtype_kwarg
from menu.rag.prompts import title_messages
from menu.rag.retriever import RecipeRetriever

Example = tuple[list[dict[str, str]], str]  # (chat messages, target title)


@dataclass
class LoraConfigSpec:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    base_revision: str | None = None
    learning_rate: float = 2e-4
    batch_size: int = 16
    grad_accum: int = 1
    epochs: int = 2
    max_len: int = 384  # RAG prompts are longer than the 192 tokens used in A4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def plain_examples(frame: pd.DataFrame) -> list[Example]:
    return [(title_messages(r.ing_clean), r.title) for r in frame.itertuples()]


def rag_examples(frame: pd.DataFrame, retriever: RecipeRetriever, k: int = 3, mode: str = "sparse") -> list[Example]:
    """Neighbours come from the train KB with the query's own group removed (no self/duplicate leakage)."""
    groups = frame["group"].tolist() if "group" in frame else [None] * len(frame)
    hits = retriever.search_batch(
        frame.ing_clean.tolist(), k=k, mode=mode, exclude_groups=groups, exclude_ids=frame.recipe_id.tolist()
    )
    return [(title_messages(r.ing_clean, h), r.title) for r, h in zip(frame.itertuples(), hits, strict=True)]


def train_lora(examples: Sequence[Example], cfg: LoraConfigSpec, out_dir: str | Path, log_every: int = 100) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    from menu.compat import patch_old_torchao

    patch_old_torchao()
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    tok = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.base_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, revision=cfg.base_revision, **dtype_kwarg(torch.float32)
    )
    total = sum(p.numel() for p in model.parameters())
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
        ),
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    items, truncated = [], 0
    for messages, answer in examples:
        p = tok(
            tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            add_special_tokens=False,
        )
        a = tok(answer + tok.eos_token, add_special_tokens=False).input_ids
        ids, labels = p.input_ids + a, [-100] * len(p.input_ids) + a
        if len(ids) > cfg.max_len:  # keep the answer: cut the prompt from the left of the context block
            truncated += 1
            ids, labels = ids[-cfg.max_len :], labels[-cfg.max_len :]
        items.append((ids, labels))

    def collate(batch):
        n = max(len(i) for i, _ in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for k, (i, lbl) in enumerate(batch):
            ids[k, : len(i)] = torch.tensor(i)
            lab[k, : len(lbl)] = torch.tensor(lbl)
            att[k, : len(i)] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    loader: Any = DataLoader(items, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, generator=gen)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps = max(1, math.ceil(len(loader) / cfg.grad_accum) * cfg.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, t0 = [], time.perf_counter()
    model.train()
    for ep in range(cfg.epochs):
        run = 0.0
        opt.zero_grad(set_to_none=True)
        for step, cpu_batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / cfg.grad_accum
            scaler.scale(loss).backward()
            run += loss.item() * cfg.grad_accum
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % log_every == 0:
                history.append({"epoch": ep, "step": ep * len(loader) + step, "loss": loss.item() * cfg.grad_accum})
        print(f"epoch {ep + 1}/{cfg.epochs}: mean train loss {run / max(1, len(loader)):.4f}")
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir)
    report = {
        "config": asdict(cfg),
        "examples": len(items),
        "truncated_examples": truncated,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share_pct": round(100 * trainable / total, 2),
        "train_minutes": round((time.perf_counter() - t0) / 60, 2),
        "history": history,
    }
    del model, opt, sched, loader, items
    import gc

    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    return report
