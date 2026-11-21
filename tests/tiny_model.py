"""Builds a tiny random Qwen2 model + LoRA adapters for offline tests and smoke runs (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

from tests.synthetic import SIGNATURES


def build_tiny(root: Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    root = Path(root)
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
    return root


if __name__ == "__main__":
    print(build_tiny(Path(sys.argv[1])))
