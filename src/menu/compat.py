"""Compatibility shims for pre-built GPU images (Kaggle / Colab)."""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger(__name__)

_PEFT_MODULES = (
    "peft.import_utils",
    "peft.tuners.lora.torchao",
    "peft.utils.quantization_utils",
    "peft.tuners.lora.model",
)


def patch_old_torchao(min_version: str = "0.16.0") -> bool:
    """Kaggle/Colab images ship torchao 0.10, which recent peft rejects with ImportError while wrapping *any* model.

    We never use torchao quantisation, so peft is told that torchao is unavailable. Returns True if patched.
    """
    try:
        import torchao  # noqa: F401
        from packaging.version import Version

        version = getattr(importlib.import_module("torchao"), "__version__", "0")
        if Version(version.split("+")[0]) >= Version(min_version):
            return False
    except ImportError:
        return False
    except Exception:  # unparsable version -> patch to be safe
        pass
    patched = False
    for name in _PEFT_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "is_torchao_available"):
            setattr(mod, "is_torchao_available", lambda: False)  # noqa: B010
            patched = True
    if patched:
        log.warning("old torchao detected: disabled peft's torchao integration (not used by this project)")
    return patched
