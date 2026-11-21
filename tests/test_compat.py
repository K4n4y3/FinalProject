import sys
import types

from menu.compat import patch_old_torchao


def test_patch_old_torchao(monkeypatch):
    fake = types.ModuleType("torchao")
    fake.__version__ = "0.10.0"
    monkeypatch.setitem(sys.modules, "torchao", fake)
    import peft.import_utils as iu

    monkeypatch.setattr(iu, "is_torchao_available", lambda: (_ for _ in ()).throw(ImportError("old")))
    assert patch_old_torchao() is True
    assert iu.is_torchao_available() is False


def test_no_patch_for_new_torchao(monkeypatch):
    fake = types.ModuleType("torchao")
    fake.__version__ = "0.17.0"
    monkeypatch.setitem(sys.modules, "torchao", fake)
    assert patch_old_torchao() is False
