import importlib.util
import sys
from importlib import metadata

from crashvlm.models import compat


def test_hide_old_torchao(monkeypatch):
    monkeypatch.delitem(sys.modules, "torchao", raising=False)
    monkeypatch.setattr(compat.metadata, "version", lambda name: "0.10.0")
    assert compat.hide_old_torchao()
    assert importlib.util.find_spec("torchao") is None
    monkeypatch.delitem(sys.modules, "torchao")


def test_keeps_new_or_missing_torchao(monkeypatch):
    monkeypatch.delitem(sys.modules, "torchao", raising=False)
    monkeypatch.setattr(compat.metadata, "version", lambda name: "0.17.0")
    assert not compat.hide_old_torchao()

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(compat.metadata, "version", missing)
    assert not compat.hide_old_torchao()
    assert "torchao" not in sys.modules
