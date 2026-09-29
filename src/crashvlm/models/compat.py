"""Workarounds for preinstalled packages on hosted notebooks (Kaggle, Colab)."""

from __future__ import annotations

import sys
from importlib import metadata

from packaging.version import Version


def hide_old_torchao(minimum: str = "0.16.0") -> bool:
    """Make an installed but too old torchao invisible to peft and transformers.

    Kaggle ships torchao 0.10, and recent peft raises ImportError while attaching LoRA layers
    if any older torchao is importable. CrashVLM never uses torchao. A ``None`` entry in
    ``sys.modules`` makes ``importlib.util.find_spec`` report the package as absent. Call this
    before importing peft or transformers. Returns True if torchao was hidden.
    """
    try:
        version = Version(metadata.version("torchao"))
    except metadata.PackageNotFoundError:
        return False
    if version >= Version(minimum) or "torchao" in sys.modules:
        return False
    sys.modules["torchao"] = None
    return True
