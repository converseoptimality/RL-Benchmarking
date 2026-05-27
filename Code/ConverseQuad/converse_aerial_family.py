"""Compatibility import for the Mode-II aerial benchmark family.

The canonical implementation lives in ``Code mode 2/Code/``.  This shim keeps
``import converse_aerial_family`` working from the ConverseQuad bundle root.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "Code mode 2" / "Code" / "converse_aerial_family.py"
_SPEC = importlib.util.spec_from_file_location("_converse_aerial_family_impl", _IMPL)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load {_IMPL}")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

for _name in getattr(_MOD, "__all__", []):
    globals()[_name] = getattr(_MOD, _name)
__all__ = list(getattr(_MOD, "__all__", []))
