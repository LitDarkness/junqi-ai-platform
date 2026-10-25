from __future__ import annotations

import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


_DIRECTORY = Path(__file__).parent
_BINARY = next((_DIRECTORY / f"fast_rule_ab{suffix}" for suffix in EXTENSION_SUFFIXES
                if (_DIRECTORY / f"fast_rule_ab{suffix}").is_file()), None)
if _BINARY is None:
    raise ImportError("fast_rule_ab extension is missing; rebuild it for this Python ABI")
_SPEC = importlib.util.spec_from_file_location("fast_rule_ab", _BINARY)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load {_BINARY}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def select_action(board, legal_actions, player_id, my_material, opp_material):
    return _MODULE.select_action(board, legal_actions, player_id)
