from __future__ import annotations

import copy
from typing import Any

import numpy as np


class CppAZStateAdapter:
    """Thin adapter over the canonical C++ open-information state."""

    def __init__(self, state: Any, provenance: dict[str, Any] | None = None):
        self.native_state = state
        self._provenance = dict(provenance or {})

    def clone(self) -> "CppAZStateAdapter":
        return CppAZStateAdapter(self.native_state.clone(), self._provenance)

    def current_player(self) -> int:
        return int(self.native_state.current_player())

    @staticmethod
    def opponent(player: int) -> int:
        return -int(player)

    def feature_planes(self) -> np.ndarray:
        features = np.asarray(self.native_state.feature_planes(), dtype=np.float32)
        if features.shape != (107, 12, 5):
            raise ValueError(f"unexpected native feature shape: {features.shape}")
        return features

    def legal_actions(self) -> np.ndarray:
        return np.asarray(self.native_state.legal_actions(), dtype=np.int32)

    def repetition_contexts(self) -> np.ndarray:
        return np.asarray(self.native_state.repetition_contexts(), dtype=np.uint8)

    def apply(self, action: int) -> None:
        self.native_state.apply(int(action))

    def is_terminal(self) -> bool:
        return bool(self.native_state.is_terminal())

    def terminal_value(self, player: int) -> float:
        return float(self.native_state.terminal_value(int(player)))

    def terminal_reason(self) -> str:
        return str(self.native_state.terminal_reason())

    def provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self._provenance)
