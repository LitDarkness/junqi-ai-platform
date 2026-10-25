import importlib
import pathlib
import sys

import numpy as np

build = pathlib.Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(build))
az = importlib.import_module("JunqiAZ")

a = az.JunqiAZState()
b = az.JunqiAZState()
a.reset_random(12345)
b.reset_random(12345)
assert a.board() == b.board(), "seeded reset is not deterministic"
assert a.key() == b.key()
assert a.feature_planes().shape == (107, 12, 5)
assert az.FEATURE_CHANNELS == 107

before = (a.key(), a.turn_count(), a.history_size(), a.no_capture_ply())
action = a.legal_actions()[0]
a.apply_with_undo(action)
assert a.undo()
after = (a.key(), a.turn_count(), a.history_size(), a.no_capture_ply())
assert before == after

clone = a.clone()
for _ in range(100):
    if a.is_terminal():
        break
    legal = a.legal_actions()
    contexts = a.repetition_contexts()
    assert len(legal) == len(contexts)
    action = legal[len(legal) // 2]
    a.apply(action)
    clone.apply(action)
    assert a.key() == clone.key()

boards = np.asarray([b.board(), b.board()], dtype=np.int32)
values = az.heuristic_084_raw_batch(boards, 1)
assert values.shape == (2,) and np.isfinite(values).all()
print("JunqiAZ fixed-layout smoke: PASS")
