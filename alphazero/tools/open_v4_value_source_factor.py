"""Select critic trajectories without changing Policy samples or terminal labels."""
import hashlib

ALLOWED = ("selfplay", "teacher_selfplay")


def audit_shard(samples, metadata):
    import numpy as np
    from collections import Counter
    if metadata.get("layout_role") != "train":
        raise ValueError("critic source audit requires training trajectories")
    modes, summary, offset = {}, {}, 0
    for game in metadata["games"]:
        mode = game["mode"]
        if mode not in (*ALLOWED, "teacher_vs_actor"):
            raise ValueError("unknown training behavior policy")
        family = int(game["family"])
        if family in modes and modes[family] != mode:
            raise ValueError("paired family behavior mismatch")
        modes[family] = mode
        rows = samples[offset:offset + len(game["trace"])]
        if len(rows) != len(game["trace"]): raise ValueError("missing complete trajectory")
        offset += len(rows)
        row = summary.setdefault(mode, dict(games=0, samples=0, wdl=Counter(), actor_turn_wdl=Counter(),
            teacher_turn_wdl=Counter(), actor_score_sum=0., terminal_reasons=Counter()))
        row["games"] += 1; row["samples"] += len(rows)
        row["actor_score_sum"] += (1 + game["result_red"] * game["actor_side"]) / 2
        row["terminal_reasons"][game["reason"]] += 1
        for sample in rows:
            if sample.game_id != family:
                raise ValueError("training family/trajectory misalignment")
            truth = np.eye(3, dtype=np.float32)[int(game["result_red"] * sample.player) + 1]
            if not np.array_equal(sample.wdl, truth):
                raise ValueError("real terminal WDL sign or class mismatch")
            cls = int(np.argmax(truth)); row["wdl"][cls] += 1
            if mode == "teacher_vs_actor":
                row["actor_turn_wdl" if sample.player == game["actor_side"] else "teacher_turn_wdl"][cls] += 1
    if offset != len(samples): raise ValueError("unassigned source rows")
    return modes, summary


def make_learner(config, root, family_modes, allowed_modes):
    import numpy as np
    from junqi_open_v4.learner import Learner
    from junqi_search_v3.recovery import _family_balanced_select
    modes = {int(k): v for k, v in family_modes.items()}
    if allowed_modes != "all" and list(allowed_modes) != list(ALLOWED):
        raise ValueError("only registered all/symmetric-behavior Value source factor")
    if any(v not in (*ALLOWED, "teacher_vs_actor") for v in modes.values()):
        raise ValueError("unknown source behavior")

    class ValueSourceLearner(Learner):
        def __init__(self, cfg, task_root):
            super().__init__(cfg, task_root)
            self.sampling_key_digest = hashlib.sha256()
            self.phase_batch_digests = {p: hashlib.sha256() for p in ("policy", "value")}
            self._source_pool = None
            self.value_pool_size = 0

        def _batch(self, samples):
            if self._source_pool is not samples:
                if any(int(s.game_id) not in modes for s in samples):
                    raise ValueError("unregistered training family in critic sampler")
                self._source_pool = samples
                self._value_pool = (samples if allowed_modes == "all" else
                    [s for s in samples if modes[int(s.game_id)] in ALLOWED])
                self.value_pool_size = len(self._value_pool)
                if not self._value_pool: raise ValueError("empty eligible Value replay")
            phase = "value" if self.counters["attempted_updates"] % 3 == 2 else "policy"
            pool = self._value_pool if phase == "value" else samples
            count = self.config.logical_batch_size
            keys = self.generator.integers(0, 2**63 - 1, size=count)
            self.sampling_key_digest.update(keys.tobytes())
            selected = _family_balanced_select(pool, count, keys, family_cap_fraction=.02)
            for s in selected:
                self.phase_batch_digests[phase].update(str((s.game_id, s.state_hash, s.player, s.ply)).encode())
            return selected

        def payload(self, *, role, lineage):
            result = super().payload(role=role, lineage=lineage)
            result["experiment_parameters"] = {"value_allowed_modes": allowed_modes}
            return result

        def resume(self, path):
            import torch
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("experiment_parameters") != {"value_allowed_modes": allowed_modes}:
                raise ValueError("critic source factor mismatch")
            return super().resume(path)

    return ValueSourceLearner(config, root)
