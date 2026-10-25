"""Experimental collection actors; production roles never change."""
from run_open_v4_search_training import paired_schedule


def collection_source(stage, variant, previous, qualified):
    if variant not in ("control", "candidate") or not 1 <= stage <= 6:
        raise ValueError("registered actor-refresh stage required")
    if variant == "control" or stage == 1:
        return qualified
    if previous["stage"] != stage - 1:
        raise ValueError("collector requires the immediately preceding full checkpoint")
    return {"path": previous["checkpoint"], "sha256": previous["sha256"],
            "model_hash": previous["model_hash"], "stage": previous["stage"]}


def verify_collection_pair(frozen, refreshed, expected_actor_hash):
    if frozen["layout_role"] != "train" or refreshed["layout_role"] != "train":
        raise ValueError("only training layouts may be collected")
    if frozen["game_count"] != 256 or refreshed["game_count"] != 256:
        raise ValueError("fixed 256-game collection required")
    if frozen["iteration"] != refreshed["iteration"] or paired_schedule(frozen) != paired_schedule(refreshed):
        raise ValueError("paired starts, colors, modes, or teachers changed")
    if refreshed["actor_model_hash"] != expected_actor_hash:
        raise ValueError("collector did not use its recorded learner")
    return True
