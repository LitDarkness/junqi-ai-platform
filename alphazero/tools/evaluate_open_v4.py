from __future__ import annotations

import argparse
import json
from pathlib import Path

def main():
    import torch
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.arena import create_dev_suite, run_arena, select_families
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.games import TeacherLeague
    from junqi_open_v4.layouts import create_manifest
    from junqi_open_v4.learner import model_hash
    from junqi_search_v3.dataset import atomic_json
    from junqi_search_v3.search import BatchedNeuralPUCT

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/open_junqi_v4_repaired.json")
    parser.add_argument("--part", choices=["search", "trained", "teachers"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--additional-checkpoint")
    parser.add_argument("--families", type=int, default=32)
    parser.add_argument("--opening-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = OpenConfig.load(root / args.config)
    torch.set_num_threads(config.workers)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    teachers = TeacherLeague(root, config.teacher_candidates, config.workers)
    layouts = root / config.layout_directory
    create_manifest(layouts, config.seed, train=64, dev=128, final=256)
    suite = create_dev_suite(root / config.run_directory / "development", layouts, teachers, config.seed ^ 0xDE)
    suite = select_families(suite, args.families, config.seed ^ 0xA2E)
    if args.opening_only:
        suite = [row for row in suite if row["start_ply"] == 0]
    models, checkpoint_provenance = {}, {}
    def load(name, path):
        source = (root / path).resolve()
        payload = torch.load(source, map_location="cpu", weights_only=False)
        model = PolicyValueNet(config.model).to(device).eval()
        model.load_state_dict(payload["model"])
        model.search_v3_leaf_value_scale = payload.get("leaf_value_scale", 1.)
        models[name] = model
        checkpoint_provenance[name] = {"path": str(source), "model_hash": model_hash(model),
                                       "leaf_value_scale": model.search_v3_leaf_value_scale}
        return model
    agents, budgets = {}, {}
    if args.part == "teachers":
        if len(teachers.modules) < 2:
            raise ValueError("teacher quality comparison needs two teachers")
        matchups = [("teacher1", "teacher0")]
    elif args.part == "search":
        model = load("frozen150", config.initial_checkpoint)
        for name, budget, leaves, reuse in (("serial64", 64, 1, False), ("serial128", 128, 1, False),
                ("bulk64", 64, 4, False), ("bulk256", 256, 4, False), ("reuse64", 64, 1, True)):
            agents[name] = BatchedNeuralPUCT(model, device=device, leaves_per_tree=leaves, reuse_roots=reuse)
            budgets[name] = budget
        matchups = [(name, "teacher0") for name in agents] + [
            ("bulk64", "serial64"), ("bulk256", "serial64"), ("reuse64", "serial64")]
    else:
        if not args.checkpoint:
            raise ValueError("trained evaluation requires an explicit frozen checkpoint")
        original = load("baseline150", config.initial_checkpoint)
        candidate = load("candidate", args.checkpoint)
        agents = {"baseline150": BatchedNeuralPUCT(original, device=device), "candidate": BatchedNeuralPUCT(candidate, device=device)}
        budgets = {name: config.simulations for name in agents}
        matchups = [("candidate", "baseline150"), ("baseline150", "teacher0"), ("candidate", "teacher0"),
                    ("baseline150", "teacher1"), ("candidate", "teacher1")]
        if args.additional_checkpoint:
            replicated = load("fixed_data_seed1", args.additional_checkpoint)
            agents["fixed_data_seed1"] = BatchedNeuralPUCT(replicated, device=device)
            budgets["fixed_data_seed1"] = config.simulations
            matchups.extend([("fixed_data_seed1", "baseline150"), ("fixed_data_seed1", "teacher0"), ("fixed_data_seed1", "teacher1")])
    result = run_arena(suite, matchups, agents, teachers, budgets, seed=config.seed ^ 0xEAA)
    result.update(checkpoints=checkpoint_provenance, teacher_provenance=teachers.provenance,
                  purpose="development diagnostics; this command does not promote models", config_hash=config.digest())
    atomic_json(root / args.output, result)
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
