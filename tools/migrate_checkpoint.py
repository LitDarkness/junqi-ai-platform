from __future__ import annotations

import argparse
from pathlib import Path

import torch

from junqi_rl.model import JunqiActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a legacy dense-policy Junqi checkpoint.")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    old_state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model = JunqiActorCritic()
    new_state = model.state_dict()
    copied = []
    skipped = []
    for name, value in old_state.items():
        if name in new_state and new_state[name].shape == value.shape:
            new_state[name] = value
            copied.append(name)
        else:
            skipped.append(name)
    model.load_state_dict(new_state)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.target)
    print(f"copied={len(copied)} skipped={len(skipped)} target={args.target}")
    print("Policy head was reinitialized; run symmetric regression before promotion.")


if __name__ == "__main__":
    main()
