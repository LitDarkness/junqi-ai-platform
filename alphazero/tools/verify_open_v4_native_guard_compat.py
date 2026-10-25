"""Fingerprint default PUCT against an archived native binary in separate processes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys


def main():
    p = argparse.ArgumentParser(); p.add_argument('--binary-directory', required=True)
    p.add_argument('--output', required=True); args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((root / args.binary_directory).resolve()))
    import JunqiAZ
    from junqi_fixed_rl.layout import FixedSameLayoutBackend, load_fixed_layout
    backend = FixedSameLayoutBackend(load_fixed_layout(root / 'config/layouts/fixed_layout_v1.json',
                                   'public_conservative_advance', project_root=root))
    records = []
    for noise in (False, True):
        for prefix in (0,16,32,48):
            state = backend.new_state(13); rng = random.Random(71)
            for _ in range(prefix):
                if state.is_terminal(): break
                state.apply(rng.choice(list(state.legal_actions())))
            if state.is_terminal(): continue
            tree = JunqiAZ.NeuralPUCTTree(state.native_state, 1.5)
            requests = []
            for simulation in range(64):
                leaf = tree.select_leaf()
                requests.append({k: leaf[k] for k in ('state_key','player','terminal','terminal_value','legal_actions','repetition_contexts')})
                legal = leaf['legal_actions']; weights = [(int(a)%17)+1 for a in legal]
                total = sum(weights)
                tree.backup([w/total for w in weights], leaf['terminal_value'] if leaf['terminal'] else .25)
                if simulation == 0 and noise: tree.add_root_dirichlet(.3,.25,19)
            records.append({'prefix':prefix,'noise':noise,'requests':requests,
                            'visits':tree.root_visits(),'edges':list(tree.root_edges())})
    text = json.dumps(records,sort_keys=True)
    Path(args.output).write_text(json.dumps({'cases':len(records),'fingerprint':hashlib.sha256(text.encode()).hexdigest()},indent=2),encoding='utf-8')


if __name__ == '__main__': main()
