"""Pre-register one new online update-ratio hypothesis from complete158."""
import argparse
from pathlib import Path


def main():
    import json, torch
    from datetime import datetime,timezone
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from open_v4_queue_contract import inside,sha
    from open_v4_single_ratio_contract import verify_single_prerequisite
    parser=argparse.ArgumentParser();parser.add_argument('--previous-plan',required=True);parser.add_argument('--review',required=True);parser.add_argument('--audit',required=True);parser.add_argument('--cpu',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'):raise ValueError('another owner is active')
    old=read_json(inside(root,args.previous_plan));run=inside(root,old['directory']);progress=read_json(run/'PROGRESS.json')
    master=read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    if master.get('paused_by_user') or any(inside(root,p).exists() for p in old['stop_paths']):raise ValueError('STOP/pause wins')
    latest=progress['variants']['learner'];payload=torch.load(inside(root,latest['checkpoint']),map_location='cpu',weights_only=False)
    source=dict(path=latest['checkpoint'],sha256=latest['sha256'],model_hash=latest['model_hash'],iteration=payload['counters']['iteration'],leaf_value_scale=payload['leaf_value_scale'])
    if source['iteration']!=158:raise ValueError('completed latest158 required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');directory=f'runs/open_junqi_v4_single_ratio_{stamp}'
    base_path=f'config/experiments/open_v4_single_ratio_base_{stamp}.json';atomic_json(root/base_path,payload['config'])
    def entry(path):return dict(path=str(Path(path)),sha256=sha(inside(root,path)))
    names=set(old['source'])|{'tools/run_open_v4_single_ratio.py','tools/open_v4_single_ratio_contract.py','tools/prepare_open_v4_single_ratio.py','tools/open_v4_update_ratio_contract.py',
        'tools/audit_open_v4_single_ratio_boundary.py','tools/review_open_v4_single_final.py','tests/test_open_v4_single_ratio.py','start_open_v4_single_ratio.ps1','stop_open_v4_single_ratio.ps1'}
    plan=dict(old)
    plan.update(schema='junqi-single-ratio-plan-1',route='single_learning_update_ratio',registered_at=utc_now(),directory=directory,
        parent_config=base_path,parent_config_sha256=sha(root/base_path),source_checkpoint=source,source_counters=payload['counters'],source_config_hash=payload['config_hash'],
        participants=dict(old['participants'],parent=source),initial_replay=progress['collections']['learner'][-3:],
        frozen_runs=[dict(directory=str(p.parent.relative_to(root)),current_sha256=sha(p)) for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json'))],
        historical_stops=[entry(str(p.relative_to(root))) for p in sorted((root/'runs').glob('*/**/STOP'))],
        source={n:sha(root/n) for n in sorted(names)},stop_paths=[directory+'/STOP',directory+'/supervision/STOP'],
        previous_experiment=dict(plan=entry(args.previous_plan),review=entry(args.review)),boundary_audit=entry(args.audit),cpu_verification=entry(args.cpu),
        collection_start=1100,updates_per_stage=256,registered_first_segment_updates=6144,registered_maximum_updates=24576,
        source_role='Unpromoted latest158 full research state; previous134to158 direct-own result positive but protected external route unconfirmed.',
        authorization='User authorizes sustained one-learner online training, necessary evaluation and a different recorded optimization after unconfirmed24. Historical STOPs/runs remain protected.',
        frozen_reference_definition='arena control is frozen archived candidate134; parent is frozen segment source158;86/Champion26 also frozen. No concurrent online control.',
        interpretation='One evolving learner; only all-PPV update count64to256 per256fresh games changes. More optimizer/RNG steps, cumulative weight decay and next collectors are consequences. Single-branch historical comparisons cannot establish strict factor causality. No4x strength or efficiency claim.',
        hypothesis='Increase useful learning from fresh experience before three-shard eviction. Measured example coverage is limited; it is not proof of underfitting. Old/new holdout fitting is mixed. Test using unchanged protected external gates.',
        horizon='First24 stages158to182,6144fresh games/6144planned updates.12uncertainty continues24 unchanged.24unconfirmed stops review_due.Only verified full64 acceptance permits immediate next24 in same owner, at most96stages.',
        efficiency='One evolving learner preserved; fourfold planned learning attempts per fixed new games costs additional optimization time, measured separately from collection. No overall speedup or strength multiplier claim.')
    plan['historical_stop_markers']=[e['path'] for e in plan['historical_stops']]
    plan['longitudinal_evidence_history']=plan.pop('longitudinal_evidence')
    verify_single_prerequisite(root,plan)
    path=f'config/experiments/open_v4_single_ratio_{stamp}.json';atomic_json(root/path,plan)
    print(json.dumps(dict(plan=path,directory=directory,source=source),ensure_ascii=False))

if __name__=='__main__':main()
