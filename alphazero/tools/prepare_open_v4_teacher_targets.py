"""Preserve a failed search-data experiment and register policy-label audit."""
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack


def main():
    import torch
    from open_v4_runtime import ProcessLock,read_json,atomic_json,utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_open_v4.pipeline import promotion_decision,source_contract
    from junqi_open_v4.config import OpenConfig
    from run_open_v4_search_training import route_signal
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    prior=root/'runs/open_junqi_v4_search_training_20261004'
    proof=read_json(prior/'CONTRACT.json');result=read_json(prior/'RESULTS.json');old=read_json(prior/'PROGRESS.json')
    base=OpenConfig.load(root/proof['parent_config']);experiment=root/'config/experiments/open_v4_teacher_targets_20261004.json'
    new_run='runs/open_junqi_v4_teacher_targets_20261004'
    if experiment.exists() or (root/new_run).exists():raise ValueError('preserve existing registration/experiment')
    stops=proof['stop_paths']+[str((prior/'STOP').relative_to(root))]
    if any((root/p).exists() for p in stops):raise RuntimeError('STOP requested; no clearing')
    with ExitStack() as locks:
        for row in proof['frozen_runs']:
            run=root/row['directory']
            for name in ('SUPERVISOR','TRAINING'):locks.enter_context(ProcessLock(run/f'supervision/{name}.lock'))
            assert read_json(run/'CURRENT.json')==row['current']
            assert sha(run/'CONTRACT.json')==row['current']['contract_hash']
            for role in ('learner','actor','champion'):assert sha(run/row['current'][role]['path'])==row['current'][role]['sha256']
        assert source_contract(root)==proof['source']
        assert sha(root/'tools/run_open_v4_search_training.py')==proof['tool_sha256']
        assert sha(root/'config/experiments/open_v4_search_training_20261004.json')==proof['plan_sha256']
        for path,digest in proof['evidence_files'].items():assert sha(root/path)==digest
        arena=read_json(prior/'arena_primary.json');gate=promotion_decision(arena,base)
        paired={key:teacher_deltas(arena,before,after) for key,before,after in (('candidate_vs_parent','parent','candidate'),('candidate_vs_control','control','candidate'),('control_vs_parent','parent','control'),('duplicate','parent','repeat'))}
        selected=route_signal([paired['candidate_vs_parent'],paired['candidate_vs_control']],paired['duplicate'],[arena['summaries'][i]['family_score'] for i in (0,1)],gate,64,.02)
        assert not selected and not result['search_data_route_confirmed'] and paired==result['primary_teacher_deltas'] and gate==result['existing_gate']
        assert result['primary_summaries']==arena['summaries']
        assert result['secondary_summaries']==read_json(prior/'arena_secondary.json')['summaries']
        checkpoint_checks={}
        for name,item in old['variants'].items():
            assert item['stage']==6 and sha(prior/item['checkpoint'])==item['sha256']
            payload=torch.load(prior/item['checkpoint'],map_location='cpu',weights_only=False)
            assert all(k in payload for k in ('model','optimizer','scheduler','counters','rng','config','config_hash','leaf_value_scale','model_hash'))
            assert payload['counters']['effective_updates']==2048 and payload['counters']['backtracking_rejections']==0
            digest=hashlib.sha256()
            for key,tensor in sorted(payload['model'].items()):digest.update(key.encode());digest.update(tensor.detach().contiguous().numpy().tobytes())
            assert digest.hexdigest()==payload['model_hash']==item['model_hash']
            checkpoint_checks[name]={'checkpoint':item,'counters':payload['counters'],'full_state_verified':True}
        sources={}
        for item in proof['initial_replay']:
            if item['iteration'] not in (25,26):continue
            sources[f"initial_{item['iteration']:06d}"]={'path':item['path'],'sha256':item['sha256'],
                'metadata':str(Path(item['path']).with_suffix('.json')),'metadata_sha256':item['metadata_sha256']}
        for number in range(45,51):
            item=old['collections'][f'candidate:{number}']
            sources[f'new_{number:06d}']={**item,'path':str((prior/item['path']).relative_to(root)),'metadata':str((prior/item['metadata']).relative_to(root))}
        for item in sources.values():
            assert sha(root/item['path'])==item['sha256'] and sha(root/item['metadata'])==item['metadata_sha256']
            assert read_json(root/item['metadata'])['layout_role']=='train'
        evidence=dict(proof['evidence_files'])
        for name in ('RESULTS.json','CONTRACT.json','arena_primary.json','arena_secondary.json','PROGRESS.json'):
            path=prior/name;evidence[str(path.relative_to(root))]=sha(path)
        for name in ('run_open_v4_search_training.py','run_open_v4_fresh_data.py','fork_open_v4_qualified.py','evaluate_open_v4_search_budgets.py','open_v4_experiment_evidence.py'):
            path=root/'tools'/name;evidence[str(path.relative_to(root))]=sha(path)
        parent=torch.load(root/proof['source_checkpoint']['path'],map_location='cpu',weights_only=False)
        assert parent['config_hash']==base.digest() and parent['counters']['effective_updates']==1664
        plan={'schema':'junqi-teacher-target-plan-1','preregistered_at':utc_now(),'directory':new_run,
            'parent_config':proof['parent_config'],'parent_config_hash':base.digest(),'source_checkpoint':proof['source_checkpoint'],
            'source_counters':parent['counters'],'frozen_runs':proof['frozen_runs'],'source':proof['source'],'teachers':proof['teachers'],
            'layout_manifest_hash':proof['layout_manifest_hash'],'suite':proof['suite'],'validation_replay':proof['initial_replay'],
            'replay_sources':sources,'evidence_files':evidence,'stop_paths':stops,'stages':6,'updates_per_stage':64,
            'evaluation_budget':256,'evaluation_families':64,'evaluation_seed':2026100404,'duplicate_control_tolerance':.02,
            'hypothesis':'Weak selfplay search policy labels may reward network-specific play. Test frozen strong-teacher sparse hard labels at identical history-safe selfplay states, preserving all trajectories, WDL, batches and complete starting state.',
            'changed_factor':'Only sample.visits for noncycle selfplay states becomes a frozen teacher legal onehot. Existing teacher labels and history-sensitive selfplay targets stay identical; raw_visits remain original search counts.',
            'label_guard':'Rebuild exact native train layout/history, check full state hash/features/legal actions/repetition and real terminal WDL before deriving labels. Board-only teacher cannot supply a history-sensitive override.',
            'data_budget':'Replay exactly the same eight frozen training shards/windows as the prior256 candidate; both variants384 PPV updates from complete26. No fresh game count claim and no development/final data.',
            'selection':'Full64 matched256 development: candidate gains over both control and qualified parent, both teachers paired CI95 lower>0, both direct-duel CI95 lower>.5, duplicate<=2pp, unchanged full champion gate. No automatic promotion or threshold changes.',
            'negative_preserved':'Prior stronger-search data route false; candidate19.47% versus control28.45% on082,paired-8.98pp[-14.84,-3.32]. Parent gain not confirmed. No adoption of either unqualified32 checkpoint.',
            'failure_policy':'No automatic retries; inspect committed progress and logs before one explicit --resume. Uncommitted shards/checkpoints must not be overwritten.',
            'development_caveat':'Repeated development corpus; exploratory optimization, not independent final validation.',
            'prepare_tool_sha256':sha(Path(__file__))}
        atomic_json(experiment,plan)
        review={'at':utc_now(),'completed_at':read_json(prior/'STATUS.json')['updated_at'],'route_confirmed':False,
            'raw_arena_gate_recomputed':gate,'paired_teacher_deltas':paired,'summaries':arena['summaries'],
            'complete_checkpoints_verified':checkpoint_checks,'evidence_files':evidence,'planned_next_experiment':str(experiment.relative_to(root)),
            'decision':'Preserve failed256-search training and all source roles. Control-versus-parent teacher gain CI crosses0; no weight promotion. Run one exact-trace policy-label factor audit/experiment, not more updates on a failed branch.'}
        atomic_json(prior/'REVIEW.json',review)
        atomic_json(root/'reports/search_budgets_20261004/TRAINING_RESULT_REVIEW.json',review)
        print(json.dumps({'registered':str(experiment),'search_data_route_confirmed':False,'teacher_deltas':paired['candidate_vs_control']},ensure_ascii=False))


if __name__=='__main__':main()
