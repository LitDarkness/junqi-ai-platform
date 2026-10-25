"""Recompute failed policy-label result from raw games and register frozen audit."""
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack


def recompute_arena(arena):
    import numpy as np
    from collections import defaultdict
    from junqi_open_v4.arena import paired_interval
    scores = {}
    for matchup, row in enumerate(arena['summaries']):
        games = [g for g in arena['games'] if g['matchup'] == matchup]
        groups = defaultdict(list)
        for game in games:
            assert game['candidate'] == row['candidate'] and game['opponent'] == row['opponent']
            assert game['score'] == .5+.5*game['value']
            groups[game['family']].append(game['score'])
        means = {f:float(np.mean(values)) for f,values in groups.items()}
        assert paired_interval(list(means.values()),arena['seed']^matchup) == row['family_score']
        assert len(games) == row['games'] and sum(g['value'] > 0 for g in games) == row['wins'] and sum(g['value'] == 0 for g in games) == row['draws'] and sum(g['value'] < 0 for g in games) == row['losses']
        scores[matchup] = means
    for pair in arena['paired_deltas']:
        first,second = pair['from_matchup'],pair['to_matchup'];keys=sorted(scores[first].keys() & scores[second].keys())
        assert paired_interval([scores[second][k]-scores[first][k] for k in keys],arena['seed']^(first<<8)^second) == pair['delta']
    return True


def main():
    import torch
    from open_v4_runtime import ProcessLock,read_json,atomic_json,utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_open_v4.pipeline import promotion_decision,source_contract
    from junqi_open_v4.config import OpenConfig
    from run_open_v4_search_training import route_signal
    root=Path(__file__).resolve().parents[1];prior=root/'runs/open_junqi_v4_teacher_targets_20261004';proof=read_json(prior/'CONTRACT.json');result=read_json(prior/'RESULTS.json');progress=read_json(prior/'PROGRESS.json')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    directory='runs/open_junqi_v4_head_crossover_20261004';experiment=root/'config/experiments/open_v4_head_crossover_20261004.json'
    if experiment.exists() or (root/directory).exists():raise ValueError('preserve existing registration or experiment')
    stops=proof['stop_paths']+[str((prior/'STOP').relative_to(root))]
    if any((root/p).exists() for p in stops):raise RuntimeError('STOP requested; no clearing')
    with ExitStack() as locks:
        for row in proof['frozen_runs']:
            run=root/row['directory']
            for name in ('SUPERVISOR','TRAINING'):locks.enter_context(ProcessLock(run/f'supervision/{name}.lock'))
            assert read_json(run/'CURRENT.json') == row['current'];assert sha(run/'CONTRACT.json') == row['current']['contract_hash']
            for role in ('learner','actor','champion'):assert sha(run/row['current'][role]['path']) == row['current'][role]['sha256']
        assert source_contract(root) == proof['source']
        for path,digest in proof['evidence_files'].items():assert sha(root/path) == digest
        assert sha(root/'tools/run_open_v4_teacher_targets.py') == proof['tool_sha256'];assert sha(root/'config/experiments/open_v4_teacher_targets_20261004.json') == proof['plan_sha256']
        arena=read_json(prior/'arena.json');assert len(arena['games']) == 3240 and arena['role'] == 'dev';assert recompute_arena(arena)
        base=OpenConfig.load(root/proof['parent_config']);gate=promotion_decision(arena,base)
        gains={key:teacher_deltas(arena,before,after) for key,before,after in (('candidate_vs_parent','parent','candidate'),('candidate_vs_control','control','candidate'),('duplicate','parent','repeat'))}
        selected=route_signal([gains['candidate_vs_parent'],gains['candidate_vs_control']],gains['duplicate'],[arena['summaries'][i]['family_score'] for i in (0,1)],gate,64,.02)
        assert not selected and not result['teacher_target_route_confirmed'];assert gains == result['teacher_deltas'] and gate == result['existing_gate'] and arena['summaries'] == result['summaries']
        for row in progress['labels'].values():assert sha(prior/row['path']) == row['sha256'] and sha(prior/row['audit']) == row['audit_sha256']
        training_dir=root/'runs/open_junqi_v4_search_training_20261004';old=read_json(training_dir/'PROGRESS.json')
        def checkpoint(run,item):return {'path':str((run/item['checkpoint']).relative_to(root)),'sha256':item['sha256'],'model_hash':item['model_hash']}
        checkpoints={'parent':proof['source_checkpoint'],'trained':checkpoint(training_dir,old['variants']['control']),'mcts32':checkpoint(prior,progress['variants']['control']),'teacher32':checkpoint(prior,progress['variants']['candidate'])}
        complete={}
        for name,item in checkpoints.items():
            path=root/item['path'];assert sha(path) == item['sha256'];payload=torch.load(path,map_location='cpu',weights_only=False)
            assert all(k in payload for k in ('model','optimizer','scheduler','rng','counters','config','config_hash','model_hash','leaf_value_scale'))
            h=hashlib.sha256()
            for key,tensor in sorted(payload['model'].items()):h.update(key.encode());h.update(tensor.detach().contiguous().numpy().tobytes())
            assert h.hexdigest() == payload['model_hash'] == item['model_hash'];assert payload['counters']['effective_updates'] == (1664 if name=='parent' else 2048);assert payload['counters']['backtracking_rejections'] == 0
            assert payload['leaf_value_scale'] == .9076346023345448
            complete[name]={'checkpoint':item,'counters':payload['counters'],'full_state_verified':True}
        holdout=[]
        for item in proof['validation_replay']:
            row=item | {'metadata':str(Path(item['path']).with_suffix('.json'))}
            assert sha(root/row['path']) == row['sha256'] and sha(root/row['metadata']) == row['metadata_sha256'];assert read_json(root/row['metadata'])['layout_role'] == 'train';holdout.append(row)
        evidence=dict(proof['evidence_files'])
        for name in ('RESULTS.json','CONTRACT.json','PROGRESS.json','arena.json','LABEL_AUDIT.json'):
            path=prior/name;evidence[str(path.relative_to(root))]=sha(path)
        for name in ('run_open_v4_teacher_targets.py','prepare_open_v4_head_crossover.py'):
            path=root/'tools'/name;evidence[str(path.relative_to(root))]=sha(path)
        plan=dict(schema='junqi-head-crossover-plan-1',preregistered_at=utc_now(),directory=directory,parent_config=proof['parent_config'],parent_config_hash=base.digest(),frozen_runs=proof['frozen_runs'],source=proof['source'],evidence_files=evidence,teachers=proof['teachers'],layout_manifest_hash=proof['layout_manifest_hash'],suite=proof['suite'],checkpoints=checkpoints,internal_holdout_sources=holdout,leaf_value_scale=.9076346023345448,stop_paths=stops,evaluation_families=32,evaluation_budget=256,evaluation_seed=2026100405,duplicate_control_tolerance=.02,
            hypothesis='Shared-trunk PPV learning may change Policy and Value contributions differently. Freeze full qualified26 and64-data learned32 networks; primary cross uses learned Policy with parent Value, compared with full learned32 and parent26. Reverse cross is secondary. Check train-only WDL calibration first.',
            changed_factor='Compose full-network outputs, never transplant head weights across different encoders. Native rules/history, Value scale, simulated budget, source weights and teacher stay fixed;zero updates.',
            primary='hybrid learned32 Policy / parent26 Value must improve paired teacher CI over both full learned32 and qualified26, both direct duel lower>.5,duplicate point<=2pp. A positive32-family signal only qualifies a full64 recheck, never promotion.',
            secondary='parent Policy / learned Value is explanatory only; cannot rescue failed primary or promote either learned32 source.',
            compute_caveat='Fixed256 simulations, not equal-time. Cross outputs use two frozen networks and cost more inference; report search throughput,batch sizes,actual time and peak memory. Calibration is train-only, no external-strength claim.',
            preserved='Teacher-target route false;candidate18.95%[15.17,22.66] versuscontrol20.25%[16.08,24.55],paired-1.30pp[-5.66,2.93]. Prior stronger-search-data route false. No unqualified32 role adoption.',
            failure_policy='No automatic retries or overwrite. Keep source roles/checkpoints; inspect STOPs,all locks,error logs and complete result before any explicit new recovery run.',
            development_caveat='Repeated exploratory development corpus;final layouts and development failures never enter replay.',tool_sha256=sha(root/'tools/evaluate_open_v4_head_crossover.py'))
        atomic_json(experiment,plan)
        review=dict(at=utc_now(),heartbeat_received='2026-10-03T20:42:20.909Z',completed_at=read_json(prior/'STATUS.json')['updated_at'],route_confirmed=False,raw_games_bootstrap_recomputed=True,actual_games=3240,paired_teacher_deltas=gains,gate=gate,summaries=arena['summaries'],complete_checkpoints_verified=complete,new_experiment=str(experiment.relative_to(root)),decision='No external benefit from teacher-label factor;preserve all failed selections and qualified26. Run train-only Value calibration plus one frozen output-factorial search screen, not repeated training on the same labels.')
        atomic_json(prior/'REVIEW.json',review);atomic_json(root/'reports/search_budgets_20261004/TEACHER_TARGET_RESULT_REVIEW.json',review)
        print(json.dumps({'registered':str(experiment),'teacher_target_route_confirmed':False,'raw_games_recomputed':True,'checkpoints':checkpoints},ensure_ascii=False))


if __name__=='__main__':main()
