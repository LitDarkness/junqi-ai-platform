"""Register a bounded fresh-data behavior experiment from the latest control38."""
from pathlib import Path
import hashlib


def main():
    import torch
    from open_v4_runtime import atomic_json, read_json, utc_now
    from junqi_open_v4.config import OpenConfig
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15]
    old=read_json(root/'config/experiments/open_v4_tactical_training_20261004_065605.json')
    progress=read_json(root/old['directory']/'PROGRESS.json')
    status=read_json(root/old['directory']/'STATUS.json')
    if status['state']!='review_due':raise ValueError('previous experiment must have naturally ended')
    item=progress['variants']['control']
    source=dict(path=item['checkpoint'],sha256=item['sha256'],model_hash=item['model_hash'],iteration=item['counters']['iteration'])
    assert source['iteration']==38 and sha(root/source['path'])==source['sha256']
    payload=torch.load(root/source['path'],map_location='cpu',weights_only=False)
    raw=dict(payload['config'],training_greedy_after_ply=None)
    base=f'config/experiments/open_v4_long_behavior_base_{stamp}.json';atomic_json(root/base,raw)
    OpenConfig.load(root/base)
    directory=f'runs/open_junqi_v4_long_behavior_{stamp}'
    if (root/directory).exists():raise ValueError('never replace an existing independent experiment')
    names=list(old['source'])+['tools/run_open_v4_long_behavior.py','tools/prepare_open_v4_long_behavior.py',
        'start_open_v4_long_behavior.ps1','tests/test_open_v4_long_behavior.py']
    frozen=[dict(directory=str(p.parent.relative_to(root)),current_sha256=sha(p))
            for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json'))]
    markers=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in sorted((root/'runs').glob('*/**/STOP'))]
    control=dict(old['control'],training_greedy_after_ply=None)
    candidate=dict(control,training_greedy_after_ply=24)
    evidence=['reports/current_training_audit_20261004/RL_COMPONENT_DIAGNOSTIC.json',
              'reports/current_training_audit_20261004/BATCH_STABILITY_DIAGNOSTIC.json',
              'reports/current_training_audit_20261004/BATCH_STABILITY_AFTER.json',
              old['directory']+'/supervision/REVIEWS/RESULT_REVIEW_20261004_COMPONENT_BOUNDARY.json']
    for e in evidence:
        if not (root/e).exists():raise ValueError('complete diagnosis before registration')
    plan=dict(schema='junqi-long-behavior-plan-1',registered_at=utc_now(),directory=directory,
        parent_config=base,parent_config_sha256=sha(root/base),source_checkpoint=source,
        source_counters=payload['counters'],source_config_hash=payload['config_hash'],
        participants=dict(parent=source,champion=old['participants']['champion']),
        suite=old['suite'],teachers=old['teachers'],layout_manifest_hash=old['layout_manifest_hash'],
        initial_replay=progress['collections']['control'][-3:],frozen_runs=frozen,
        source={name:sha(root/name) for name in sorted(set(names))},historical_stops=markers,
        historical_stop_markers=[m['path'] for m in markers],stop_paths=[f'{directory}/STOP',f'{directory}/supervision/STOP'],
        evaluation_seed=old['evaluation_seed'],screen_families=16,full_families=64,evaluation_budget=256,
        evaluation_inference_batch_size=32,collection_start=200,stages_per_block=12,maximum_stages=24,
        games_per_stage=256,updates_per_stage=64,replay_stages=3,control=control,candidate=candidate,
        evidence=[dict(path=e,sha256=sha(root/e)) for e in evidence],
        authorization='User explicitly requested continual training, thorough multi-agent audit and a long-horizon route to much stronger AI. New independent experiment only; no archived run is resumed.',
        source_role='Latest unfiltered experimentalcontrol38 full state583dd. Not a production qualification or extension of the failed guard candidate; productionChampion26 remains frozen.',
        hypothesis='Late stochastic training behavior may produce avoidable blunders and less useful actual-terminal feedback even when search greedy recommendation has learned tactical safety. Test greedy behavior atply>=24 while keeping root noise and original visit targets. This is not a required RL bugfix or guaranteed improvement.',
        collector='Each branch256newgames before64updates using its own preceding full Learner, paired starts/modes/colors/teacher namespace200+. Predeclared24stages=6144newgames and1536attemptedupdates perbranch; aggregate12288/3072.',
        policy_contract='Original25/25/50 hybrid and teacher hard labels remain. Search targets are exact original visits, unchanged by selected behavior. True actual terminalWDL and independent draws preserved. No guard, dedup, history filtering or invented counterfactual targets.',
        behavior_contract='Both branches consume same-key stochastic RNG draw; only candidate overrides behavior after24ply with greedy visit action from the same noisy search. Control remains visits before24, squared visits afterwards. Noise retained every neural move.',
        evaluation_contract='All roles unguarded256search, fixed physicalBF16batch32, padding discarded, no noise. Recompute same-context source baseline; historical dynamic batch scores are not new baseline. Teachers082/084 are behavior-equivalent in previous430paired traces, not independent evidence. Final untouched.',
        screen_rule=old['screen_rule'],promotion_rule=old['promotion_rule'],
        horizon_contract='Observe at12 and24, no discretionary factor/horizon changes. Intermediate uncertainty is not permission to promote and is not a reason to reset the source. Candidate only qualifies through original full64 gate.',
        failure_policy='At12stop only integrity/STOP/nonfinite failure or severe regression: both relatedteacher paired upperCI<-.10 vs source38 AND direct candidate-source upperCI<.40. Otherwise finish only preregistered24. At24unconfirmed endsreview_due, preserve all evidence; no automatic extension or relaxed gate.',
        validation_policy=old['validation_policy'])
    path=root/f'config/experiments/open_v4_long_behavior_{stamp}.json';atomic_json(path,plan)
    atomic_json(root/'reports/current_training_audit_20261004/LONG_REGISTERED.json',dict(plan=str(path.relative_to(root)),directory=directory,source=source))
    print(str(path.relative_to(root)))


if __name__=='__main__':main()
