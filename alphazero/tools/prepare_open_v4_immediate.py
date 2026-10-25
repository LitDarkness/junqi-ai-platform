"""Preregister both conditional routes without changing the active arena."""
from pathlib import Path


def main():
    import json
    import torch
    from datetime import datetime,timezone
    from open_v4_runtime import read_json,atomic_json,utc_now
    from open_v4_queue_contract import sha
    from junqi_open_v4.queued_config import QueuedConfig
    root=Path(__file__).resolve().parents[1]
    main_plan=read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    completion_path=main_plan['optimization']['config'];completion=read_json(root/completion_path)
    if completion['schema']!='junqi-frozen-completion-plan-1':raise ValueError('unexpected active review')
    original=read_json(root/completion['original_plan']['path']);progress=read_json(root/original['directory']/'PROGRESS.json')
    source=completion['participants']['candidate'];payload=torch.load(root/source['path'],map_location='cpu',weights_only=False)
    if sha(root/source['path'])!=source['sha256'] or payload['counters']['iteration']!=62:raise ValueError('latest complete source changed')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');directory=f'runs/open_junqi_v4_immediate_queue_{stamp}'
    queue_path=f'config/experiments/open_v4_immediate_queue_{stamp}.json'
    def entry(path):return {'path':str(Path(path)),'sha256':sha(root/path)}
    raw=dict(payload['config'],root_noise_until_ply=None)
    base_path=f'config/experiments/open_v4_queued_base_{stamp}.json';atomic_json(root/base_path,raw);QueuedConfig.load(root/base_path)
    names=list(completion['source'])+['python/junqi_open_v4/queued_config.py','python/junqi_open_v4/queued_games.py',
        'tools/run_open_v4_queued_training.py','tools/open_v4_queue_contract.py','tools/audit_open_v4_queue_boundary.py',
        'tools/prepare_open_v4_immediate.py','tools/run_open_v4_immediate.py','start_open_v4_immediate.ps1','tests/test_open_v4_immediate.py']
    sources={name:sha(root/name) for name in sorted(set(names))}
    frozen=[{'directory':str(p.parent.relative_to(root)),'current_sha256':sha(p)} for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json'))]
    historical=[entry(str(p.relative_to(root))) for p in sorted((root/'runs').glob('*/**/STOP'))]
    initial=progress['collections']['candidate'][-3:]
    if len(initial)!=3:raise ValueError('three complete initial training shards required')
    for item in initial:
        if sha(root/item['path'])!=item['sha256'] or sha(root/item['metadata'])!=item['metadata_sha256']:raise ValueError('initial replay changed')
        if read_json(root/item['metadata'])['layout_role']!='train':raise ValueError('development data cannot enter replay')
    choices={};stop_paths=list(completion['stop_paths'])+[directory+'/STOP',directory+'/supervision/STOP']
    for route in ('qualified_continuation','late_noise_factor'):
        run=f'runs/open_junqi_v4_{route}_{stamp}'
        if (root/run).exists():raise ValueError('fresh independent run required')
        own_stops=[run+'/STOP',run+'/supervision/STOP'];stop_paths+=own_stops
        control=dict(original['candidate'],root_noise_until_ply=None)
        candidate=dict(control,root_noise_until_ply=24 if route=='late_noise_factor' else None)
        variants=['candidate'] if route=='qualified_continuation' else ['control','candidate']
        champion=source if route=='qualified_continuation' else completion['participants']['champion']
        plan={'schema':'junqi-queued-online-plan-1','registered_at':utc_now(),'directory':run,'route':route,
            'training_variants':variants,'parent_config':base_path,'parent_config_sha256':sha(root/base_path),
            'source_checkpoint':source,'source_counters':payload['counters'],'source_config_hash':payload['config_hash'],
            'participants':{'parent':source,'champion':champion,'control':completion['participants']['control']},
            'suite':original['suite'],'teachers':original['teachers'],'layout_manifest_hash':original['layout_manifest_hash'],
            'initial_replay':initial,'frozen_runs':frozen,'historical_stops':historical,'historical_stop_markers':[m['path'] for m in historical],
            'source':sources,'stop_paths':own_stops+completion['stop_paths']+[directory+'/STOP',directory+'/supervision/STOP'],
            'completion_plan':entry(completion_path),'queue_directory':directory,
            'prerequisite':{'required_route_confirmed':route=='qualified_continuation','native_all_games_replayed':4212,'boundary_audit_if_unconfirmed':True},
            'evaluation_seed':original['evaluation_seed'],'screen_families':16,'full_families':64,'evaluation_budget':256,
            'evaluation_inference_batch_size':32,'collection_start':500,'stages_per_block':12,'maximum_stages':24,
            'games_per_stage':256,'updates_per_stage':64,'replay_stages':3,'control':control,'candidate':candidate,
            'authorization':'User explicitly requires automatic fresh training immediately after full review, without waiting for another heartbeat or manual launch.',
            'hypothesis':'Continue a qualified latest62 route unchanged.' if route=='qualified_continuation' else 'Late noisy search may give less useful behavior/targets even with greedy late selection; test only root noise off atply>=24 versus noise every neural move. A hypothesis, not a proven repair.',
            'interpretation':'Single online continuation from qualified62; control label is frozen previouscontrol62, not a simultaneously trained control or independent causal evidence. Each stage uses its own preceding full Learner.' if route=='qualified_continuation' else 'Paired online experiment from unqualified researchcandidate62; both late greedy24, only candidate root-noise schedule differs. Upstream noise changes visits, selected actions and later terminal outcomes. Original25/25/50 teacher/search supervision and all true WDL preserved; no claim Policy visits remain unchanged.',
            'promotion_rule':'Full64 original gate: six teacher paired lower95>0 vsChampion/source/control; three direct lower95>.5; duplicate source drift<=.02; unchanged Champion gate. Qualified single route control is frozen reference; do not imply online causal control.082/084 correlated, final untouched.',
            'horizon':'24stages;12/24 evaluations; intermediate uncertainty continues to24 without promotion/factor change unless STOP/integrity/nonfinite or prespecified severe regression; final unconfirmed endsreview_due, no blind extension.',
            'registered_new_games':6144*len(variants),'registered_updates':1536*len(variants)}
        path=f'config/experiments/open_v4_{route}_{stamp}.json';atomic_json(root/path,plan);choices[route]=entry(path)
    queue={'schema':'junqi-immediate-queue-plan-1','registered_at':utc_now(),'directory':directory,
        'completion_plan':entry(completion_path),'waiting_for_run':completion['directory'],'source':sources,'choices':choices,
        'stop_paths':stop_paths,'poll_seconds':5,'authorization':'Explicit user instruction: review completion immediately triggers fresh training, no next scheduled turn required.',
        'contract':'CPU-only wait; complete raw full64 and native terminal replay first; no duplicate GPU process; previous locks must release; qualified single continuation or unconfirmed paired late-noise experiment after small saved training audit. Unexpected error/STOP never auto-restarts. One launch recorded before spawning.'}
    atomic_json(root/queue_path,queue)
    print(json.dumps({'queue_plan':queue_path,'directory':directory,'choices':choices},ensure_ascii=False))


if __name__=='__main__':main()
