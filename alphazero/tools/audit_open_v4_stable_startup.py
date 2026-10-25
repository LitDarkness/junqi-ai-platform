"""Read committed first stage on CPU; verify real games, LR, source and protections."""
import argparse
from pathlib import Path


def main():
    import numpy as np,torch
    from open_v4_runtime import read_json,atomic_json,utc_now
    from open_v4_queue_contract import sha,inside
    from open_v4_stable_policy_contract import verify_stable_seed
    from run_open_v4_search_training import verify_learning_state
    from junqi_open_v4.stable_policy_config import StablePolicyConfig
    from junqi_open_v4.training_replay import complete_games
    from junqi_open_v4.layouts import MultiLayoutBackend,load_role
    from junqi_search_v3.loop import read_selfplay_shard
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];plan=read_json(inside(root,args.plan));run=inside(root,plan['directory'])
    progress=read_json(run/'PROGRESS.json');records=progress['collections']['learner']
    if not any(r['stage']==1 for r in records) or not (run/'learner/reports/stage_000001.json').exists():raise ValueError('first complete commit not yet available')
    checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:raise ValueError('startup check failed: '+name)
    cfg=StablePolicyConfig.load(run/'learner/config.json');torch.set_num_threads(2)
    source=torch.load(inside(root,plan['source_checkpoint']['path']),map_location='cpu',weights_only=False)
    seed=torch.load(run/'learner/checkpoints/learner_stage_000000.pt',map_location='cpu',weights_only=False)
    checks.update({'seed:'+k:v for k,v in verify_stable_seed(source,seed).items()})
    after=torch.load(run/'learner/checkpoints/learner_stage_000001.pt',map_location='cpu',weights_only=False)
    check('initial_model_exact',seed['model_hash']==plan['source_checkpoint']['model_hash'])
    check('first_config_matches_actual_config',after['config_hash']==cfg.digest() and after['config']==cfg.as_dict())
    check('iteration159_from158',after['counters']['iteration']==159)
    check('attempted256',after['counters']['attempted_updates']-source['counters']['attempted_updates']==256)
    accepted=after['counters']['effective_updates']-source['counters']['effective_updates']
    report=read_json(run/'learner/reports/stage_000001.json')
    check('effective_updates_counted_from_acceptance',accepted==sum(int(x['accepted']) for x in report['updates']))
    for i,row in enumerate(report['updates']):
        phase='value' if (source['counters']['attempted_updates']+i)%3==2 else 'policy'
        check(f'update{i}:PPV',row['phase']==phase)
        if row['accepted']:
            factor=({'encoder':.1,'policy':0.,'value':1.} if phase=='value' else {'encoder':.25*.25,'policy':.25,'value':0.})
            check(f'update{i}:real_phase_LR',all(abs(row['optimizer_group_learning_rates'][k]-cfg.learning_rate*row['accepted_lr_scale']*v)<1e-15 for k,v in factor.items()))
    check('fresh_policy_draws_used',sum(v for k,v in report['actual_policy_replay_origin_sources'].items() if k.startswith('fresh:'))>0)
    item=next(r for r in records if r['stage']==1)
    check('shard_digest',sha(inside(root,item['path']))==item['sha256'])
    check('metadata_digest',sha(inside(root,item['metadata']))==item['metadata_sha256'])
    meta=read_json(inside(root,item['metadata']));samples=read_selfplay_shard(inside(root,item['path']))
    check('collector_own_preceding158',meta['actor_model_hash']==seed['model_hash'])
    check('training_role',meta['layout_role']=='train')
    backend=MultiLayoutBackend(load_role(inside(root,cfg.layout_directory),'train'),cfg.seed)
    games=list(complete_games(samples,meta));check('256_real_complete_games',len(games)==256)
    native_steps=0
    for game,rows in games:
        state=backend.new_state(game['provenance']['setup_seed'])
        check(f'game{game["index"]}:paired_mode_count',len(game['trace'])==len(rows))
        for action,sample in zip(game['trace'],rows):
            check(f'game{game["index"]}:native_row{sample.ply}',not state.is_terminal() and int(state.native_state.key())==int(sample.state_hash) and state.current_player()==sample.player and action in state.legal_actions() and np.array_equal(state.legal_actions(),sample.legal_actions) and np.array_equal(state.repetition_contexts(),sample.repetition_contexts) and np.array_equal(state.feature_planes().astype(np.float16),sample.observation))
            expected=np.zeros(3,np.float32);expected[1+int(game['result_red'])*sample.player]=1.
            check(f'game{game["index"]}:true_WDL{sample.ply}',np.array_equal(sample.wdl,expected))
            state.apply(action);native_steps+=1
        check(f'game{game["index"]}:real_terminal',state.is_terminal() and state.terminal_value(1)==game['result_red'] and state.terminal_reason()==game['reason'])
    for p,h in plan['source'].items():check('source:'+p,sha(inside(root,p))==h)
    for e in plan['historical_stops']:check('STOP:'+e['path'],sha(inside(root,e['path']))==e['sha256'])
    for e in plan['frozen_runs']:check('old_CURRENT:'+e['directory'],sha(inside(root,e['directory'])/'CURRENT.json')==e['current_sha256'])
    check('cpu_no_gpu_initialized',not torch.cuda.is_initialized())
    result=dict(at=utc_now(),all_checks_passed=all(checks.values()),checks_count=len(checks),checks=checks,native_games=256,native_plies=native_steps,
        new_training_games=256,effective_updates=accepted,source=plan['source_checkpoint'],latest_first_model_hash=after['model_hash'],
        phase_learning_rates={'policy':{'encoder':3.125e-6,'head':1.25e-5},'value':{'encoder':5e-6,'head':5e-5}},
        interpretation='Verifies actual online startup, not strength improvement. All old protected states retained; no extra GPU task or learning in this audit.')
    atomic_json(inside(root,args.output),result)
    print({k:result[k] for k in ('all_checks_passed','checks_count','native_games','native_plies','effective_updates','latest_first_model_hash')})


if __name__=='__main__':main()
