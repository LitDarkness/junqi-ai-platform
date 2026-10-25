"""Register an independent immutable completion of the paused full64 gate."""
from pathlib import Path
import hashlib
import json


def main():
    import torch
    from open_v4_runtime import read_json,atomic_json,utc_now
    from datetime import datetime,timezone
    root=Path(__file__).resolve().parents[1]
    original_path='config/experiments/open_v4_long_behavior_20261004_081501.json'
    original=read_json(root/original_path);old=root/original['directory']
    cached_path=old/'arena/block_000012_full.json';cached=read_json(cached_path)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    directory=f'runs/open_junqi_v4_frozen_completion_{stamp}'
    def entry(path):
        path=Path(path);path=(root/path).resolve()
        if not path.is_relative_to(root):raise ValueError('outside alphazero')
        return {'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    frozen=[entry(old/'CURRENT.json'),entry(old/'PROGRESS.json'),entry(original['parent_config']),entry(original['suite']['path']),
            entry(root/read_json(root/original['parent_config'])['layout_directory']/'manifest.json')]
    frozen+=[entry(r['directory']+'/CURRENT.json') for r in original['frozen_runs']]
    frozen+=[entry(r['path']) for r in original['historical_stops']]+[entry(p) for p in original['stop_paths']]
    current=read_json(old/'CURRENT.json');participants={}
    for name in ('champion','parent','control','candidate','repeat'):
        path=(old/f'{name}/checkpoints/learner_stage_000024.pt') if name in ('control','candidate') else root/cached['checkpoints'][name]['path']
        item=entry(path);saved=torch.load(path,map_location='cpu',weights_only=False)
        item.update(model_hash=saved['model_hash'],iteration=saved['counters']['iteration'],leaf_value_scale=saved['leaf_value_scale'])
        if name in ('control','candidate'):
            reference=current['experimental_collectors'][name]
            if item['sha256']!=reference['sha256'] or item['model_hash']!=reference['model_hash']:
                raise ValueError('latest complete branch identity mismatch')
        participants[name]=item
    source=dict(original['source'])
    for name in ('tools/open_v4_cached_arena.py','tools/run_open_v4_frozen_completion.py','tools/prepare_open_v4_frozen_completion.py',
                 'start_open_v4_frozen_completion.ps1','tests/test_open_v4_cached_arena.py'):
        source[name]=entry(name)['sha256']
    plan={'schema':'junqi-frozen-completion-plan-1','registered_at':utc_now(),'directory':directory,
          'original_plan':entry(original_path),'cached_arena':entry(cached_path),'frozen_files':frozen,'source':source,
          'participants':participants,'lock_directories':[r['directory'] for r in original['frozen_runs']]+[original['directory']],
          'stop_paths':[directory+'/STOP',directory+'/supervision/STOP'],'inference_contract':cached['inference_contract'],
          'engine_contract':{'cpuct':1.5,'cache_size':0,'leaves_per_tree':1,'reuse_roots':False,'cuda_graphs':False,
              'physical_batch':32,'precision':'BF16','root_noise':False,'root_guard':False,'budget':256},
          'authorization':'Latest user explicitly requested efficiency improvements and continued training. New independent directory only; paused original and its STOP markers remain frozen.',
          'phase':'frozen_development_evaluation','new_training_games':0,'completed_updates':0,
          'planned_fresh_development_games':2916,'planned_reused_development_games':1296,'fallback_fresh_development_games':4212,
          'cache_rule':'Only immutable champion26/source38 teacher baselines, same full64 suite, seed, native rules, teacher bytes, full weights/Value scales, BF16 physical32 and exact search settings. Reaggregation reproduces all original intervals/gate. Before reuse, replay 648 actual fresh source38 games; all full traces and terminal reasons/results must match. Any mismatch disables cache and reruns all1296 baseline games. Old compute cost is not counted as fresh.',
          'qualification':'Original full64 route and Champion gates unchanged: six teacher paired lower95>0 versus Champion/source/control, three direct lower95>.5, duplicate source drift<=.02, original production gate.082/084 related; no independent final claim.',
          'next_action':'Review complete4212 raw records, fresh/cached provenance, source repeat and full states. If original62 route qualifies, preregister a new24-stage fresh continuation from qualified62,256new/64updates perstage with own preceding Learner, latest3 replay shards and frozen62 protection. If route unconfirmed, preserve evidence and audit old/new Value on saved training holdout, then preregister a different single major hypothesis using latest complete research62; do not automatically extend failed old factor or rewind26. No second GPU task while this job runs.',
          'failure_policy':'STOP honored without clearing. If unexpected disappearance, confirm no lock owner and inspect last complete committed part and logs; at most one reviewed explicit resume with the same contract; repeated fault preserves evidence and stops.'}
    path=root/f'config/experiments/open_v4_frozen_completion_{stamp}.json'
    if path.exists() or (root/directory).exists():raise ValueError('unique new registration required')
    atomic_json(path,plan)
    print(json.dumps({'config':str(path.relative_to(root)),'directory':directory,'registered':True}))


if __name__=='__main__':main()
