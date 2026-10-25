"""Create a separately frozen continuation entry without editing pilot sources."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
changes=[]
def make(old,new,replacements):
    src=(ROOT/old).read_text(encoding='utf-8-sig')
    original=src
    for before,after in replacements:
        if before not in src:raise ValueError('Missing source fragment: '+repr(before))
        src=src.replace(before,after)
    target=ROOT/new
    if target.exists():raise ValueError('New entry already exists: '+new)
    target.write_text(src,encoding='utf-8')
    changes.append(dict(original=old,created=new,original_sha256=hashlib.sha256((ROOT/old).read_bytes()).hexdigest(),new_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),replacements=[dict(before=b,after=a) for b,a in replacements]))

make('tools/open_v4_recipe50_contract.py','tools/open_v4_recipe50_to600_contract.py',[
    ('complete400','complete424'),('source400','qualification400'),
    ('junqi-recipe50-400to424-plan-1','junqi-recipe50-424to600-plan-1'),
    ("'recipe50_selfplay_vs082'","'recipe50_selfplay_vs082_to600'"),
    ('(400, 424, 24)','(424, 600, 176)'),
    ('bounded complete424to424 recipe pilot required','bounded complete424to600 continuation required'),
    ("plan['participants']['parent'] != plan['source_checkpoint']","plan['participants']['parent'] != plan['qualification_reference400']"),
    ('[6, 12, 24]', '[6] + list(range(12, 177, 12)) + [176]'),
    ("if plan['performance_stop_enabled']", "if plan['monitor084_boundaries'] != list(range(24,177,24)) + [176]:\n        raise ValueError('preregistered084 coverage monitoring required')\n    if plan['qualification_reference400']['iteration'] != 400 or plan['source_checkpoint']['iteration'] != 424:\n        raise ValueError('424learning source and400 qualification reference must remain distinct')\n    if plan['performance_stop_enabled']"),
    ("plan['benchmark']):","plan['benchmark'], plan['frozen_baseline']):"),
    ("auth['allow_other_learning_changes']:","auth['allow_other_learning_changes'] or auth['target_iteration'] != 600 or auth['source_iteration'] != 424:"),
    ("return dict(qualification400_complete=True", "return dict(source424_complete=True, qualification400_preserved=True")
])
make('tools/run_open_v4_recipe50.py','tools/run_open_v4_recipe50_to600.py',[
    ('open_v4_recipe50_contract','open_v4_recipe50_to600_contract'),
    ('junqi-recipe50-400to424-plan-1','junqi-recipe50-424to600-plan-1'),
    ('Exact400 learning state','Exact424 learning state'),
    ("baseline=teacher_monitor('startup_baseline',{'parent':plan['source_checkpoint'],'repeat':plan['source_checkpoint']},teachers,[0,1])", "baseline=read_json(inside(plan['frozen_baseline']['path']))\n                    if sha(inside(plan['frozen_baseline']['path']))!=plan['frozen_baseline']['sha256']:raise ValueError('original400 baseline changed')\n                    baseline_path=directory/'arena/startup_baseline.json';atomic_json(baseline_path,baseline)\n                    progress['evaluations']['startup_baseline']={'path':str(baseline_path.relative_to(root)),'sha256':sha(baseline_path),'review':baseline['review'],'imported_from':plan['frozen_baseline']}\n                    atomic_json(directory/'verification/IMPORTED_BASELINE400.json',{'original':plan['frozen_baseline'],'frozen400_not_replaced':True,'new_training_games':0,'effective_updates':0})"),
    ("if stage==plan['maximum_stages']:\n                        teacher_monitor('iteration424_true084'", "if stage in plan['monitor084_boundaries']:\n                        teacher_monitor(f\"iteration{source['counters']['iteration']+stage}_true084\""),
])
make('tools/review_open_v4_recipe50.py','tools/review_open_v4_recipe50_to600.py',[
    ('open_v4_recipe50_contract','open_v4_recipe50_to600_contract'),
    ('authorized50_50_recipe_to424','authorized50_50_recipe_to600'),
    ("plan['target_iteration'] == 424","plan['target_iteration'] == 600 and plan['source_checkpoint']['iteration'] == 424 and plan['participants']['parent']['iteration'] == 400"),
    ("plan['source_checkpoint']['model_hash'] for x in arena['checkpoints'].values()", "plan['qualification_reference400']['model_hash'] for x in arena['checkpoints'].values()"),
    ("if name.endswith('_full') or name == 'iteration424_true084':", "if name.endswith('_true084'):\n            from open_v4_monitor_evidence import reconstruct_monitor\n            proof=reconstruct_monitor(arena, selected, plan, root)\n            check('monitor_independent_statistics_native:'+name, proof['all_checks_passed'])\n            arena_reviews[name]=proof\n            continue\n        if name.endswith('_full'):"),
    ("name == 'iteration424_true084'", "name.endswith('_true084')"),
    ('mechanism_changed=True, authorized_recipe_only=True','mechanism_changed=False, authorized_recipe_only=True, continuation400to424_prior_online_games=6144, continuation400to424_prior_effective_updates=1536'),
    ('through424','through600'),
])
make('tools/review_open_v4_recipe50_full.py','tools/review_open_v4_recipe50_to600_full.py',[
    ('from run_open_v4_recipe50 import','from run_open_v4_recipe50_to600 import'),
    ('continuation through400','same-recipe continuation from424 through600'),
])
make('start_open_v4_recipe50.ps1','start_open_v4_recipe50_to600.ps1',[
    ('run_open_v4_recipe50.py','run_open_v4_recipe50_to600.py'),
    ('reports/recipe50_launches','reports/recipe50_to600_launches'),
    ('recipe50_$TaskStamp','recipe50_to600_$TaskStamp'),
])
make('stop_open_v4_recipe50.ps1','stop_open_v4_recipe50_to600.ps1',[
    ("'recipe50_selfplay_vs082'","'recipe50_selfplay_vs082_to600'"),
])
path=ROOT/'reports/recipe50_to600_entry_changes.json'
path.write_text(json.dumps(dict(new_entry_only=True,old_frozen_files_unchanged=True,changes=changes),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(new_files=[r['created'] for r in changes],evidence=str(path.relative_to(ROOT))),ensure_ascii=False))
