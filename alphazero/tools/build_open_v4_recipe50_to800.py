"""Copy a separately bounded continuation entry; never edit frozen600 sources."""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
changes = []

def make(old, new, replacements):
    source = (ROOT / old).read_text(encoding='utf-8-sig')
    for before, after in replacements:
        if before not in source:
            raise ValueError('Missing source fragment: ' + repr(before))
        source = source.replace(before, after)
    target = ROOT / new
    if target.exists():
        if target.read_text(encoding='utf-8') != source:
            raise ValueError('New entry differs from this recorded copy: ' + new)
    else:
        target.write_text(source, encoding='utf-8')
    changes.append(dict(original=old, created=new,
        original_sha256=hashlib.sha256((ROOT / old).read_bytes()).hexdigest(),
        new_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        replacements=[dict(before=b, after=a) for b, a in replacements]))

make('tools/open_v4_recipe50_to600_contract.py', 'tools/open_v4_recipe50_to800_contract.py', [
    ('to600', 'to800'),
    ('junqi-recipe50-424to800-plan-1', 'junqi-recipe50-600to800-plan-1'),
    ('(424, 600, 176)', '(600, 800, 200)'),
    ('complete424to800', 'complete600to800'),
    ('range(12, 177, 12)) + [176]', 'range(12, 201, 12)) + [200]'),
    ('range(24,177,24)) + [176]', 'range(24,201,24)) + [200]'),
    ("plan['source_checkpoint']['iteration'] != 424", "plan['source_checkpoint']['iteration'] != 600"),
    ('424learning source', '600learning source'),
    ("auth['target_iteration'] != 600 or auth['source_iteration'] != 424", "auth['target_iteration'] != 800 or auth['source_iteration'] != 600"),
    ('source424_complete', 'source600_complete'),
])
make('tools/run_open_v4_recipe50_to600.py', 'tools/run_open_v4_recipe50_to800.py', [
    ('open_v4_recipe50_to600_contract', 'open_v4_recipe50_to800_contract'),
    ('junqi-recipe50-424to600-plan-1', 'junqi-recipe50-600to800-plan-1'),
    ('Exact424 learning state', 'Exact600 learning state'),
    ('directory.mkdir(parents=True, exist_ok=True)', "directory.mkdir(parents=True, exist_ok=True)\n        launch_ref = master['optimization']['launch_receipt']\n        if sha(inside(launch_ref['path'])) != launch_ref['sha256']: raise ValueError('detached launch receipt changed')\n        launch = read_json(inside(launch_ref['path']))\n        if launch['pid'] != os.getpid() or launch['plan'] != str(plan_path.relative_to(root)).replace('\\\\', '/'): raise ValueError('detached owner identity mismatch')\n        atomic_json(directory / 'LAUNCH.json', launch)"),
])
make('tools/review_open_v4_recipe50_to600.py', 'tools/review_open_v4_recipe50_to800.py', [
    ('open_v4_recipe50_to600_contract', 'open_v4_recipe50_to800_contract'),
    ('authorized50_50_recipe_to600', 'authorized50_50_recipe_to800'),
    ("plan['target_iteration'] == 600 and plan['source_checkpoint']['iteration'] == 424", "plan['target_iteration'] == 800 and plan['source_checkpoint']['iteration'] == 600"),
    ('through600', 'through800'),
    ('continuation400to424_prior_effective_updates=1536', 'continuation400to424_prior_effective_updates=1536, continuation424to600_prior_online_games=45056, continuation424to600_prior_effective_updates=11264'),
])
make('tools/review_open_v4_recipe50_to600_full.py', 'tools/review_open_v4_recipe50_to800_full.py', [
    ('from run_open_v4_recipe50_to600 import', 'from run_open_v4_recipe50_to800 import'),
    ('from424 through600', 'from600 through800'),
])
make('tests/test_open_v4_recipe50_to600.py', 'tests/test_open_v4_recipe50_to800.py', [
    ('open_v4_recipe50_to600_contract', 'open_v4_recipe50_to800_contract'),
    ('run_open_v4_recipe50_to600', 'run_open_v4_recipe50_to800'),
    ("SOURCE = ROOT / 'runs/open_junqi_v4_recipe50_20261007_012129/learner/checkpoints/learner_stage_000024.pt'", "SOURCE = ROOT / 'runs/open_junqi_v4_recipe50_to600_20261007_034300/learner/checkpoints/learner_stage_000176.pt'"),
    ('test_complete424_learning_state', 'test_complete600_learning_state'),
    ("self.source['counters']['iteration'], 424", "self.source['counters']['iteration'], 600"),
    ('(1342, 1343, 1344)', '(1542, 1543, 1544)'),
    ("schema='junqi-recipe50-424to600-plan-1',route='recipe50_selfplay_vs082_to600',source_checkpoint=dict(iteration=424),target_iteration=600,maximum_stages=176,screen_boundaries=[6]+list(range(12,177,12))+[176],monitor084_boundaries=list(range(24,177,24))+[176]", "schema='junqi-recipe50-600to800-plan-1',route='recipe50_selfplay_vs082_to800',source_checkpoint=dict(iteration=600),target_iteration=800,maximum_stages=200,screen_boundaries=[6]+list(range(12,201,12))+[200],monitor084_boundaries=list(range(24,201,24))+[200]"),
    ("[('target_iteration',601),('maximum_stages',177),('monitor084_boundaries',[176])]", "[('target_iteration',801),('maximum_stages',201),('monitor084_boundaries',[200])]"),
    ("bad['participants']['parent']=dict(iteration=424)", "bad['participants']['parent']=dict(iteration=600)"),
])
make('tools/activate_open_v4_recipe50_to600.py', 'tools/activate_open_v4_recipe50_to800.py', [
    ('to600', 'to800'),
    ('424', '600'),
    ('target_iteration=600', 'target_iteration=800'),
    ('maximum_stages=176', 'maximum_stages=200'),
    ('through_iteration=600', 'through_iteration=800'),
])
make('stop_open_v4_recipe50_to600.ps1', 'stop_open_v4_recipe50_to800.ps1', [
    ('recipe50_selfplay_vs082_to600', 'recipe50_selfplay_vs082_to800'),
])
path = ROOT / 'reports/recipe50_to800_entry_changes.json'
path.write_text(json.dumps(dict(new_entry_only=True, old_frozen_files_unchanged=True,
    training_and_evaluation_algorithms_unchanged=True, changes=changes), ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(new_files=[r['created'] for r in changes], evidence=str(path.relative_to(ROOT))), ensure_ascii=False))
