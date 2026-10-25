"""Publish the new authorized recipe entry; preserve completed400 and all STOPs."""
from pathlib import Path
import copy, hashlib, json
ROOT = Path(__file__).resolve().parents[1]


def main():
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    from open_v4_recipe50_contract import verify_single_prerequisite
    registration = read_json(ROOT / 'reports/recipe50_latest_registration.json')
    planpath = ROOT / registration['plan']; plan = read_json(planpath)
    verify_single_prerequisite(ROOT, plan)
    assert not (ROOT / plan['directory']).exists(), 'fresh branch only'
    assert not lock_held(ROOT / 'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name, digest in plan['source'].items(): assert sha(ROOT / name) == digest, name
    for name in ('open_junqi_v4_repair_pilot', 'open_junqi_v4_fresh_continuation'):
        path = ROOT / f'runs/{name}/supervision/PLAN.json'; master = read_json(path)
        assert not master['paused_by_user'] and master['optimization']['state'] == 'horizon_complete' and master['optimization']['latest_iteration'] == 400
        master['before_recipe50_optimization'] = copy.deepcopy(master['optimization'])
        master['before_recipe50_auto_next_stage'] = copy.deepcopy(master['auto_next_stage'])
        master.update(paused_by_user=False, pause_reason=None, updated_at=utc_now(),
                      next_action='Launch registered recipe50 once; freeze actual400/082/084 source baseline, then own-preceding-Learner256fresh/64PPV until424. Fresh50%selfplay/50%actual082, teacher selfplay0. Verify first401 and natural legacy replay exit atstage3. Negative scores do not stop; endpoint requires review.')
        opt = dict(type=plan['route'], state='ready_to_launch', config=registration['plan'], config_sha256=sha(planpath),
                   run_directory=plan['directory'], status_tool='tools/run_open_v4_recipe50.py', launcher='start_open_v4_recipe50.ps1',
                   stop_tool='stop_open_v4_recipe50.ps1', review_tool='tools/review_open_v4_recipe50.py', formal_review_tool='tools/review_open_v4_recipe50_full.py',
                   training_variants=['learner'], training_active=False, automatic_continuation_enabled=True, source_iteration=400, latest_iteration=400,
                   source_model_hash=plan['source_checkpoint']['model_hash'], latest_model_hash=plan['source_checkpoint']['model_hash'],
                   source_complete_state_preserved=True, no_weight_reset=True, updates_per256fresh=64, target_iteration=424, maximum_stages=24,
                   new_training_games=0, effective_updates=0, registered_at=plan['registered_at'], phase='launching',
                   recipe={'selfplay':.5,'teacher_selfplay':0.,'vs_actual_082':.5}, training_teacher_ids=[0], training_search_unchanged=True,
                   allow_major_mechanism_change=False, user_authorization=plan['user_contract'], registration_report=registration['report'])
        master['optimization'] = opt
        master['auto_next_stage'] = dict(type='single_online_supervisor', state='ready_to_launch', config=registration['plan'], run_directory=plan['directory'],
                                       status_tool=opt['status_tool'], launcher=opt['launcher'], ownership_lock='runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock',
                                       requires_next_heartbeat=False, automatic_continuation_enabled=True, training_active=False, waiting_controller_active=False,
                                       requires_user_resumption=False, condition='Authorized50/50 recipe pilot to424; sole owner continues immediately, scores never stop. UserSTOP/safety/integrity always take priority. No other unapproved learning changes or extension.')
        atomic_json(path, master)
    atomic_json(ROOT / registration['report'] / 'ACTIVATION.json', dict(at=utc_now(), plan=registration['plan'], plan_sha256=sha(planpath),
                fresh_recipe_authorization_applied=True, historical_STOPs_and_completed400_preserved=True, authorization=plan['user_contract']))
    print(json.dumps(registration, ensure_ascii=False))


if __name__ == '__main__': main()
