"""Activate only the reviewed new branch; never clear historical STOP files."""
from pathlib import Path
import copy,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from open_v4_engineered_contract import verify_single_prerequisite
    registration=read_json(ROOT/'reports/search_engineering_20261006/REGISTRATION.json')
    planpath=ROOT/registration['plan'];plan=read_json(planpath);verify_single_prerequisite(ROOT,plan)
    assert not (ROOT/plan['directory']).exists(),'fresh registered branch only'
    assert not lock_held(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name,h in plan['source'].items():assert sha(ROOT/name)==h,name
    for name in ('open_junqi_v4_repair_pilot','open_junqi_v4_fresh_continuation'):
        path=ROOT/f'runs/{name}/supervision/PLAN.json';master=read_json(path)
        assert master['paused_by_user']
        master['before_engineered_performance']=copy.deepcopy(master['optimization'])
        master['before_engineered_auto_next_stage']=copy.deepcopy(master['auto_next_stage'])
        master.update(paused_by_user=False,pause_reason=None,updated_at=utc_now(),
            next_action='Launch reviewed new performance-only branch once. Verify first363 own-model256fresh/64PPV complete state, finish frozen362full in the same owner, then unchanged online learning to400; no recipe/method change or unregistered extension.')
        opt=dict(type=plan['route'],state='ready_to_launch',config=registration['plan'],config_sha256=sha(planpath),
            run_directory=plan['directory'],status_tool='tools/run_open_v4_engineered.py',launcher='start_open_v4_engineered.ps1',
            stop_tool='stop_open_v4_engineered.ps1',review_tool='tools/review_open_v4_engineered.py',
            formal_review_tool='tools/review_open_v4_engineered_full.py',training_variants=['learner'],
            training_active=False,automatic_continuation_enabled=True,source_iteration=362,latest_iteration=362,
            source_model_hash=plan['source_checkpoint']['model_hash'],latest_model_hash=plan['source_checkpoint']['model_hash'],
            source_complete_state_preserved=True,no_weight_reset=True,updates_per256fresh=64,target_iteration=400,maximum_stages=38,
            new_training_games=0,effective_updates=0,registered_at=plan['registered_at'],phase='launching',
            performance_only_engineering=True,training_search_unchanged=True,allow_major_mechanism_change=False,
            last_strength_screen=dict(iteration=362,arena=plan['pending_screen']['path'],sha256=plan['pending_screen']['sha256'],full_gate_pending=True),
            performance_report='reports/search_engineering_20261006/REPORT.html')
        master['optimization']=opt
        master['auto_next_stage']=dict(type='single_online_supervisor',state='ready_to_launch',
            config=registration['plan'],run_directory=plan['directory'],status_tool=opt['status_tool'],launcher=opt['launcher'],
            ownership_lock='runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock',
            requires_next_heartbeat=False,automatic_continuation_enabled=True,training_active=False,
            waiting_controller_active=False,requires_user_resumption=False,
            condition='Approved unchanged mechanism until400; score uncertainty or decline never stops training; safety/integrity and userSTOP always take priority.')
        atomic_json(path,master)
    atomic_json(ROOT/registration['report']/'ACTIVATION.json',dict(at=utc_now(),plan=registration['plan'],plan_sha256=sha(planpath),
        only_new_branch_unpaused=True,historical_STOPs_preserved=True,user_authorization=plan['user_contract']))
    print(json.dumps(registration,ensure_ascii=False))
if __name__=='__main__':main()
