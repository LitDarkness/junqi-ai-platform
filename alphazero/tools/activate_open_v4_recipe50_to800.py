"""Publish newly authorized600to800 run; keep completed pilot evidence frozen."""
from pathlib import Path
import copy,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from open_v4_recipe50_to800_contract import verify_single_prerequisite
    reg=read_json(ROOT/'reports/recipe50_to800_latest_registration.json');path=ROOT/reg['plan'];plan=read_json(path)
    verify_single_prerequisite(ROOT,plan)
    assert not (ROOT/plan['directory']).exists() and not lock_held(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name,h in plan['source'].items():assert sha(ROOT/name)==h,name
    for name in ('open_junqi_v4_repair_pilot','open_junqi_v4_fresh_continuation'):
        p=ROOT/f'runs/{name}/supervision/PLAN.json';master=read_json(p)
        assert not master['paused_by_user'] and master['optimization']['state']=='horizon_complete' and master['optimization']['latest_iteration']==600
        master['before_recipe50_to800_optimization']=copy.deepcopy(master['optimization']);master['before_recipe50_to800_auto_next_stage']=copy.deepcopy(master['auto_next_stage'])
        action='Launch new registered same-recipe600to800 once; exact full600 state, own preceding learner256fresh/64PPV, unchanged50/50 with actual082. Preserve original400 qualification baseline and negative600 evidence. Scores never stop. No other methods or unregistered extension.'
        opt=dict(type=plan['route'],state='ready_to_launch',config=reg['plan'],config_sha256=sha(path),run_directory=plan['directory'],status_tool='tools/run_open_v4_recipe50_to800.py',review_tool='tools/review_open_v4_recipe50_to800.py',formal_review_tool='tools/review_open_v4_recipe50_to800_full.py',launcher='start_open_v4_recipe50_to800.ps1',stop_tool='stop_open_v4_recipe50_to800.ps1',training_variants=['learner'],training_active=False,automatic_continuation_enabled=True,source_iteration=600,latest_iteration=600,source_model_hash=plan['source_checkpoint']['model_hash'],latest_model_hash=plan['source_checkpoint']['model_hash'],qualification_reference_iteration=400,source_complete_state_preserved=True,no_weight_reset=True,updates_per256fresh=64,target_iteration=800,maximum_stages=200,new_training_games=0,effective_updates=0,registered_at=plan['registered_at'],phase='launching',recipe={'selfplay':.5,'teacher_selfplay':0.,'vs_actual_082':.5},training_teacher_ids=[0],training_search_unchanged=True,allow_major_mechanism_change=False,user_authorization=plan['user_contract'],registration_report=reg['report'],supervision_automation_pending_first_actual_stage=True)
        master.update(paused_by_user=False,pause_reason=None,optimization=opt,updated_at=utc_now(),next_action=action,automatic_continuation_enabled=True,latest_user_authorization=plan['user_contract'])
        master['auto_next_stage']=dict(type='single_online_supervisor',state='ready_to_launch',phase='launching',config=reg['plan'],run_directory=plan['directory'],status_tool=opt['status_tool'],launcher=opt['launcher'],ownership_lock='runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock',requires_next_heartbeat=False,automatic_continuation_enabled=True,training_active=False,waiting_controller_active=False,requires_user_resumption=False,condition=action)
        master['continuation'].update(through_iteration=800,state='new_authorized_same_recipe600to800',requires_new_explicit_authorization=False,automatic_continuation_enabled=True)
        atomic_json(p,master)
    atomic_json(ROOT/reg['report']/'ACTIVATION.json',dict(at=utc_now(),plan=reg['plan'],plan_sha256=sha(path),historical_STOPs_CURRENT_and_negative600_preserved=True,latest_full600_source=True,qualification400_not_reset=True))
    print(json.dumps(reg,ensure_ascii=False))
if __name__=='__main__':main()
