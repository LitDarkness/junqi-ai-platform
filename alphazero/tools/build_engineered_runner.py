"""New independent entry point; never edit a frozen previous runner/collector."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=ROOT/'python/junqi_open_v4/root_explore_games.py'
    # Capture-inclusive actual collector groups did not establish a gain.
    # Retain the untouched original training search; add observation only.
    code=p.read_text(encoding='utf-8')
    code=code.replace('"iteration": iteration, "games": completed',
        '"engineering_metrics": {"inference_strategy":"original_dynamic_training_search","graph_capture_seconds":None,"graph_replays":search._graph_replays,"graph_shapes":sorted(search._graphs),"graph_failed_shapes":sorted(search._graph_failures),"exact_physical_shapes":True}, "iteration": iteration, "games": completed')
    (ROOT/'python/junqi_open_v4/engineered_games.py').write_text(code,encoding='utf-8')
    code=(ROOT/'tools/run_open_v4_until400.py').read_text(encoding='utf-8')
    code=code.replace('open_v4_until400_contract','open_v4_engineered_contract')
    code=code.replace('junqi-until400-isolated-teachers-plan-1','junqi-engineered-362to400-plan-1')
    code=code.replace('from junqi_open_v4.root_explore_games import collect_training_games','from junqi_open_v4.engineered_games import collect_training_games')
    code=code.replace('from junqi_open_v4.arena import run_arena, select_families',
        'from junqi_open_v4.arena import select_families\n    from junqi_open_v4.resumable_arena import run_resumable_arena as run_arena')
    code=code.replace('from junqi_search_v3.search import BatchedNeuralPUCT','from junqi_search_v3.graph_inference import GraphNeuralPUCT as BatchedNeuralPUCT')
    code=code.replace('cuda_graphs=False), check)', 'cuda_graphs=True), check)')
    code=code.replace("raise ValueError('small-screen source drift exceeds predeclared2pp; review measurement context')",
        "arena['review']['measurement_review_required']=True; arena['review']['positive_signal']=False")
    code=code.replace('full_seed_state_preserved_except_restored_original_policy_group_lr','full_seed_state_preserved_with_identical_actual_group_lr')
    code=code.replace('Identical current phase LR and full Adam/scheduler/counters/RNG; teacher identities corrected only','Identical full current state and actual teacher providers; performance scheduling only')
    code=code.replace('"segment_parent": plan["source_checkpoint"]', '"segment_parent": plan["participants"]["parent"]')
    code=code.replace('"evaluations": {},', '"evaluations": {"startup_baseline":plan["frozen_baseline"]},')
    code=code.replace('seed=plan["evaluation_seed"])', '''seed=plan["evaluation_seed"], checkpoint_directory=directory/'arena_chunks'/name,
                              binding={'plan_sha256':sha(plan_path),'participants':participants,'teachers':plan['teachers'],
                                       'physical_batch':32,'precision':'BF16','source':plan['source']},check_stop=check)''')
    code=code.replace("seed=plan['evaluation_seed'])", '''seed=plan['evaluation_seed'],checkpoint_directory=directory/'arena_chunks'/name,
                binding={'plan_sha256':sha(plan_path),'participants':participants,'teachers':plan['teachers'],
                         'physical_batch':32,'precision':'BF16','source':plan['source']},check_stop=check)''')
    code=code.replace('"learning_elapsed_seconds": learning_elapsed,', '"engineering_metrics":meta["engineering_metrics"], "learning_elapsed_seconds": learning_elapsed,')
    code=code.replace('arena["teacher_runtime_identity"] = teachers.identity_report()',
        '''arena["engineering_metrics"]={k:{'graph_capture_seconds':a.graph_capture_seconds,'graph_replays':a._graph_replays,'graph_shapes':sorted(a._graphs),'graph_failed_shapes':sorted(a._graph_failures)} for k,a in agents.items()}
            arena["teacher_runtime_identity"] = teachers.identity_report()''')
    # Save the first resumed online commit, then finish the separately frozen362
    # full gate. The current363 learner is never replaced by the qualified model.
    marker='                    if not is_screen_stage(stage, plan): continue'
    insertion='''                    if stage == 1:
                        pending = dict(candidate=plan['source_checkpoint'],control=plan['participants']['control'],
                            parent=plan['participants']['parent'],repeat=plan['participants']['parent'],
                            guard86=plan['participants']['guard86'],champion=progress['champion'])
                        full362 = evaluate('pending_iteration362_full',pending,64,teachers)
                        atomic_json(directory/'PENDING_362_RESULTS.json',{'review':full362['review'],
                            'native_verification':full362['native_verification'],'training_learner_not_reset':True})
                        if full362['review']['route_confirmed'] and progress['champion']['iteration'] < 362:
                            promoted=torch.load(inside(plan['source_checkpoint']['path']),map_location='cpu',weights_only=False)
                            promoted['role']='champion';promoted['lineage']['qualification']=progress['evaluations']['pending_iteration362_full']
                            path=directory/'champions/champion_000362.pt';atomic_checkpoint(path,promoted)
                            progress['champion']={'path':str(path.relative_to(root)),'sha256':sha(path),
                                'model_hash':plan['source_checkpoint']['model_hash'],'iteration':362}
                            commit();publish_current()
'''
    assert marker in code
    code=code.replace(marker,insertion+marker)
    (ROOT/'tools/run_open_v4_engineered.py').write_text(code,encoding='utf-8')
    code=(ROOT/'tools/review_open_v4_until400.py').read_text(encoding='utf-8')
    code=code.replace('open_v4_until400_contract','open_v4_engineered_contract')
    code=code.replace('302 + stage',"plan['source_checkpoint']['iteration'] + stage").replace('302+stage',"plan['source_checkpoint']['iteration']+stage")
    code=code.replace('302+n',"plan['source_checkpoint']['iteration']+n").replace('source_iteration=302',"source_iteration=plan['source_checkpoint']['iteration']")
    code=code.replace("    for name in ('NEXT_STAGE', 'SUPERVISOR', 'TRAINING'):","    runtime=read_json(run/'STATUS.json'); expected_owned=runtime['state']!='horizon_complete'\n    for name in ('NEXT_STAGE', 'SUPERVISOR', 'TRAINING'):")
    code=code.replace("lock_held(root / f'runs/open_junqi_v4_repair_pilot/supervision/{name}.lock'))", "lock_held(root / f'runs/open_junqi_v4_repair_pilot/supervision/{name}.lock')==expected_owned)")
    code=code.replace("lock_held(run / f'supervision/{name}.lock'))", "lock_held(run / f'supervision/{name}.lock')==expected_owned)")
    code=code.replace('waiting_controller_active=False)', "waiting_controller_active=False, automatic_continuation_enabled=expected_owned)")
    code=code.replace('training_active=True,', 'training_active=expected_owned,')
    code=code.replace("p['optimization'].update(state=current['state'],", "p['optimization'].update(training_active=expected_owned, automatic_continuation_enabled=expected_owned,state=current['state'],")
    (ROOT/'tools/review_open_v4_engineered.py').write_text(code,encoding='utf-8')
    code=(ROOT/'tools/review_open_v4_until400_full.py').read_text(encoding='utf-8').replace('from run_open_v4_until400 import','from run_open_v4_engineered import')
    (ROOT/'tools/review_open_v4_engineered_full.py').write_text(code,encoding='utf-8')
    for name in ('start','stop'):
        code=(ROOT/f'{name}_open_v4_until400.ps1').read_text(encoding='utf-8')
        code=code.replace('run_open_v4_until400.py','run_open_v4_engineered.py').replace('until400_launches','engineered_launches').replace('until400_', 'engineered_')
        code=code.replace('restored_mechanism_isolated_teachers_to400','engineered_fixed_mechanism_to400')
        (ROOT/f'{name}_open_v4_engineered.ps1').write_text(code,encoding='utf-8')
if __name__=='__main__':main()
