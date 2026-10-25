"""Create reviewable isolated entry, preserving old executable sources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    original=(ROOT/'tools/run_open_v4_continuous_fixed.py').read_text(encoding='utf-8')
    source=original.replace('from junqi_open_v4.teachers import TeacherLeague','from junqi_open_v4.isolated_teachers import TeacherLeague')
    source=source.replace('open_v4_continuous_fixed_contract','open_v4_until400_contract')
    source=source.replace('junqi-continuous-fixed-mechanism-plan-1','junqi-until400-isolated-teachers-plan-1')
    source=source.replace('for stage in count(1):','for stage in range(1, plan["maximum_stages"]+1):')
    source=source.replace("if stage % plan['periodic_screen_stages'] == 0 and screen['review']['positive_signal'] else None)","if (stage % plan['periodic_screen_stages'] == 0 or stage == plan['maximum_stages']) and screen['review']['positive_signal'] else None)")
    source=source.replace('meta.update(actor_model_hash=old["model_hash"], actor_checkpoint=old,','meta.update(teacher_runtime_identity=teachers.identity_report(),\n                                        actor_model_hash=old["model_hash"], actor_checkpoint=old,')
    marker='                original_submit = teachers.submit_actions'
    source=source.replace(marker,'                atomic_json(directory/"verification/TEACHER_WORKERS.json",teachers.identity_report())\n'+marker)
    # Fixed source measurements are registered prospectively before any learning.
    marker='                for stage in range(1, plan["maximum_stages"]+1):'
    source=source.replace(marker,'''                if 'startup_baseline' not in progress['evaluations']:
                    baseline=teacher_monitor('startup_baseline',{'parent':plan['source_checkpoint'],'repeat':plan['source_checkpoint']},teachers,[0,1])
                    parent082=next(r['family_score']['mean'] for r in baseline['summaries'] if r['candidate']=='parent' and r['opponent']=='teacher0')
                    progress['segment_parent_screen_score']=parent082
                    progress['corrected_source_baseline']={r['opponent']:r['family_score'] for r in baseline['summaries'] if r['candidate']=='parent'}
                    commit()
                else:
                    baseline=read_json(inside(progress['evaluations']['startup_baseline']['path']))
                    progress['segment_parent_screen_score']=next(r['family_score']['mean'] for r in baseline['summaries'] if r['candidate']=='parent' and r['opponent']=='teacher0')
                check()
'''+marker)
    # Separate monitoring helper does not change the four-matchup small gate.
    marker='        def read_shard(item):'
    source=source.replace(marker,'''        def teacher_monitor(name,participants,teachers,teacher_ids):
            check();verify()
            if name in progress['evaluations']:
                record=progress['evaluations'][name]
                if sha(inside(record['path']))!=record['sha256']:raise ValueError('committed teacher monitor changed')
                return read_json(inside(record['path']))
            selected=select_families(suite,16,plan['evaluation_seed'])
            models={k:load_model(item) for k,item in participants.items()}
            agents={k:search(m) for k,m in models.items()}
            matchups=[(k,f'teacher{i}') for i in teacher_ids for k in models]
            update(state='arena',phase='frozen_development_evaluation',arena=name,planned_games=2*len(selected)*len(matchups),evaluation_families=16)
            arena=run_arena(selected,matchups,agents,teachers,{k:256 for k in agents},seed=plan['evaluation_seed'])
            arena['checkpoints']=participants;arena['teacher_runtime_identity']=teachers.identity_report()
            from evaluate_open_v4_longitudinal import verify_native
            arena['native_verification']=verify_native(arena,selected,check)
            repeat_diagnostics={}
            for i in teacher_ids:
                if 'parent' in participants and 'repeat' in participants:
                    parent=[g for g in arena['games'] if g['candidate']=='parent' and g['opponent']==f'teacher{i}']
                    repeat=[g for g in arena['games'] if g['candidate']=='repeat' and g['opponent']==f'teacher{i}']
                    if len(parent)!=len(repeat):raise ValueError('incomplete source duplicate monitor')
                    repeat_diagnostics[f'teacher{i}']={'games':len(parent),'identical_trace_and_result':sum((a['trace'],a['value'])==(b['trace'],b['value']) for a,b in zip(parent,repeat)),'interpretation':'Original time-limited teacher recommendations may vary; retain repeats as measurement evidence, not a strength-based training stop.'}
            arena['source_repeat_diagnostics']=repeat_diagnostics
            arena['review']={'monitor_only':True,'promotion':False,'new_training_games':0,'effective_updates':0,'actual_teacher_identities_verified':True}
            path=directory/'arena'/f'{name}.json';atomic_json(path,arena)
            progress['evaluations'][name]={'path':str(path.relative_to(root)),'sha256':sha(path),'review':arena['review']}
            commit();return arena
'''+marker)
    source=source.replace('            arena["checkpoints"] = participants','            arena["teacher_runtime_identity"] = teachers.identity_report()\n            arena["checkpoints"] = participants')
    source=source.replace('                    action = continuation_action(stage, confirmed, screen["review"]["observed_negative_external_point"])', '''                    if stage==plan['maximum_stages']:
                        teacher_monitor('iteration400_true084',{'parent':progress['segment_parent'],'candidate':participants['candidate'],'repeat':progress['segment_parent']},teachers,[1])
                    action = continuation_action(stage, confirmed, screen["review"]["observed_negative_external_point"])''')
    source=source.replace('            raise AssertionError("continuous stage iterator unexpectedly ended")','''            final=progress['variants']['learner']
            if final['counters']['iteration']!=400:raise ValueError('registered absolute400 endpoint not reached')
            atomic_json(directory/'COMPLETION.json',{'state':'horizon_complete','target_iteration':400,'latest':final,'new_training_games':len(progress['collections']['learner'])*256,'effective_updates':final['counters']['effective_updates']-source['counters']['effective_updates'],'review_required':True,'automatic_method_change':False,'historical_weights_not_replaced':True})
            update(state='horizon_complete',phase='review_due',review_required=True,reason='user_requested_iteration400_review',completed_iteration=400,next_action='Report strength and plateau; no autonomous mechanism change or unregistered extension')''')
    source=source.replace('"Restore original phase LR only; Adam moments/step and scheduler/RNG exact"','"Identical current phase LR and full Adam/scheduler/counters/RNG; teacher identities corrected only"')
    target=ROOT/'tools/run_open_v4_until400.py';assert not target.exists();target.write_text(source,encoding='utf-8')
    # The launcher has a persistent once-only receipt before any spawn.
    launch=(ROOT/'start_open_v4_continuous_fixed.ps1').read_text(encoding='utf-8').replace('run_open_v4_continuous_fixed.py','run_open_v4_until400.py').replace('continuous_fixed_launches','until400_launches').replace('continuous_fixed_$TaskStamp','until400_$TaskStamp')
    marker='$TaskCheck=& $TaskPython'
    launch=launch.replace(marker,'''$TaskOnce=Join-Path $TaskRoot ($TaskSettings.user_contract.path.Replace('AUTHORIZATION.json','SPAWN_ONCE.json'))
if((Test-Path -LiteralPath $TaskOnce) -and -not $Resume){throw 'Once-only launch already attempted; review receipt, no duplicate spawn'}
'''+marker)
    marker='$TaskProcess=Start-Process'
    launch=launch.replace(marker,'''@{state='spawn_attempt_registered';plan=$TaskPlan;receipt=$TaskReceipt;at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $TaskOnce -Encoding UTF8
'''+marker)
    (ROOT/'start_open_v4_until400.ps1').write_text(launch,encoding='utf-8')
    stop=(ROOT/'stop_open_v4_continuous_fixed.ps1').read_text(encoding='utf-8').replace('continuous_restored_mechanism','restored_mechanism_isolated_teachers_to400')
    (ROOT/'stop_open_v4_until400.ps1').write_text(stop,encoding='utf-8')
    compile(source,str(target),'exec')
    assert 'for stage in range(1, plan["maximum_stages"]+1)' in source
    assert 'from junqi_open_v4.isolated_teachers import TeacherLeague' in source
    print(str(target))
if __name__=='__main__':main()
