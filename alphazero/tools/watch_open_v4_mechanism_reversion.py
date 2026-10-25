"""CPU-only boundary interlock for the user's conditional mechanism reversion."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import time


def main():
    from open_v4_runtime import read_json,atomic_json,utc_now,ProcessLock,lock_held
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--contract',required=True);args=parser.parse_args()
    contract=read_json(root/args.contract);current=root/contract['current_run'];queue=root/contract['queue_directory']
    queue.mkdir(parents=True,exist_ok=True)
    def update(state,**kw):atomic_json(queue/'STATUS.json',dict(state=state,pid=os.getpid(),at=utc_now(),**kw))
    def user_stop():
        return (queue/'STOP').exists() or read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json').get('paused_by_user',False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    with ProcessLock(queue/'SUPERVISOR.lock'):
        update('waiting_next_registered_boundary',current_run=contract['current_run'])
        trigger=None
        while trigger is None:
            if user_stop() or any((root/p).exists() for p in contract['current_stop_paths']):update('cancelled_by_stop');return
            progress=read_json(current/'PROGRESS.json')
            for stage in contract['boundaries']:
                name=f'block_{stage:06d}_small';entry=progress['evaluations'].get(name)
                if not entry:continue
                arena_path=root/entry['path']
                if sha(arena_path)!=entry['sha256']:raise ValueError('committed development result changed')
                arena=read_json(arena_path)
                delta=arena['review']['paired']['mean']
                if delta<0:
                    trigger=dict(boundary_stage=stage,arena=entry,paired_delta=delta,paired95=arena['review']['paired95'],
                        source_teacher_score=arena['summaries'][1]['family_score'],candidate_teacher_score=arena['summaries'][2]['family_score'],
                        reason='User requests mechanism reversion at next negative external point result; no significance-delay requirement.')
                    break
            if trigger is None:
                status=read_json(current/'STATUS.json')
                if status['state'] in ('review_due','segment_complete','needs_attention','stopped_by_user') and not lock_held(current/'supervision/TRAINING.lock'):
                    update('no_negative_trigger',runtime_state=status['state']);return
                time.sleep(1)
        atomic_json(queue/'TRIGGER.json',dict(at=utc_now(),**trigger))
        # Preserve the old run as an archive. Never clear this marker.
        marker=current/'STOP'
        if marker.exists():update('cancelled_by_existing_stop');return
        atomic_json(marker,dict(at=utc_now(),reason='user_authorized_conditional_mechanism_reversion',contract=args.contract,trigger=trigger['arena']))
        update('stopping_old_mechanism',**trigger)
        while lock_held(current/'supervision/TRAINING.lock') or lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'):
            if user_stop():update('cancelled_by_stop');return
            time.sleep(1)
        update('awaiting_validated_reversion_entry',**trigger)
        ready=queue/'READY.json'
        while not ready.exists():
            if user_stop():update('cancelled_by_stop');return
            time.sleep(1)
        settings=read_json(ready)
        for p,h in settings['sources'].items():
            if sha(root/p)!=h:raise ValueError('reversion implementation changed: '+p)
        if (queue/'HANDOFF.json').exists():raise ValueError('one-time launch already recorded')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1')
        checked=subprocess.run([settings['python'],str(root/settings['builder']),'--contract',str(root/args.contract)],cwd=root,env=env,text=True,capture_output=True)
        atomic_json(queue/'PREPARATION_RESULT.json',dict(returncode=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
        if checked.returncode:raise RuntimeError('reversion preflight failed; no retry: '+checked.stderr[-1200:])
        launch_plan=read_json(queue/'PREPARED.json')['plan']
        if user_stop():update('cancelled_by_stop');return
        atomic_json(queue/'HANDOFF.json',dict(at=utc_now(),state='launching_once',plan=launch_plan,trigger=trigger,
            weight_reset=False,mechanism='256fresh then64updates, original Policy/Value phase rates'))
        update('launching_validated_reversion',plan=launch_plan)
        env['CUDA_VISIBLE_DEVICES']='0'
        launched=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(root/'start_open_v4_mechanism_reversion.ps1'),'-Plan',launch_plan],cwd=root,env=env,text=True,capture_output=True)
        atomic_json(queue/'LAUNCH_RESULT.json',dict(at=utc_now(),returncode=launched.returncode,stdout=launched.stdout,stderr=launched.stderr))
        if launched.returncode:raise RuntimeError('launch failed; never repeat spawn')
        update('handoff_complete',plan=launch_plan)


if __name__=='__main__':
    try:main()
    except Exception as e:
        from open_v4_runtime import atomic_json,utc_now
        import sys
        arg=sys.argv[sys.argv.index('--contract')+1]
        root=Path(__file__).resolve().parents[1]
        from open_v4_runtime import read_json
        c=read_json(root/arg)
        atomic_json(root/c['queue_directory']/'STATUS.json',dict(state='needs_attention',pid=os.getpid(),at=utc_now(),reason=repr(e)))
        raise
