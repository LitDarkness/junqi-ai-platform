"""CPU waiting controller: verified arena completion triggers exactly one online job."""
from pathlib import Path
import argparse
import os
import json
import subprocess
import time
from contextlib import ExitStack
from open_v4_queue_contract import inside,sha,queue_decision


def preflight(root,path):
    from open_v4_runtime import read_json
    queue=read_json(path)
    if queue['schema']!='junqi-immediate-queue-plan-1' or queue['poll_seconds']!=5:raise ValueError('queue contract changed')
    if sha(inside(root,queue['completion_plan']['path']))!=queue['completion_plan']['sha256']:raise ValueError('completion contract changed')
    for name,digest in queue['source'].items():
        if sha(inside(root,name))!=digest:raise ValueError('queue source changed: '+name)
    import torch
    from fork_open_v4_qualified import fork_payload
    from run_open_v4_search_training import verify_learning_state
    from junqi_open_v4.queued_config import QueuedConfig
    for route,entry in queue['choices'].items():
        if sha(inside(root,entry['path']))!=entry['sha256']:raise ValueError('queued experiment changed')
        plan=read_json(inside(root,entry['path']));base=QueuedConfig.load(inside(root,plan['parent_config']))
        source=plan['source_checkpoint']
        if sha(inside(root,source['path']))!=source['sha256']:raise ValueError('queued source checkpoint changed')
        saved=torch.load(inside(root,source['path']),map_location='cpu',weights_only=False)
        if saved['counters']!=plan['source_counters'] or saved['config_hash']!=plan['source_config_hash'] or saved['model_hash']!=source['model_hash']:
            raise ValueError('queued full learning state identity mismatch')
        expected=['candidate'] if route=='qualified_continuation' else ['control','candidate']
        if plan['training_variants']!=expected or plan['maximum_stages']!=24 or plan['stages_per_block']!=12:
            raise ValueError('registered horizon/branches changed')
        if base.simulations!=64 or base.repeated_simulations!=128 or base.training_greedy_after_ply!=24 or base.root_tactical_guard:
            raise ValueError('registered baseline behavior changed')
        for variant in expected:
            import copy
            cfg=copy.deepcopy(base);cfg.run_directory=plan['directory']+'/'+variant;cfg.initial_checkpoint=source['path']
            cfg.root_noise_until_ply=plan[variant]['root_noise_until_ply'];cfg.validate()
            if cfg.root_noise_until_ply!=(24 if route=='late_noise_factor' and variant=='candidate' else None):raise ValueError('noise factor changed')
            migrated=fork_payload(saved,cfg,{'queue':str(path),'route':route,'variant':variant})
            verify_learning_state(saved,migrated)
    return queue


def open_process_guard(status):
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes
    from datetime import datetime,timezone
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    handle=api.OpenProcess(0x100000|0x1000,False,status['pid'])
    if not handle:
        if ctypes.get_last_error()==87:return None
        raise OSError('cannot verify original process exit')
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];api.WaitForSingleObject.restype=wintypes.DWORD
    api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    api.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    times=[wintypes.FILETIME() for _ in range(4)]
    if not api.GetProcessTimes(handle,*(ctypes.byref(t) for t in times)):api.CloseHandle(handle);raise OSError('cannot verify process creation time')
    creation=((times[0].dwHighDateTime<<32)+times[0].dwLowDateTime)/10_000_000-11644473600
    expected=datetime.fromisoformat(status['started_at']).astimezone(timezone.utc).timestamp()
    if abs(creation-expected)>120:api.CloseHandle(handle);raise ValueError('PID identity does not match registered job')
    return api,handle


def verify_process_exit(guard):
    if guard is None:return
    import ctypes
    from ctypes import wintypes
    api,handle=guard;code=wintypes.DWORD()
    if api.WaitForSingleObject(handle,0)!=0 or not api.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value!=0:
        raise RuntimeError('original review did not exit normally; no automatic launch')


def request_stop(path):
    """Create a new request without overwriting existing user evidence."""
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x',encoding='utf-8') as handle:
            from open_v4_runtime import utc_now
            json.dump({'at':utc_now(),'cause':'user stopped immediate workflow'},handle)
            handle.flush();os.fsync(handle.fileno())
    except FileExistsError:pass


def main():
    from open_v4_runtime import read_json,atomic_json,utc_now,ProcessLock,lock_held,supervision_status
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--check-only',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];path=inside(root,args.plan);queue=read_json(path);directory=inside(root,queue['directory'])
    if args.status:print(json.dumps(supervision_status(directory),ensure_ascii=False));return
    queue=preflight(root,path)
    if args.check_only:print(json.dumps({'valid':True,'cpu_only':True,'complete_learning_states_preserved':True,'one_launch_only':True,'poll_seconds':5}));return
    if directory.exists():raise ValueError('queue already exists; inspect recorded launch instead of restarting')
    directory.mkdir(parents=True)
    primary=root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json';completion=read_json(inside(root,queue['completion_plan']['path']))
    old=inside(root,completion['directory']);original_status=read_json(old/'STATUS.json');guard=open_process_guard(original_status)
    state={'state':'waiting_for_completion','phase':'cpu_waiting_controller','pid':os.getpid(),'started_at':utc_now(),'new_training_games':0,'completed_updates':0,
        'plan':str(path.relative_to(root)),'waiting_for_run':completion['directory'],'poll_seconds':5,'gpu_job_started':False}
    allowed={str(Path(queue['completion_plan']['path']))};child=None;child_plan=None
    def update(**values):
        state.update(values,updated_at=utc_now());atomic_json(directory/'supervision/STATUS.json',state);atomic_json(directory/'STATUS.json',state)
    def stopped():
        return any(inside(root,p).exists() for p in queue['stop_paths']) or read_json(primary).get('paused_by_user',False)
    def check():
        if stopped():raise InterruptedError('User STOP/pause requested; no automatic restart')
        if str(Path(read_json(primary)['optimization']['config'])) not in allowed:raise ValueError('latest PLAN routed to another experiment')
    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    def checked_call(command,label):
        check();out=directory/(label+'.out.log');err=directory/(label+'.err.log')
        with out.open('w',encoding='utf-8') as stdout,err.open('w',encoding='utf-8') as stderr:
            result=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr,creationflags=creationflags)
        if result.returncode:raise RuntimeError(label+' failed; inspect '+str(err.relative_to(root)))
        check()
    with ExitStack() as locks:
        locks.enter_context(ProcessLock(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'))
        locks.enter_context(ProcessLock(directory/'supervision/SUPERVISOR.lock'))
        if guard:locks.callback(guard[0].CloseHandle,guard[1])
        atomic_json(directory/'CONTRACT.json',{'plan_sha256':sha(path),'source':queue['source']});update()
        print('Immediate handoff armed; CPU-only waiting, no second GPU job.',flush=True)
        try:
            while True:
                check();status=read_json(old/'STATUS.json')
                held=any(lock_held(inside(root,run)/f'supervision/{kind}.lock') for run in completion['lock_directories']+[completion['directory']] for kind in ('SUPERVISOR','TRAINING'))
                running=guard is not None and guard[0].WaitForSingleObject(guard[1],0)==258
                decision=queue_decision(status['state'],stopped=False,locked=held or running,complete=(old/'RESULTS.json').exists())
                if decision=='fail':raise RuntimeError('Review failed or process disappeared without complete result')
                if decision=='stop':raise InterruptedError('Previous review stopped by user')
                if decision=='launch':break
                update(waiting_arena=status.get('arena'),waiting_last_update=status.get('updated_at'),completed_fresh_development_games=status.get('committed_fresh_games',0))
                time.sleep(5)
            check();verify_process_exit(guard);update(state='verifying_complete_review',phase='saved_result_verification')
            from open_v4_queue_contract import validate_finished_arena
            proof=validate_finished_arena(root,queue['completion_plan'],native=True)
            atomic_json(directory/'COMPLETION_REVIEW.json',proof)
            route='qualified_continuation' if proof['review']['route_confirmed'] else 'late_noise_factor'
            entry=queue['choices'][route];child_plan=read_json(inside(root,entry['path']))
            if sha(inside(root,entry['path']))!=entry['sha256']:raise ValueError('registered next route changed')
            if not proof['review']['route_confirmed']:
                update(state='boundary_audit',phase='saved_training_diagnostic',selected_route=route)
                with ExitStack() as audit_locks:
                    audit_locks.enter_context(ProcessLock(directory/'supervision/TRAINING.lock'))
                    for run in completion['lock_directories']+[completion['directory']]:
                        for kind in ('SUPERVISOR','TRAINING'):audit_locks.enter_context(ProcessLock(inside(root,run)/f'supervision/{kind}.lock'))
                    checked_call([os.sys.executable,str(root/'tools/audit_open_v4_queue_boundary.py'),'--queue',str(path)],'boundary_audit')
                    from open_v4_queue_contract import verify_boundary_audit
                    verify_boundary_audit(root,queue)
            check();update(state='prelaunch_verification',phase='cpu_preflight',selected_route=route)
            checked_call([os.sys.executable,str(root/'tools/run_open_v4_queued_training.py'),'--plan',str(inside(root,entry['path'])),'--check-only'],'training_preflight')
            check()
            if inside(root,child_plan['directory']).exists():raise ValueError('next run already exists; no duplicate launch')
            main_plan=read_json(primary);main_plan.setdefault('optimization_history',[]).append(main_plan['optimization'])
            main_plan['optimization']={'type':route,'state':'launching','config':entry['path'],'run_directory':child_plan['directory'],
                'status_tool':'tools/run_open_v4_queued_training.py','source':child_plan['source_checkpoint'],'stop_paths':child_plan['stop_paths'],
                'phase':'online_training','source_iteration':62,'registered_horizon':{'stages':24,'new_games':child_plan['registered_new_games'],'updates':child_plan['registered_updates']},
                'launch_owner':str(path.relative_to(root)),'source_route_qualified':proof['review']['route_confirmed']}
            main_plan['next_action']='Immediate controller owns this preregistered24stage online job. No manual second launch while NEXT_STAGE/controller or GPU locks are held. Read its STATUS and new training shards, preceding-learner hashes and saved updates.12/24 evaluation boundaries and original64 gates unchanged. At completed horizon review actual results, no blind extension.'
            main_plan['auto_next_stage'].update(state='launching',selected_route=route,completion_review=str((directory/'COMPLETION_REVIEW.json').relative_to(root)))
            allowed.add(str(Path(entry['path'])))
            for target in [primary,root/'runs/open_junqi_v4_fresh_continuation/supervision/PLAN.json']:atomic_json(target,main_plan)
            stdout_path=directory/'online_training.out.log';stderr_path=directory/'online_training.err.log'
            # Durable launching receipt precedes spawn; an interrupted controller never silently retries.
            atomic_json(directory/'HANDOFF.json',{'state':'launching','at':utc_now(),'route':route,'plan':entry,'source':child_plan['source_checkpoint']})
            with stdout_path.open('w',encoding='utf-8') as stdout,stderr_path.open('w',encoding='utf-8') as stderr:
                child=subprocess.Popen([os.sys.executable,str(root/'tools/run_open_v4_queued_training.py'),'--plan',str(inside(root,entry['path']))],cwd=root,stdout=stdout,stderr=stderr,creationflags=creationflags)
                launch={'pid':child.pid,'launched_at':utc_now(),'status':'launched','plan':entry['path'],'run':child_plan['directory'],
                    'stdout':str(stdout_path.relative_to(root)),'stderr':str(stderr_path.relative_to(root)),'route':route,'automatic_after_review':True}
                atomic_json(directory/'HANDOFF.json',launch)
                main_plan=read_json(primary);main_plan['optimization'].update(pid=child.pid,stdout=launch['stdout'],stderr=launch['stderr'],state='running')
                main_plan['auto_next_stage'].update(state='training',training_pid=child.pid)
                for target in [primary,root/'runs/open_junqi_v4_fresh_continuation/supervision/PLAN.json']:atomic_json(target,main_plan)
                update(state='training',phase='online_training',training_pid=child.pid,gpu_job_started=True,selected_route=route)
                published=False
                while child.poll() is None:
                    if stopped():
                        request_stop(inside(root,child_plan['directory'])/'supervision/STOP')
                    run=inside(root,child_plan['directory'])
                    if (run/'STATUS.json').exists():
                        status=read_json(run/'STATUS.json');update(phase=status.get('phase','online_training'),new_training_games=status.get('new_training_games',0),completed_updates=status.get('completed_updates',0),training_state=status['state'])
                        if not published:atomic_json(run/'LAUNCH.json',launch);published=True
                    time.sleep(5)
                if child.returncode:raise InterruptedError('User STOP requested') if stopped() else RuntimeError('Online process failed; inspect logs; no restart')
            update(state='segment_complete',review_required=True)
        except InterruptedError as error:update(state='stopped_by_user',reason=str(error),review_required=False)
        except BaseException as error:
            update(state='needs_attention',reason=repr(error),review_required=True);raise


if __name__=='__main__':main()
