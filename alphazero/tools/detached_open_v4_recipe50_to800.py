"""A single WMI-owned fresh trainer, independent of desktop process ancestry."""
from pathlib import Path
import os, sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    import hashlib, runpy, traceback
    os.chdir(ROOT)
    for path in (ROOT / 'build', ROOT / 'python', ROOT / 'tools'):
        sys.path.insert(0, str(path))
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'build'), str(ROOT / 'python'), str(ROOT / 'tools')])
    from open_v4_runtime import read_json, atomic_json, utc_now, ProcessLock, lock_held
    from open_v4_recipe50_to800_contract import verify_single_prerequisite
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    reg = read_json(ROOT / 'reports/recipe50_to800_latest_registration.json')
    planpath = ROOT / reg['plan']; plan = read_json(planpath)
    out = ROOT / reg['report']
    masters = [ROOT / f'runs/{name}/supervision/PLAN.json' for name in ('open_junqi_v4_repair_pilot', 'open_junqi_v4_fresh_continuation')]
    master = read_json(masters[0])
    if master['paused_by_user'] or master['optimization']['config'] != reg['plan'] or master['optimization']['config_sha256'] != sha(planpath):
        raise ValueError('Latest human-authorized plan must match this fresh entry')
    if (ROOT / plan['directory']).exists() or any((ROOT / p).exists() for p in plan['stop_paths']):
        raise ValueError('Fresh directory required; STOPs and completed runs are never cleared')
    if lock_held(ROOT / 'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'):
        raise ValueError('Existing owner has priority')
    verify_single_prerequisite(ROOT, plan)
    proof = read_json(out / 'CHECK_ONLY.json')
    if not proof['valid'] or proof['plan_sha256'] != sha(planpath):
        raise ValueError('Exact frozen entry prelaunch verification required')
    with ProcessLock(out / 'SPAWN_ONCE.lock'):
        if (out / 'SPAWN_ONCE.json').exists():
            raise ValueError('SPAWN_ONCE: no repeat launch without explicit fault audit')
        receipt = dict(at=utc_now(), pid=os.getpid(), parent_pid=os.getppid(), plan=reg['plan'], plan_sha256=sha(planpath),
            directory=plan['directory'], source_iteration=600, target_iteration=800, source=plan['source_checkpoint'],
            detached_launch_method='Windows WMI Win32_Process.Create; service-owned process ancestry', resume=False,
            stdout=str((out / 'trainer.stdout.log').relative_to(ROOT)), stderr=str((out / 'trainer.stderr.log').relative_to(ROOT)),
            single_owner_same_process_continuation=True, no_cpu_waiting_controller=True)
        atomic_json(out / 'SPAWN_ONCE.json', receipt)
        atomic_json(out / 'LAUNCH.json', receipt)
        for filename in masters:
            p = read_json(filename)
            if p['paused_by_user'] or p['optimization']['config'] != reg['plan']:
                raise ValueError('Authorization changed before dispatch')
            p['optimization'].update(pid=os.getpid(), parent_pid=os.getppid(), stdout=receipt['stdout'], stderr=receipt['stderr'],
                launch_receipt=dict(path=str((out / 'LAUNCH.json').relative_to(ROOT)), sha256=sha(out / 'LAUNCH.json')),
                detached_launch_method=receipt['detached_launch_method'], training_active=True, state='launching', phase='launching')
            p['auto_next_stage'].update(pid=os.getpid(), training_active=True, waiting_controller_active=False)
            atomic_json(filename, p)
        sys.stdout = open(out / 'trainer.stdout.log', 'a', encoding='utf-8', buffering=1)
        sys.stderr = open(out / 'trainer.stderr.log', 'a', encoding='utf-8', buffering=1)
        sys.stdin = open(os.devnull, 'r')
        print('Sole detached trainer dispatch ' + str(receipt), flush=True)
        sys.argv = ['run_open_v4_recipe50_to800.py', '--plan', reg['plan']]
        try:
            runpy.run_path(str(ROOT / 'tools/run_open_v4_recipe50_to800.py'), run_name='__main__')
        except BaseException:
            traceback.print_exc()
            atomic_json(out / 'EXIT.json', dict(at=utc_now(), pid=os.getpid(), success=False, explicit_fault_review_required_before_any_recovery=True))
            raise
        else:
            atomic_json(out / 'EXIT.json', dict(at=utc_now(), pid=os.getpid(), success=True))

if __name__ == '__main__': main()
