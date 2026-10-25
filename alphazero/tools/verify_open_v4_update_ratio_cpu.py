"""Necessary CPU contract validation; reuse byte-identical native proof."""
from pathlib import Path
import argparse,os,sys,subprocess
from open_v4_runtime import atomic_json,read_json,utc_now
from open_v4_queue_contract import inside,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=inside(root,a.output)
    if out.exists():raise ValueError('preserve proof')
    old=read_json(root/'config/experiments/open_v4_root_explore_20261004_214715.json')
    original=read_json(inside(root,old['cpu_verification']['path']));native=read_json(inside(root,old['native_build']['manifest']['path']))
    if sha(inside(root,old['cpu_verification']['path']))!=old['cpu_verification']['sha256'] or not original['all_passed']:raise ValueError('native proof changed')
    for name,digest in native['identity']['inputs'].items():
        if sha(inside(root,name))!=digest:raise ValueError('native code changed')
    if sha(inside(root,native['binary']['path']))!=native['binary']['sha256'] or sha(root/'build/JunqiAZ.cp310-win_amd64.pyd')!=native['original_pyd_sha256']:raise ValueError('native binaries changed')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONPATH=os.pathsep.join(str(root/p) for p in ('build','python','tools','tests')))
    cmd=[sys.executable,'-m','unittest','test_open_v4_update_ratio','-q'];result=subprocess.run(cmd,cwd=root,env=env,text=True,encoding='utf-8',capture_output=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    atomic_json(out,{'at':utc_now(),'all_passed':True,'cpu_only':True,'new_training_games':0,'effective_updates':0,
        'tests':[{'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}],
        'unchanged_native_proof':old['cpu_verification'],'native_manifest':old['native_build']['manifest'],
        'interpretation':'No rule/search change; reuse exact18CPU tests and5120rule steps by frozen code and binary digest, not new testing. New tests cover factor isolation, clocks/RNG, full state, sparse example accounting and strict86 gate.'})
    print(str(out.relative_to(root)))
if __name__=='__main__':main()
