"""Record necessary isolated extension and original rule checks before registration."""
from pathlib import Path
import argparse,os,sys,subprocess
from open_v4_runtime import atomic_json,read_json,utc_now
from open_v4_queue_contract import sha,inside

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-manifest',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();root=Path(__file__).resolve().parents[1];native=read_json(inside(root,a.native_manifest));out=inside(root,a.output)
    if out.exists():raise ValueError('preserve proof')
    env=dict(os.environ,JUNQI_ROOT_EXPLORE_BUILD=native['directory'],PYTHONUTF8='1',
        PYTHONPATH=os.pathsep.join(str(root/p) for p in ('build','python','tools','tests')))
    commands=[[sys.executable,'-m','unittest','test_open_v4_root_explore','test_search_repairs','test_search_v3_puct','-q'],
              [sys.executable,'tests/native_conformance.py','--games','32','--plies','160','--seed','2026100503']]
    records=[]
    for cmd in commands:
        result=subprocess.run(cmd,cwd=root,env=env,text=True,encoding='utf-8',capture_output=True)
        records.append({'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    original=root/'build/JunqiAZ.cp310-win_amd64.pyd'
    if sha(original)!=native['original_pyd_sha256']:raise ValueError('original binary changed')
    for name,digest in native['identity']['inputs'].items():
        if sha(inside(root,name))!=digest:raise ValueError('native rule/search source changed')
    atomic_json(out,{'at':utc_now(),'all_passed':True,'tests':records,'native_manifest':{'path':a.native_manifest,'sha256':sha(inside(root,a.native_manifest))},
                     'original_binary_preserved':True,'new_training_games':0,'effective_updates':0,'cpu_only':True})
    print(str(out.relative_to(root)))
if __name__=='__main__':main()
