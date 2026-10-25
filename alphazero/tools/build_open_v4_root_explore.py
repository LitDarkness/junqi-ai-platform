"""Content-addressed extension build; original JunqiAZ and rule sources stay immutable."""
from pathlib import Path
import hashlib,json,subprocess,sys,sysconfig

def main():
    import pybind11,shutil
    root=Path(__file__).resolve().parents[1]
    rule_names=('junqi_board','junqi_state','setup_validator','move_generator','combat_resolver','setup_generator','junqi_environment','az_state')
    sources=[root/f'cpp/{n}.cpp' for n in rule_names]+[root/'cpp/root_explore/neural_puct.cpp',root/'cpp/root_explore/bindings.cpp']
    inputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'cpp').glob('*.h'))}
    for p in sources+[root/'cpp/root_explore/neural_puct.h']:inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    compiler=shutil.which('g++')
    if not compiler:raise RuntimeError('same g++ toolchain required; no automatic compiler substitution')
    version=subprocess.check_output([compiler,'--version'],text=True)
    identity={'inputs':inputs,'compiler':compiler,'version':version,'python':sys.version,'pybind':pybind11.__version__,
              'flags':['-O3','-DNDEBUG','-std=c++17','-shared','-static']}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory=root/'build-experiments'/digest
    suffix=sysconfig.get_config_var('EXT_SUFFIX');target=directory/('JunqiAZRootExplore'+suffix)
    if directory.exists():raise ValueError('preserve previous independent build')
    directory.mkdir(parents=True)
    original=root/'build'/('JunqiAZ'+suffix);before=hashlib.sha256(original.read_bytes()).hexdigest()
    args=[compiler,*identity['flags'],'-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(root/'cpp'),
          *map(str,sources),'-L'+str(Path(sys.base_prefix)/'libs'),f'-lpython{sys.version_info.major}{sys.version_info.minor}','-o',str(target)]
    subprocess.run(args,check=True)
    if hashlib.sha256(original.read_bytes()).hexdigest()!=before:raise ValueError('original pyd changed')
    for name,value in inputs.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=value:raise ValueError('source changed during build')
    result={'directory':str(directory.relative_to(root)),'binary':{'path':str(target.relative_to(root)),
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},'identity':identity,'original_pyd_sha256':before}
    manifest=directory/'BUILD.json';manifest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':str(manifest.relative_to(root)),**result},ensure_ascii=False))

if __name__=='__main__':main()
