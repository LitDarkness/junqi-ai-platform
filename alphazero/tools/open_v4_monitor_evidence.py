"""Independently reconstruct saved dev-only teacher monitors on CPU."""
from collections import Counter,defaultdict
import hashlib,json
import numpy as np
from junqi_open_v4.arena import paired_interval
from evaluate_open_v4_longitudinal import verify_native

def reconstruct_monitor(arena,selected,plan,root):
    checks={}
    def check(k,ok):
        checks[k]=bool(ok)
        if not ok:raise ValueError('Teacher monitor: '+k)
    check('dev16only',arena['role']=='dev' and len(arena['layout_families'])==16 and arena['seed']==plan['evaluation_seed'])
    check('suite',arena['suite_digest']==hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest())
    check('IDs',len({g['id'] for g in arena['games']})==len(arena['games']))
    maps,summaries={},[]
    for i,saved in enumerate(arena['summaries']):
        games=[g for g in arena['games'] if g['matchup']==i]
        check(f'm{i}:starts_colors',Counter((g['position'],g['candidate_side']) for g in games)==Counter((j,s) for j in range(len(selected)) for s in (-1,1)))
        check(f'm{i}:identity_score',all((g['candidate'],g['opponent'])==(saved['candidate'],saved['opponent']) and g['family']==selected[g['position']]['family'] and g['score']==.5+.5*g['value'] for g in games))
        family,starts=defaultdict(list),defaultdict(list)
        for g in games:family[g['family']].append(g['score']);starts[g['start_ply']].append(g['score'])
        maps[i]={k:float(np.mean(v)) for k,v in family.items()}
        summary=dict(candidate=saved['candidate'],opponent=saved['opponent'],games=len(games),wins=sum(g['value']>0 for g in games),draws=sum(g['value']==0 for g in games),losses=sum(g['value']<0 for g in games),score=float(np.mean([g['score'] for g in games])),family_score=paired_interval(list(maps[i].values()),arena['seed']^i),score_by_start_ply={str(k):float(np.mean(v)) for k,v in starts.items()},repetition_fraction=float(np.mean([g['reason']=='REPETITION' for g in games])))
        check(f'm{i}:summary',summary==saved);summaries.append(summary)
    pairs=[]
    for i,a in enumerate(summaries):
        for j in range(i+1,len(summaries)):
            if a['opponent']==summaries[j]['opponent']:
                ids=sorted(maps[i].keys()&maps[j].keys())
                pairs.append(dict(from_matchup=i,to_matchup=j,delta=paired_interval([maps[j][k]-maps[i][k] for k in ids],arena['seed']^(i<<8)^j)))
    check('all_pairs',pairs==arena['paired_deltas'])
    native=verify_native(arena,selected,lambda:None)
    check('native_terminal',native==arena['native_verification'])
    check('teacher_identity',arena['teacher_runtime_identity']['all_workers_verified'] and Counter(x['binary_sha256'] for x in arena['teacher_runtime_identity']['verified_workers'])==Counter({t['binary_sha256']:6 for t in plan['teachers']}))
    for k,e in arena['checkpoints'].items():check('model:'+k,hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256'])
    for k,m in arena['search_metrics'].items():check('256_no_cache_reuse:'+k,m['simulations']==256*m['moves'] and all(m[t]==0 for t in ('cache_hits','deduplicated_rows','reused_roots','inherited_root_visits')))
    check('monitor_only',arena['review']['monitor_only'] and not arena['review']['promotion'] and arena['review']['new_training_games']==arena['review']['effective_updates']==0)
    return dict(all_checks_passed=True,checks_count=len(checks),checks=checks,monitor_statistics_independently_recomputed=True,monitor_only=True,promotion=False,native_verification=native,summaries=summaries,paired_deltas=pairs,source_repeat=arena['source_repeat_diagnostics'],new_training_games=0,effective_updates=0,interpretation='Family-cluster95bootstrap, related teachers and repeated development; not a formal adoption gate or final evidence.')
