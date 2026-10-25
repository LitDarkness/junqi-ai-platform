"""Reuse immutable baseline games, retaining the original arena ordering."""
from __future__ import annotations
import copy
import hashlib
import json
from collections import defaultdict
import numpy as np
from junqi_open_v4.arena import paired_interval

ROLES = ('champion', 'parent', 'control', 'candidate', 'repeat')
def matchups():
    return [('candidate', x) for x in ('champion', 'parent', 'control')] + [
        (role, f'teacher{i}') for i in range(2) for role in ROLES]

def suite_digest(suite):
    return hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()

def key(row):
    return row['candidate'], row['opponent'], int(row['position']), int(row['candidate_side'])

def index_games(games, suite, pairs):
    allowed = set(pairs)
    indexed = {}
    for row in games:
        k = key(row)
        if k in indexed or k[:2] not in allowed or k[3] not in (1, -1):
            raise ValueError('duplicate or unexpected arena fixture')
        if not 0 <= k[2] < len(suite):
            raise ValueError('invalid suite position')
        descriptor = suite[k[2]]
        if row['family'] != descriptor['family'] or row['start_ply'] != descriptor['start_ply']:
            raise ValueError('fixture family/history changed')
        if row['value'] not in (-1, 0, 1) or row['score'] != .5 + .5 * row['value']:
            raise ValueError('invalid actual-terminal score')
        if not row['trace']:
            raise ValueError('missing completed game trace')
        indexed[k] = row
    expected = {(a, b, position, side) for a,b in pairs for position in range(len(suite)) for side in (1,-1)}
    if set(indexed) != expected:
        raise ValueError('missing or extra completed arena fixtures')
    return indexed

def compare_repeat(cached, fresh, suite):
    old_pairs = [('parent', f'teacher{i}') for i in range(2)]
    new_pairs = [('repeat', f'teacher{i}') for i in range(2)]
    old = index_games([g for g in cached['games'] if (g['candidate'],g['opponent']) in old_pairs], suite, old_pairs)
    new = index_games(fresh['games'], suite, new_pairs)
    differences = []
    for k, row in new.items():
        reference = old[('parent', *k[1:])]
        changed = [name for name in ('trace','reason','value','score','family','start_ply') if row[name] != reference[name]]
        if changed:
            differences.append({'fixture': list(k), 'fields': changed})
    return {'games':len(new),'exact_matches':len(new)-len(differences),
            'all_exact':not differences,'first_differences':differences[:24]}

def assemble(cached, fresh_parts, suite, seed, *, reuse):
    pairs = matchups()
    cached_pairs = [(r, f'teacher{i}') for i in range(2) for r in ('champion','parent')]
    records = []
    if reuse:
        selected = [g for g in cached['games'] if (g['candidate'],g['opponent']) in cached_pairs]
        index_games(selected, suite, cached_pairs)
        records.extend(dict(copy.deepcopy(g), execution_origin='cached_baseline') for g in selected)
    for part in fresh_parts:
        records.extend(dict(copy.deepcopy(g), execution_origin='fresh_completion') for g in part['games'])
    indexed = index_games(records, suite, pairs)
    for row in indexed.values():
        matchup = pairs.index((row['candidate'], row['opponent']))
        row['matchup'] = matchup
        row['id'] = row['position'] * len(pairs) * 2 + matchup * 2 + (0 if row['candidate_side']==1 else 1)
    # run_arena completes in ply order, then its canonical active-fixture order.
    records = sorted(indexed.values(), key=lambda r:(len(r['trace']),r['id']))
    summaries, scores = [], {}
    for matchup, names in enumerate(pairs):
        games = [r for r in records if r['matchup']==matchup]
        families, starts = defaultdict(list), defaultdict(list)
        for row in games:
            families[row['family']].append(row['score']);starts[row['start_ply']].append(row['score'])
        scores[matchup] = {f:float(np.mean(rows)) for f,rows in families.items()}
        summaries.append({'candidate':names[0],'opponent':names[1],'games':len(games),
            'wins':sum(r['value']>0 for r in games),'draws':sum(r['value']==0 for r in games),'losses':sum(r['value']<0 for r in games),
            'score':float(np.mean([r['score'] for r in games])),
            'family_score':paired_interval(list(scores[matchup].values()),seed ^ matchup),
            'score_by_start_ply':{str(k):float(np.mean(v)) for k,v in starts.items()},
            'repetition_fraction':float(np.mean([r['reason']=='REPETITION' for r in games]))})
    deltas=[]
    for first in range(len(pairs)):
        for second in range(first+1,len(pairs)):
            if pairs[first][1]!=pairs[second][1]:continue
            common=sorted(scores[first].keys() & scores[second].keys())
            deltas.append({'from_matchup':first,'to_matchup':second,
                'delta':paired_interval([scores[second][f]-scores[first][f] for f in common],seed ^ (first<<8) ^ second)})
    return {'schema':'junqi-development-arena-v4','role':'dev','seed':seed,'suite_digest':suite_digest(suite),
            'layout_families':sorted({r['family'] for r in suite}),'summaries':summaries,'paired_deltas':deltas,
            'games':records,'fresh_games':sum(r['execution_origin']=='fresh_completion' for r in records),
            'reused_games':sum(r['execution_origin']=='cached_baseline' for r in records)}
