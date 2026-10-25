"""Atomic arena round journal with immutable completed-game chunks.

Never reuse results across a changed model, source, layout, budget or seed.
Replay active trajectories against the unchanged native rules before resuming.
"""
import hashlib
import json
from pathlib import Path
from junqi_search_v3.dataset import atomic_json

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class ArenaJournal:
    def __init__(self,directory,suite,matchups,budgets,seed,binding):
        self.directory=Path(directory) if directory is not None else None
        self.contract=dict(schema='arena-round-journal-1',suite=suite,matchups=matchups,budgets=budgets,seed=seed,binding=binding)
        self.contract=json.loads(json.dumps(self.contract))
        self.written=0; self.chunks=[]; self.generation=0

    def restore(self,initial):
        if self.directory is None: return initial,[],{}
        self.directory.mkdir(parents=True,exist_ok=True)
        manifest=self.directory/'MANIFEST.json'
        if not manifest.exists(): return initial,[],{}
        record=json.loads(manifest.read_text(encoding='utf-8'))
        if record['contract']!=self.contract: raise ValueError('arena journal contract mismatch')
        def read(entry):
            path=self.directory/entry['path']
            if path.parent!=self.directory or file_digest(path)!=entry['sha256']:
                raise ValueError('arena journal checksum/path mismatch')
            return json.loads(path.read_text(encoding='utf-8'))
        completed=[]
        for entry in record['chunks']: completed.extend(read(entry)['games'])
        snapshot=read(record['snapshot']); byid={g['id']:g for g in initial}
        active=[]; ids=[]
        for row in completed:
            expected=byid.get(row['id'])
            if expected is None or any(row[k]!=expected[k] for k in ('position','family','start_ply','matchup','candidate','opponent','candidate_side')):
                raise ValueError('completed arena game identity mismatch')
            # Complete terminal traces are rechecked by the final native verifier.
            ids.append(row['id'])
        for saved in snapshot['active']:
            game=byid.get(saved['id'])
            if game is None: raise ValueError('unknown active arena id')
            for action in saved['trace']:
                if action not in game['state'].legal_actions(): raise ValueError('active arena illegal replay')
                game['state'].apply(int(action))
            if game['state'].is_terminal() or int(game['state'].native_state.key())!=saved['state_key']:
                raise ValueError('active arena state/history mismatch')
            game.update(trace=saved['trace'],candidate_seconds=saved['candidate_seconds'],opponent_seconds=saved['opponent_seconds'])
            active.append(game); ids.append(game['id'])
        if len(ids)!=len(set(ids)) or set(ids)!=set(byid): raise ValueError('missing or duplicated arena games')
        self.written=len(completed);self.chunks=record['chunks'];self.generation=record['generation']
        return active,completed,snapshot

    def save(self,active,completed,metrics,rounds,elapsed,teacher_identity=None):
        if self.directory is None: return
        self.generation+=1
        # Unique immutable file names also retain unreferenced crash remnants.
        import uuid
        suffix=f'{self.generation:06d}_{uuid.uuid4().hex}'
        def write(name,payload):
            path=self.directory/name;atomic_json(path,payload)
            return dict(path=name,sha256=file_digest(path))
        new=completed[self.written:]
        chunks=list(self.chunks)
        if new: chunks.append(write('games_'+suffix+'.json',dict(games=new)))
        rows=[dict(id=g['id'],trace=g['trace'],state_key=int(g['state'].native_state.key()),
              candidate_seconds=g['candidate_seconds'],opponent_seconds=g['opponent_seconds']) for g in active]
        snapshot=write('active_'+suffix+'.json',dict(active=rows,rounds=rounds,elapsed_seconds=elapsed,
            teacher_runtime_identity=teacher_identity,metrics={k:dict(v) for k,v in metrics.items()}))
        atomic_json(self.directory/'MANIFEST.json',dict(contract=self.contract,chunks=chunks,snapshot=snapshot,
            generation=self.generation,completed_games=len(completed),active_games=len(active),rounds=rounds,
            resume_replays_at_most_rounds=20))
        self.chunks=chunks; self.written=len(completed)
