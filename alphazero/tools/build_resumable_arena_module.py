"""Generate a new arena implementation, preserving the frozen scheduler/statistics."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=(ROOT/'python/junqi_open_v4/arena.py').read_text(encoding='utf-8')
    source=source.replace('def run_arena(', 'def run_resumable_arena(')
    source=source.replace('*, seed: int, progress: bool = True)',
        '*, seed: int, progress: bool = True, checkpoint_directory=None, binding=None, check_stop=None, checkpoint_rounds=20)')
    source=source.replace('    metrics = {name: defaultdict(float) for name in agents}', '''    journal = ArenaJournal(checkpoint_directory, suite, matchups, budgets, seed, binding)
    active, completed, restored = journal.restore(active)
    metrics = {name: defaultdict(float, restored.get('metrics', {}).get(name, {})) for name in agents}''')
    source=source.replace('    rounds = 0\n    while active:', '''    rounds = restored.get('rounds', 0)
    prior_elapsed = restored.get('elapsed_seconds', 0.)
    journal.save(active, completed, metrics, rounds, prior_elapsed, teachers.identity_report() if hasattr(teachers,'identity_report') else None)
    while active:
        if check_stop is not None: check_stop()''')
    source=source.replace('        rounds += 1\n', '''        rounds += 1
        if rounds % checkpoint_rounds == 0 or not active:
            journal.save(active, completed, metrics, rounds, prior_elapsed + time.perf_counter() - started, teachers.identity_report() if hasattr(teachers,'identity_report') else None)
''')
    source=source.replace('"elapsed_seconds": time.perf_counter() - started, "summaries": summaries',
        '"elapsed_seconds": prior_elapsed + time.perf_counter() - started, "summaries": summaries')
    source=source.replace('from .games import TeacherLeague', 'from .games import TeacherLeague\nfrom .arena_journal import ArenaJournal')
    (ROOT/'python/junqi_open_v4/resumable_arena.py').write_text(source,encoding='utf-8')
if __name__=='__main__': main()
