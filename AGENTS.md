# Development instructions

- Treat `cpp/` and `alphazero/cpp/` rule implementations as the rules oracle; rule-facing changes require conformance tests.
- Keep model, optimizer, scheduler, counters and RNG together in resumable checkpoints.
- Store policy targets as legal action IDs plus sparse visit counts.
- Never train, tune or promote against final/hidden layout corpora.
- Do not commit models, replay, private run state, machine-local launch records, native binaries or generated layouts.
- `alphazero/config/experiments/` and older tools are historical source records; their old checkpoint and SHA references are not available in this distribution.
- The canonical build and artifact-independent checks are `python tools/build_native.py` and `python tools/verify_source.py`.
