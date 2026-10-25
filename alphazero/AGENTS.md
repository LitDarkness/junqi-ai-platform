# AlphaZero development contract

This directory is an isolated AlphaZero implementation for the Junqi platform.

## Scope and invariants

- Do not edit files outside `alphazero/` while working on this implementation.
- The copied C++ rules are an oracle. Search and state-management code may be
  optimized, but move legality, combat, terminal, and draw rules must not drift.
- Any rule-facing C++ change must pass the conformance and snapshot tests.
- Initial layouts and evaluations must be reproducible from an explicit seed.
- Never select or promote a model using the hidden/final evaluation corpus.
- Store policy targets sparsely as legal action ids plus visit counts. Do not
  persist dense 3600-element targets in the replay buffer.
- Checkpoints must include model, optimizer, scheduler, counters, configuration,
  and random-number-generator state so training can resume exactly.

## Performance principles

- MCTS tree traversal and state transitions belong in C++; Python coordinates
  batched neural-network inference and training.
- Batch leaf evaluations across independent games. Avoid one GPU call per tree.
- Report search simulations/s, self-play moves/s, inference batch size, replay
  size, and GPU memory in every benchmark.
- The default laptop profile uses 48-64 simulations per move. Raising this must
  be justified by an arena improvement, not by convention.

## Generated files

Do not commit or package `build/`, `runs/`, replay shards, TensorBoard logs,
temporary exported GUI weights, `__pycache__/`, or compiled Python extensions.

