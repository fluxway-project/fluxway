# Fluxway Documentation

[Chinese edition](README_zh.md)

Start with the architecture, then follow the guide for the mechanism you need.
Each guide pairs a short explanation with code, expected results, and limits.

## Run the examples

Use Linux and Python 3.12 or 3.13. Install Fluxway in your environment:

```bash
python -m pip install fluxway
```

Examples that only build graphs, configure policies, or exercise local protocols
do not start a cluster. Functions that accept a compiled graph or runtime session
are integration examples: their surrounding text states who supplies those values.
Initialize Ray before starting execution. All participating nodes need compatible
dependencies; recovery also requires a checkpoint repository they can access.
Individual guides name optional dependencies and device requirements.

## Guides

| Topic | English | Chinese |
| --- | --- | --- |
| Architecture and lifecycle | [ARCHITECTURE.md](ARCHITECTURE.md) | [ARCHITECTURE_zh.md](ARCHITECTURE_zh.md) |
| Execution lifecycle | [execution_lifecycle.md](execution_lifecycle.md) | [execution_lifecycle_zh.md](execution_lifecycle_zh.md) |
| Source coordination | [source_coordination.md](source_coordination.md) | [source_coordination_zh.md](source_coordination_zh.md) |
| Reader API | [base_reader.md](base_reader.md) | [base_reader_zh.md](base_reader_zh.md) |
| Checkpoint and recovery | [checkpoint.md](checkpoint.md) | [checkpoint_zh.md](checkpoint_zh.md) |
| Source work protocol | [source_spmd_formal_protocol.md](source_spmd_formal_protocol.md) | [source_spmd_formal_protocol_zh.md](source_spmd_formal_protocol_zh.md) |
| Transfer planning | [transfer_planning.md](transfer_planning.md) | [transfer_planning_zh.md](transfer_planning_zh.md) |
| Data-plane runtime | [dataplane_runtime.md](dataplane_runtime.md) | [dataplane_runtime_zh.md](dataplane_runtime_zh.md) |
| Coherence protocol | [coherence_protocol.md](coherence_protocol.md) | [coherence_protocol_zh.md](coherence_protocol_zh.md) |
| Placement | [placement_recipe.md](placement_recipe.md) | [placement_recipe_zh.md](placement_recipe_zh.md) |
| Operators and tooling | [worker_process_operators.md](worker_process_operators.md) | [worker_process_operators_zh.md](worker_process_operators_zh.md) |
| Placement mathematics | [placement_mathematics.md](placement_mathematics.md) | [placement_mathematics_zh.md](placement_mathematics_zh.md) |
| Graph visualization | [graph_visualization.md](graph_visualization.md) | [graph_visualization_zh.md](graph_visualization_zh.md) |
| User profiler | [user_profiler.md](user_profiler.md) | [user_profiler_zh.md](user_profiler_zh.md) |
| Release process | [releasing.md](releasing.md) | [releasing_zh.md](releasing_zh.md) |

Read [execution lifecycle](execution_lifecycle.md) to start, observe, and stop a
graph. Read [checkpoints](checkpoint.md) before adding recovery. Source adapters
should also follow [source coordination](source_coordination.md) and the
[source work protocol](source_spmd_formal_protocol.md).

## Documentation scope

These guides describe Fluxway's public APIs and protocol boundaries. Internal
integration examples are labeled explicitly. Local examples illustrate behavior;
they do not establish cluster throughput, crash recovery, or model convergence.
Application configuration, remote catalogs, credentials, and acceptance results
belong in the application repository. English and Chinese companions cover the
same mechanisms and are maintained together.

## Reading sequence diagrams

Diagrams use Mermaid. Solid arrows show calls or message submission; dashed
arrows show returns or completion notifications. Message labels and barriers
state whether an asynchronous call is awaited. `par` branches may progress
concurrently, `alt` selects mutually exclusive paths, and `opt` marks an optional
path. Vertical order expresses required ordering, not elapsed time. One participant
may represent a group of concurrent actors.

| Ordering to inspect | Guide |
| --- | --- |
| Startup, restore, admission, drain, and cancellation | [Execution lifecycle](execution_lifecycle.md) |
| Scanning, compilation, window installation, and shutdown | [Source coordination](source_coordination.md) |
| Batched acquisition, out-of-order ACKs, and contiguous progress | [Source work protocol](source_spmd_formal_protocol.md) |
| Asynchronous reads and complete exhaustion | [Reader API](base_reader.md) |
| Snapshot preparation, commit, crash reconciliation, and read protection | [Checkpoints](checkpoint.md) |
| Plan handoff, fanout delivery, and commit barriers | [Transfer planning](transfer_planning.md), [data-plane runtime](dataplane_runtime.md) |
| Share returns, exclusive ownership, ordered retirement, and input release | [Coherence protocol](coherence_protocol.md) |
| Placement readiness, thread execution, and trace export | [Placement recipes](placement_recipe.md), [worker-process operators](worker_process_operators.md), [user profiling](user_profiler.md) |
| CI, package publishing, and release creation | [Releasing](releasing.md) |
