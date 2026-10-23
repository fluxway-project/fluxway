Fluxway benchmarks are executable performance probes, not pytest tests.

Keep benchmarks that exercise Fluxway's runtime interfaces:

- `ray_sparse_embedding_dataplane_granularity.py` compares Ray data-plane task
  granularity while preserving same-target bundling.
- `local_dag_executor_benchmark.py` measures generic local DAG overlap without
  adding training semantics to the scheduler.
- `transport_structural_speedups.py` measures transport hot-path structure
  changes before they are promoted into runtime code.

One-off proof scripts belong in the project `tmp/` directory. Backend-specific
experiments outside Fluxway's Ray/Axon data-plane direction should not live here.
