# Fluxway

[中文文档](README_zh.md)

Fluxway is a distributed computation orchestration library for heterogeneous
clusters. It compiles Python operators into a graph and connects SPMD worker
groups through an asynchronous data plane, with separate resources for reading,
storage, and computation.

- Build graphs from functions, classes, and decorators; use JAX, PyTorch, DLPack,
  or native extensions inside operators.
- Place workers using resource declarations, node-affinity constraints, and
  ordered optimization goals.
- Bound concurrent execution with pipeline depth, state-access policies, phase
  dependencies, and backpressure.
- Coordinate source leases, whole-graph drain, checkpoint publication, and
  recovery into fresh executions.
- Export compiled graphs and performance traces to compare declared dependencies
  with actual execution.

The [recommendation training example](fluxway/example/recommendation/README.md)
combines readers, CPU sparse tables, and data-parallel computation. Overlap and
copy costs depend on operators, backends, and deployment; measure them for each
workload. The public entry point submits a compiled graph; operators may use
eager computation internally.

## Documentation

Start with the [documentation index](docs/README.md). Sequence diagrams for
complex flows identify concurrent calls, barriers, cancellation, and commit boundaries.

| Topic | Guides |
| --- | --- |
| Architecture and execution | [Architecture](docs/ARCHITECTURE.md), [execution lifecycle](docs/execution_lifecycle.md), [checkpoints](docs/checkpoint.md) |
| Reading data | [Source coordination](docs/source_coordination.md), [source work protocol](docs/source_spmd_formal_protocol.md), [reader API](docs/base_reader.md) |
| Delivering data | [Transfer planning](docs/transfer_planning.md), [data-plane runtime](docs/dataplane_runtime.md), [coherence protocol](docs/coherence_protocol.md) |
| Resources and operators | [Placement recipes](docs/placement_recipe.md), [placement mathematics](docs/placement_mathematics.md), [worker-process operators](docs/worker_process_operators.md) |
| Observation and diagnostics | [User profiling](docs/user_profiler.md), [graph visualization](docs/graph_visualization.md) |
| Development and releases | [Contributing](CONTRIBUTING.md), [releasing](docs/releasing.md) |

## View a graph

Install the viewer dependency:

```bash
python -m pip install "fluxway[visualization]"
```

This example needs an existing `graph` and a new output directory. Export does
not execute the graph; the viewer blocks while running a local server.

```python
import fluxway

bundle = fluxway.visualize_graph(graph, "tmp/training-graph")
fluxway.view_graph(bundle)
```

See [graph visualization](docs/graph_visualization.md) for a complete graph-building
example, remote viewing, and trace correlation.
