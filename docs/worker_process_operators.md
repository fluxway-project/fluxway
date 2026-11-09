# Worker-process operators

[Chinese edition](worker_process_operators_zh.md)

A stateful operator instance belongs to one SPMD worker process. Use
`@requires_worker_process` when its constructor creates process-local handles or
needs the final worker topology, such as `jax.distributed.initialize`.
Ordinary stateful classes do not need this marker.

## Declare an operator and release its resources

This example separates a stateless preparation step from a stateful file write.
`bind_spmd` leaves the original class available for local checks and creates an
SPMD declaration for graph execution. The writer has one worker, so its output
path has one owner. That path must not already exist.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import fluxway as fw
from fluxway.common.operator_policy import AsyncPolicy, StatePolicy


@fw.requires_worker_process
class BatchWriter:
    def __init__(self, output_path: str):
        self._stream = Path(output_path).open("x", encoding="utf-8")

    def prepare(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return {"values": np.abs(values)}

    def write(self, values: np.ndarray) -> int:
        self._stream.writelines(f"{value}\n" for value in values.tolist())
        return len(values)

    def on_shutdown(self) -> None:
        self._stream.close()


replicated = {"type": "static", "axis": []}
WriterSPMD = fw.bind_spmd(
    BatchWriter,
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
    prepare={
        "input_sharding": {"values": replicated},
        "output_sharding": {"values": replicated},
        "async_policy": AsyncPolicy.pipelined(depth=2, state=StatePolicy.thread_safe()),
    },
    write={
        "input_sharding": {"values": replicated},
        "async_policy": AsyncPolicy.pipelined(depth=2, state=StatePolicy.exclusive()),
    },
)


def build_writer_graph(output_path: str) -> fw.Graph:
    builder = fw.Builder()
    builder.add_node("writer", WriterSPMD, output_path=output_path)
    builder.add_func_edge(
        "writer",
        "prepare",
        "writer",
        "write",
        tensor_map={"values": "values"},
        local_only=True,
    )
    return builder.build("batch-writer")


Path("tmp").mkdir(exist_ok=True)
with TemporaryDirectory(dir="tmp") as directory:
    output = Path(directory) / "batch.txt"
    writer = BatchWriter(str(output))
    try:
        prepared = writer.prepare(np.array([-1, 2, -3]))
        assert writer.write(prepared["values"]) == 3
    finally:
        writer.on_shutdown()
    assert output.read_text() == "1\n2\n3\n"

    graph = build_writer_graph(str(Path(directory) / "runtime.txt"))
    view = graph.require_compiled_graph().spmd("writer")
    assert view.operator("prepare").spec.async_policy.state.access == "thread_safe"
    assert view.operator("write").spec.async_policy.state.access == "exclusive"
    assert view.operator("write").spec.requires_worker_process
```

The direct call checks the file contents and resource lifetime on the CPU; it
also checks the compiled policies. It does not run Ray workers. The graph shows
the local handoff; connect a real source to `prepare` before starting a job.
`local_only=True` means a dependency within the same worker, not a remote call.

At startup Fluxway constructs one instance per owner class in each worker and
shares it across that class's operators. Constructor parameters come from the
node's static arguments, such as `output_path` above. Before `__init__`, Fluxway
also supplies `rank`, `world_size`, `local_rank`, `local_world_size`, `node_rank`,
`num_nodes`, `spmd_id`, `rank0_address`, and `ray_component_limits`.
`self.ctx` carries the current invocation's context and can vary between calls.

On graceful worker shutdown, Fluxway calls each owner's `on_shutdown()` once,
even when several operators share it. The hook may be synchronous or return an
awaitable. Shutdown attempts all owners and propagates cleanup errors. Forced
process termination cannot guarantee a hook; buffered output is not a
checkpoint or a crash-recovery guarantee. A resource registry does not close
registered objects for you; see [resource
ownership](placement_recipe.md#register-resources-created-inside-a-worker).

## Where user code runs

| Callable | Execution |
| --- | --- |
| Ordinary Python function | Submitted asynchronously to a named worker-local thread lane, from its first call |
| Ordinary function returning a coroutine object | The worker thread runs that coroutine to completion |
| Coroutine function (`async def`) | Uses `COROUTINE_ASYNC` and runs cooperatively on the actor event loop |

`@requires_worker_process` controls construction, not this choice. A coroutine
must yield while waiting; blocking work inside `async def` blocks the actor
loop. Context variables, including rank and step, follow the call into worker
threads. Exceptions propagate to the local step plan and fail the worker.

This separation lets transport reception, mailbox updates, input preparation,
outbound delivery, and scheduler wakeups progress while ordinary operators run.
CPU-bound Python still competes for the GIL on standard CPython. Native code
that releases the GIL can overlap. Free-threaded CPython can also run Python
bytecode in parallel, with the same lane and scheduling rules.

```mermaid
sequenceDiagram
    participant P as Local step plan
    participant E as UserCallableExecutor
    participant T as Named thread lane
    participant L as Worker event loop
    P->>P: Acquire resources required by state policy
    P->>E: execute(spec)
    alt Ordinary Python function
        E->>T: Submit function and context variables
        par User function runs in thread
            T->>T: fn()
            opt Returns a coroutine object
                T->>T: Run coroutine to completion in thread
            end
        and Event loop keeps progressing
            L->>L: Ingress, mailboxes, delivery, scheduling
        end
        T-->>E: Result or exception
    else Coroutine function
        E->>L: await fn()
        L->>L: Yield cooperatively at each await
        L-->>E: Result or exception
    end
    E-->>P: Return result or fail the local plan
    P->>P: Release state resources for this call
```

Thread lanes and state locks are separate boundaries. `thread_safe()` permits
overlapping calls on the same owner; `exclusive()` preserves mutual exclusion. Blocking
work inside a coroutine still blocks the event loop.

### Concurrency during shutdown

```mermaid
sequenceDiagram
    participant C as Close caller
    participant W as Shared Worker close task
    participant T as User-call runtime
    participant O as Operator owners
    participant P as Checkpoint coordinator
    C->>W: await close()
    W->>W: Stop scheduling, cancel and join plans,<br/>deliveries, background tasks
    par Settle user calls
        W->>T: await drain()
        T-->>W: Call outcomes
    and Clean up instances
        W->>O: Call on_shutdown() once per unique instance
        O-->>W: Return synchronously or await async hook
    and Clean up checkpoint state
        W->>P: Roll back active attempt and close
        P-->>W: Cleanup outcome
    end
    W->>W: Close executors, invalidate tensors, fence<br/>protocol
    W-->>C: Shared result or aggregated errors
```

Synchronous `on_shutdown()` runs directly on the Worker event loop; use an async hook
and explicitly offload blocking cleanup. Owner hooks and user-call drain run
concurrently; hooks are not guaranteed to start after every thread call ends. Owners
must coordinate outstanding resource access. Cancelling a coroutine wait cannot stop a
running Python thread. Forced process termination does not guarantee hooks.

## State and scheduling policy

Concurrent operator execution in Fluxway is governed by **state mutex policies (`StatePolicy`)** and **queue scheduling policies (`SchedulePolicy`)**.

### 1. State access policy (`StatePolicy`)
* `StatePolicy.exclusive()`: **Default policy**. Serializes execution for operators sharing the same SPMD instance lock. Model parameter updates and mutable stateful stores must preserve this policy.
* `StatePolicy.thread_safe()`: Removes the instance lock. Used when the operator is completely stateless or its internal resources are thread-safe, permitting concurrent cross-step execution on the same actor.

### 2. 1-Minute SchedulePolicy selection guide

To help developers choose the appropriate policy within seconds, follow this decision tree and cheatsheet:

```mermaid
flowchart TD
    Q1{"Does the SPMD contain multiple sequential phases that mutate shared state?\n(e.g., Forward -> Backward -> Update)"}
    Q1 -- "No (90% of pipelines: Reader, ETL, stateless transforms, single-operator training)" --> R1["Use default FIFO (No explicit schedule needed!)\nOut-of-the-box + Canonical Lanes + Backpressure priority"]
    Q1 -- "Yes (Multi-phase model update in same actor)" --> Q2{"Are there complex cross-step asymmetric overlap requirements?"}
    Q2 -- "No (Standard sequential stages within step)" --> R2["SchedulePolicy.phase_order\nStrict step barrier gates prevent dirty reads"]
    Q2 -- "Yes (Fine-grained overlap optimization)" --> R3["SchedulePolicy.phase_graph\nExplicitly defines cross-step causal DAG"]
    
    Q3{"Does the pipeline require straggler prevention / staleness escalation?"} -->|Yes| R4["SchedulePolicy.deadline_boost\nDynamically escalates priority when lag exceeds threshold"]
```

#### Policy Cheatsheet & Code Patterns

| Policy | When to use | Framework default behavior | Core Code Pattern |
| :--- | :--- | :--- | :--- |
| **`fifo` (Default)** | **90% of use cases**: Data ingestion, feature transforms, stateless forward passes, single-operator training. | **Out-of-the-box default**. Automatically assigned to `io`/`compute` canonical lanes and paired with `("compute", "io")` cooperative priority to consume data first. | `@operator("read", async_policy=AsyncPolicy.pipelined(depth=4))` *(no schedule parameter required)* |
| **`phase_order`** | **Multi-phase model training**: When an actor contains `train_core` and `update` phases and Step $S+1$ must never read half-updated parameters from Step $S$. | Compiles a `PhaseGate` barrier. Step $S+1$ is **physically held outside the ready queue** until Step $S$ optimizer update commits. | `SchedulePolicy.phase_order(phase="fwd", phase_order=("fwd", "bwd", "opt"))` |
| **`phase_graph`** | **Complex cross-step overlap**: When helper work for Step $S+1$ (e.g. gradient pre-dispatch) can overlap with Step $S$ update, but core computation must wait. | Declares fine-grained causal DAG via `(before, after, step_delta)` triples. | `SchedulePolicy.phase_graph(phase="fwd", phase_order=..., dependencies=(("opt", "fwd", 1),))` |
| **`deadline_boost`** | **Straggler mitigation / Periodic flush**: Keeps low priority by default to yield CPU, but jumps to `boost_priority` when lag exceeds `staleness_limit`. | Automatically monitors scheduling lag and escalates queue priority when threshold is violated. | `SchedulePolicy.deadline_boost(base_priority=0, boost_priority=100)` |

#### Code Comparison

```python
# 1. Default FIFO: Minimal boilerplate, ideal for streaming pipelines
@operator("process", async_policy=AsyncPolicy.pipelined(depth=4))
def process(self, batch: Batch) -> Tensor:
    return self.transform(batch)

# 2. Phase Order: Multi-phase training with strict barrier protection
@operator(
    "train_core",
    async_policy=AsyncPolicy.pipelined(
        depth=1,
        schedule=SchedulePolicy.phase_order(phase="train", phase_order=("train", "update")),
    ),
)
def train_core(self, batch: Batch) -> Gradients:
    return self.backward(batch)

@operator(
    "update",
    async_policy=AsyncPolicy.pipelined(
        depth=1,
        schedule=SchedulePolicy.phase_order(phase="update", phase_order=("train", "update")),
    ),
)
def update(self, grads: Gradients) -> None:
    self.optimizer_step(grads)
```

With profiling enabled, `user_callable.execute` includes queue wait;
`user_callable.function_run` measures the function invocation without lane
queueing or worker context setup. `worker_process_call.thread_run` records the
worker-thread interval. Profiling observes the selected lane without dispatching
it a second time.

## Initialization errors

A class that calls `jax.distributed.initialize` without the marker is rejected
with a `Fluxway Worker Process Error`. Use the decorator instead of setting the
internal `__fluxway_process_affine__` field. The marker does not supply JAX,
create a distributed coordinator, or make mismatched worker topology valid.

## Canonical execution lanes and cooperative scheduling

In distributed high-performance computing, operators exhibit distinct computation and I/O characteristics. Mixing all operators into an undifferentiated execution queue or dynamically spawning an unbounded number of thread pools per function leads to thread thrashing, scheduler overhead, and pipeline backpressure breakdown.

During graph compilation, Fluxway automatically classifies operators into coarse-grained **Canonical Lanes** and establishes cooperative scheduling policies based on operator metadata:

```mermaid
flowchart TD
    subgraph SPMDWorker["SPMD Worker Actor"]
        direction TB
        subgraph EventLoop["AsyncIO Event Loop (COROUTINE_ASYNC)"]
            CoroutineOp["async def Operators\n(Scheduling, network transfers, IPC orchestration)"]
            Heartbeat["Node Heartbeat & Cluster Coordination"]
        end

        subgraph WorkerProcessRuntime["WorkerProcessCallRuntime (Managed Bounded Lanes)"]
            ComputeLane["'compute' Lane\n(Sync compute operators, matrix math, transformations)"]
            IOLane["'io' Lane\n(BaseReader operators, is_source=True, ingestion I/O)"]
        end

        subgraph PriorityQueue["Min-Heap Cooperative Priority Queue"]
            ComputeHigher["compute op (Higher weight: 1)"]
            IOLower["io op (Lower weight: 0)"]
            ComputeHigher -->|Consumes ready batches first| IOLower
        end
    end

    EventLoop -. "Stays fully responsive to heartbeats" .-> Heartbeat
    PriorityQueue --> ComputeLane
    PriorityQueue --> IOLane
```

### 1. Canonical lane classification rules

During graph compilation (`Builder.build`), the framework infers canonical lanes based on operator metadata (`fluxway.common.canonical_lanes`):

* **`io` lane**: Operators marked with `is_source=True`, declaring a `source` descriptor, or belonging to a class inheriting from `BaseReader` default to the `"io"` lane.
* **`compute` lane**: Regular synchronous compute operators (`def` methods) default to the `"compute"` lane and run on bounded worker thread pools where C extensions release the GIL without blocking the event loop.
* **Async coroutine operators**: `async def` operators remain unassigned to thread lanes (`lane=None`) and execute directly on the asyncio event loop for lightweight asynchronous I/O and network coordination.

### 2. Auto-paired cooperative priority and natural backpressure

When the graph compiler identifies the presence of both `compute` and `io` lanes, it automatically pairs and injects standard cooperative priority:
```python
lane_order = ("compute", "io")
```
Inside the worker scheduling engine, compute operators receive a higher scheduling priority than I/O operators (`priority(compute) > priority(io)`). This guarantees that ready batches are prioritized for downstream consumption before the reader over-allocates memory by reading ahead excessively, providing natural pipeline backpressure.

### 3. Open-Closed Principle and explicit overrides

The default canonical lane preset is an out-of-the-box best practice. Operators with bespoke scheduling requirements can explicitly declare custom lanes or orderings via `SchedulePolicy`, which the framework respects without overriding:

```python
# Example: Custom lane declaration and ordering
@operator(
    "custom_transform",
    async_policy=AsyncPolicy.pipelined(
        depth=2,
        schedule=SchedulePolicy.fifo(
            lane="custom_lane",
            lane_order=("custom_lane", "io"),
        ),
    ),
)
def custom_transform(self, batch: UserBatch) -> NodeOutput:
    ...
```

## Checkpoint extension boundary

Operators do not implement checkpoint hooks. Register a
`CheckpointComponentRegistration` through the SPMD's `checkpoint_components`.
A `SAVE` component handles `prepare_save`, `save`, `finish_save`, `commit_save`,
and `publish_save`; `rollback_save` handles failed transactions. A `RESTORE`
component handles `restore`. Object holds and cleanup reservations belong to
`OBJECT_LIFECYCLE`; physical deletion belongs to `OBJECT_CLEANUP`.
The control runtime calls `abort_save` to discard an incomplete attempt and
`acknowledge_save` after the global commit succeeds.

The controller orders transactions and publishes the manifest. Components own
physical URIs, staging layouts, versions, and retention operations.
Save methods receive `SaveSession`; `commit_save` receives `CommitSaveSession`
with all worker results. `restore` receives `RestoreSession` with the saved and
target topologies and component publication. Use `session.checkpoint_object_ref`
to describe a component-owned object and `session.component_checkpoint` to
publish those references. Worker `save` returns `WorkerSaveResult`, while
`commit_save` returns `CheckpointComponentPublication`. Mutating save sessions
carry `CheckpointCommitPermit`; stale permits are rejected after a coordinator
generation change. See [checkpoint components](checkpoint.md) for registration
examples and the complete capability contract.
