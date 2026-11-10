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

## Async Execution and Cooperative Scheduling Architecture

In Fluxway, concurrent operator execution is governed by three orthogonal policy dimensions: **Async Execution Policy (`AsyncPolicy`)**, **Queue Scheduling Policy (`SchedulePolicy`)**, and **State Mutex Policy (`StatePolicy`)**. These dimensions are decoupled and unified by the framework compiler into a high-performance, memory-safe execution runtime block.

```mermaid
flowchart TD
    subgraph Spec["Operator Policy Declaration"]
        AP["AsyncPolicy<br/>(Sliding window & in-flight depth)"]
        SP["SchedulePolicy<br/>(Phase ordering & queue priority)"]
        ST["StatePolicy<br/>(Instance lock & thread safety)"]
    end

    AP -->|Compiles to| CAP["CompiledAsyncPolicy<br/>(is_pipelined, max_staleness)"]
    SP -->|Compiles to| PG["PhaseGate Barriers & Queue Sorter"]
    ST -->|Binds to| LK["Actor Instance Lock / Thread-Safe Dispatch"]

    CAP & PG & LK --> RUNTIME["SPMDWorker Cooperative Execution Engine"]
```

---

### 1. Async Execution Policy (`AsyncPolicy`)

`AsyncPolicy` defines the operator's forward progression window across steps or microbatches, offering three peer and orthogonal concurrency execution modes:

#### `AsyncPolicy.sync` (Strict Single-Step Synchronous)

* **Core Semantics**: Strict single-step execution with fixed pipeline depth 1 (`depth=1`). The next step is never scheduled until the current step finishes input preparation, execution, and commit.
* **When to Use**: Single-step debugging and verification, strong barrier synchronization, or strictly non-overlappable sequential workloads.

```mermaid
gantt
    title AsyncPolicy.sync (depth=1) Execution Gantt
    dateFormat X
    axisFormat %s
    section Worker Compute
    Step 0 Compute :active, 0, 10
    Step 1 Compute :active, 10, 20
    section Ingress Prepare
    Step 0 Prepare :0, 3
    Step 1 Prepare :10, 13
```

```mermaid
sequenceDiagram
    participant Ingress as Input Ingress
    participant Worker as Compute Worker
    participant Sink as Commit Sink

    Ingress->>Worker: Step 0 Input Prepared
    Worker->>Sink: Step 0 Compute & Commit
    Ingress->>Worker: Step 1 Input Prepared (Waits for Step 0 Commit)
    Worker->>Sink: Step 1 Compute & Commit
```

#### `AsyncPolicy.pipelined` (Sliding Window Pipelining)

* **Core Semantics**: Sliding-window asynchronous pipelining. Permits up to $K$ microbatches or steps to be in flight concurrently. The runtime automatically overlaps upstream prefetching, input preparation, and compute execution; applies natural backpressure when $K$ in-flight steps are reached.
* **When to Use**: Data ingestion, feature transforms, deep learning forward/backward passes, standard pipeline parallelism.

```mermaid
gantt
    title AsyncPolicy.pipelined (depth=2) Execution Gantt
    dateFormat X
    axisFormat %s
    section Worker Compute
    Step 0 Compute :active, 3, 10
    Step 1 Compute :active, 10, 17
    Step 2 Compute :active, 17, 24
    section Ingress Prepare
    Step 0 Prepare :0, 3
    Step 1 Prefetch (Overlap) :3, 6
    Step 2 Prefetch (Overlap) :10, 13
    Step 3 Prefetch (Overlap) :17, 20
```

```mermaid
sequenceDiagram
    participant Ingress as Input Ingress
    participant Worker as Compute Worker
    participant Sink as Commit Sink

    Ingress->>Worker: Step 0 Input Prepared
    par Compute and Prefetch Overlap
        Worker->>Sink: Step 0 Compute Execution
    and
        Ingress->>Worker: Step 1 Input Ready Ahead of Time (Prefetch)
    end
    Worker->>Sink: Step 1 Compute Execution (Zero-bubble handoff)
```

#### `AsyncPolicy.bounded_stale` (Bounded Staleness Parallelism / SSP)

* **Core Semantics**: Bounded Stale Synchronous Parallelism (SSP). Allows operators to iterate locally using parameters that are up to $M$ steps stale; when staleness lag reaches $M$, the runtime triggers a synchronization fence barrier to pause compute until parameters catch up.
* **When to Use**: Large-scale recommendation embedding lookups, Parameter Server architectures, or asynchronous SGD tolerant to bounded staleness.

```mermaid
gantt
    title AsyncPolicy.bounded_stale (depth=2, max_staleness=2) Execution Gantt
    dateFormat X
    axisFormat %s
    section Worker Compute
    Step 0 (Weights W0) :active, 0, 6
    Step 1 (Weights W0) :active, 6, 12
    Step 2 (Weights W0) :active, 12, 18
    Staleness Barrier Stall :crit, 18, 24
    Step 3 (Weights W1) :active, 24, 30
    section Parameter Sync
    W0 to W1 Background Sync :0, 24
```

```mermaid
sequenceDiagram
    participant Worker as Compute Worker
    participant Param as Parameter Synchronization

    Worker->>Worker: Step 0 Compute (Using weights W0)
    Worker->>Worker: Step 1 Compute (Permitted to use stale weights W0)
    Worker->>Worker: Step 2 Compute (Reaches staleness limit M=2)
    Note over Worker,Param: Staleness Fence Barrier triggered, pausing until weights synchronize
    Param-->>Worker: W1 Weights Synchronized
    Worker->>Worker: Step 3 Compute (Resumes with fresh weights W1)
```

#### Rapid Selection Decision Tree

```mermaid
flowchart TD
    Start{"What execution concurrency does the operator require?"}
    Start -->|Single-step serial debugging / Strict step commit| Sync["AsyncPolicy.sync()<br/>(depth=1, zero cross-step overlap)"]
    Start -->|Streaming data / Pipelined overlap / Multi-microbatch training| Pipe["AsyncPolicy.pipelined(depth=K)<br/>(Sliding window overlap, bounded backpressure)"]
    Start -->|Parameter Server / Asynchronous SGD / Bounded staleness| Stale["AsyncPolicy.bounded_stale(depth=K, max_staleness=M)<br/>(SSP mode, auto-fence on staleness bound)"]
```

---

### 2. State Access Policy (`StatePolicy`)

* **`StatePolicy.exclusive()`**: **Default policy**. Serializes execution for operators sharing the same SPMD instance lock. Parameter updates and stateful mutable buffers must preserve this policy.
* **`StatePolicy.thread_safe()`**: Removes the instance lock. Used when the operator is completely stateless or its internal resources are thread-safe, permitting concurrent cross-step execution on the same actor.

---

### 3. Queue Scheduling Policy (`SchedulePolicy`)

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

---

### 4. Orthogonal Composition Patterns

Fluxway avoids rigid monolithic modes. By orthogonally composing `AsyncPolicy` (progression depth and window) with `SchedulePolicy` (causal barriers and ordering), diverse high-performance patterns emerge naturally:

#### Pattern A: Streaming Ingestion & Backpressure

Ideal for pure data pipelines, feature transformation, and stateless passes. Works out-of-the-box with `pipelined` and default `FIFO` scheduling without any phase barriers, providing natural bounded memory backpressure:

```python
# Streaming pipeline: upstream prefetch overlaps with compute; naturally backpressures at depth=4
@operator("process", async_policy=AsyncPolicy.pipelined(depth=4))
def process(self, batch: Batch) -> Tensor:
    return self.transform(batch)
```

#### Pattern B: Cross-Phase Pipeline Parallelism (1F1B / GPipe)

Tailored for modern distributed deep learning. `AsyncPolicy.pipelined(depth=K)` drives $K$ microbatches through the pipeline concurrently, while `SchedulePolicy.phase_order` enforces causal barrier gates between Forward and Backward phases:

```mermaid
sequenceDiagram
    participant Driver as Ingestion / Driver
    participant Gate as PhaseGate Barrier
    participant FWD as Forward Operator (fwd)
    participant BWD as Backward Operator (bwd)

    Driver->>Gate: Microbatch 0 Ingress
    Gate->>FWD: Admit Step 0 Forward
    FWD-->>Gate: Step 0 Forward Committed
    Gate->>BWD: Admit Step 0 Backward
    par Pipelined Overlap (1F1B)
        Driver->>Gate: Microbatch 1 Ingress
        Gate->>FWD: Admit Step 1 Forward
    and
        BWD-->>Gate: Step 0 Backward Committed
    end
```

```python
class PipelineParallelStage:
    """Standard 1F1B Pipeline Parallel Stage Operator Group"""

    @operator(
        "fwd_stage",
        async_policy=AsyncPolicy.pipelined(
            depth=4,  # In-flight microbatch depth
            schedule=SchedulePolicy.phase_order(phase="fwd", phase_order=("fwd", "bwd")),
            state=StatePolicy.exclusive(),
        ),
    )
    def fwd_stage(self, microbatch: Tensor) -> Tensor:
        return self.forward_compute(microbatch)

    @operator(
        "bwd_stage",
        async_policy=AsyncPolicy.pipelined(
            depth=4,
            schedule=SchedulePolicy.phase_order(phase="bwd", phase_order=("fwd", "bwd")),
            state=StatePolicy.exclusive(),
        ),
    )
    def bwd_stage(self, grad_output: Tensor) -> Tensor:
        return self.backward_compute(grad_output)
```

#### Pattern C: Barrier-Isolated Multi-Phase Training

For actors encapsulating both training and optimizer updates. Using `depth=1` with `phase_order` physically prevents the next step from observing partially updated parameters:

```python
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

During graph compilation (`Builder.build`), the framework resolves operator execution traits and infers canonical lanes based on static operator metadata (`fluxway.common.canonical_lanes`):

* **`OperatorExecutionTrait.IO_SOURCE`**: Operators marked with `is_source=True`, declaring a `source` descriptor, or belonging to a class inheriting from `BaseReader` default to the `"io"` lane.
* **`OperatorExecutionTrait.SYNC_COMPUTE`**: Regular synchronous compute operators (`def` methods) default to the `"compute"` lane and run on bounded worker thread pools where C extensions release the GIL without blocking the event loop.
* **`OperatorExecutionTrait.ASYNC_COROUTINE`**: `async def` operators remain unassigned to thread lanes (`lane=None`) and execute directly on the asyncio event loop for lightweight asynchronous I/O and network coordination.

```mermaid
flowchart TD
    Op["@operator Definition"] --> Resolve["resolve_operator_execution_trait()"]
    Resolve --> Trait{"OperatorExecutionTrait"}
    Trait -->|IO_SOURCE| IOLane["Canonical 'io' Lane"]
    Trait -->|SYNC_COMPUTE| CompLane["Canonical 'compute' Lane"]
    Trait -->|ASYNC_COROUTINE| Loop["Asyncio Event Loop (lane=None)"]

    IOLane --> Sched["compute_lane_priority(policy, lane)"]
    CompLane --> Sched
    Sched --> PriQueue["Cooperative Priority Dispatcher"]
```

The helper `compute_lane_priority(policy, lane)` serves as the single source of truth for priority calculation based on `SchedulePolicy.lane_priorities` or `SchedulePolicy.lane_ordering`.

The following example demonstrates static trait resolution, canonical lane inference, and priority computation:

```python
from fluxway.common.canonical_lanes import (
    compute_lane_priority,
    infer_canonical_lane,
    resolve_operator_execution_trait,
)
from fluxway.common.decorators import get_spmd_config, operator, spmd
from fluxway.common.operator_policy import SchedulePolicy
from fluxway.common.operator_spec import OperatorExecutionTrait
from fluxway.resources import ResourceProfile


@spmd(resource_profile=ResourceProfile(base={"CPU": 1}))
class PipelineModel:
    @operator(name="read_batch", is_source=True)
    def read_batch(self) -> dict:
        return {"x": 1}

    @operator(name="sync_compute")
    def sync_compute(self, x: int) -> int:
        return x * 2

    @operator(name="async_coord")
    async def async_coord(self, x: int) -> int:
        return x + 1


config = get_spmd_config(PipelineModel)
read_spec = config.operator_specs["read_batch"]
compute_spec = config.operator_specs["sync_compute"]
async_spec = config.operator_specs["async_coord"]

assert resolve_operator_execution_trait(read_spec, PipelineModel) == OperatorExecutionTrait.IO_SOURCE
assert infer_canonical_lane(read_spec, PipelineModel) == "io"

assert resolve_operator_execution_trait(compute_spec, PipelineModel) == OperatorExecutionTrait.SYNC_COMPUTE
assert infer_canonical_lane(compute_spec, PipelineModel) == "compute"

assert resolve_operator_execution_trait(async_spec, PipelineModel) == OperatorExecutionTrait.ASYNC_COROUTINE
assert infer_canonical_lane(async_spec, PipelineModel) is None

# Single-source priority calculation
policy = SchedulePolicy.fifo(lane="compute", lane_order=("compute", "io"))
assert compute_lane_priority(policy, "compute") == 1
assert compute_lane_priority(policy, "io") == 0
```

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
