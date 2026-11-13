# Fluxway Architecture

[Chinese edition](ARCHITECTURE_zh.md)

Fluxway compiles operator declarations into one graph, places its workers, and
coordinates execution and recovery. The control plane decides when work may run;
the data plane moves the tensors required by that work.

Each SPMD graph node represents workers running the same program over a mesh.
Edges connect operator outputs to inputs.

```mermaid
sequenceDiagram
    participant A as Application
    participant G as Compiled graph
    participant C as Controller
    participant D as Drivers
    participant W as Workers
    participant T as Data plane
    A->>G: Build declarations and compile dependencies,<br/>routes, resources
    G-->>A: Immutable graph and SPMD projections
    A->>C: await start(graph, recipe)
    C->>D: Allocate resources, create actors, register<br/>topology
    D->>W: Install compiled projections
    W-->>C: Readiness aggregated through Drivers
    opt Restore requested
        C->>D: Restore all selected state concurrently
        D->>W: Restore components with execution gates closed
        W-->>C: Restore completion through Drivers
    end
    C->>D: Activate execution and start sources when needed
    C->>D: Deliver finite admission for controlled<br/>execution
    C-->>A: GraphExecution
    loop Admitted execution steps
        W->>W: Invoke operators
        W->>T: Publish tensors
        T-->>W: Receiver ingress and mailbox delivery complete
        W-->>C: Progress or drain reports through Drivers
    end
    A->>C: await wait() or cancel()
    C->>D: Terminal cleanup and actor/resource reclamation
    C-->>A: Completion, cancellation, or failure
```

This overview shows ordinary tensor delivery and collapses intermediate
acknowledgements. Driver and Worker calls remain concurrent, and admitted work
may begin before `start()` returns. Protocol-only paths and the delivery fence
are covered in [data-plane runtime](dataplane_runtime.md); see [execution
lifecycle](execution_lifecycle.md) for barriers
and failure boundaries.

## Layered architecture

Fluxway is organized into five specialized planes:

1. **Control plane**: owns global graph compilation, placement optimization, lifecycle coordination, ingestion layouts, and distributed checkpoint transactions.
2. **Driver / coordination plane**: supervises each SPMD mesh, coordinating step synchronization, flow-control credit, and worker health.
3. **Worker / execution plane**: executes partitioned user operators, managing local step pipelines, input assembly, and local checkpoint state.
4. **Data / transport plane**: handles peer-to-peer tensor delivery, cache coherence (MESI), frame batching, and high-performance carriers (`AxonDataPlane`, `RayActorDataPlane`).
5. **Diagnostics, profiling & cross-cutting plane**: provides non-blocking OS signal interception, actor exit attribution, HLC-synchronized Perfetto tracing, and dynamic resource binding.

```mermaid
flowchart TB
    subgraph ControlPlane["Control Plane"]
        direction TB
        CP_Ctrl["ControllerActor / Controller<br/>(Global Lifecycle & Orchestrator)"]
        CP_FSM["ControllerStateMachine (FSM)<br/>& BoundedEffectScheduler (Async/Thread/Process)"]
        CP_SPMD["SPMDManager & SPMDHandle<br/>(Mesh Topology & Actor Lifecycle)"]
        CP_Sched["PlacementPlanner & Scheduler (PGManager)<br/>(PlacementRecipe, STRICT_SPREAD)"]
        CP_HBFc["HeartbeatManager, Monitor<br/>& FlowControlManager (Credit & Health)"]
        CP_Ing["IngestionLayoutPlanningComponent<br/>& SourceProgressSupervisor (IOLane)"]
        CP_Ledger["LogicalWorkLedger & AlgebraicLayoutCompiler<br/>(Leases, ACK Frontiers, Cuts)"]
        CP_Ckpt["FluxwayCheckpointer & CheckpointRepository<br/>(2PC: STAGED -> COMMIT_DECIDED -> COMMITTED)"]
        CP_CkptReg["CheckpointComponentRegistry & Retention<br/>(SafeTensors, Orbax, WorkerLocal, Inline)"]
        
        CP_Ctrl --- CP_FSM
        CP_Ctrl --> CP_SPMD
        CP_Ctrl --> CP_Sched
        CP_Ctrl --> CP_HBFc
        CP_Ctrl --> CP_Ing
        CP_Ing --> CP_Ledger
        CP_Ctrl --> CP_Ckpt
        CP_Ckpt --> CP_CkptReg
    end

    subgraph DriverPlane["Driver / Coordination Plane (Per SPMD Mesh)"]
        direction TB
        DR_Driver["SPMDDriver (Ray Actor)<br/>(Mesh Supervisor & Admission Coordinator)"]
        DR_FSM["DriverStateMachine (FSM)<br/>& BoundedEffectScheduler"]
        DR_Exec["RayWorkerExecutor<br/>& DriverComponentRegistry (Worker Supervision)"]
        DR_Credit["CreditManager (CreditWindow/Edges)<br/>& SPMDSynchronizer (MaxRankStepSkewSyncPolicy)"]
        DR_Mon["SnapshotOrchestrator, SPMDMonitor<br/>& HeartbeatManager (Readiness Aggregator)"]
        
        DR_Driver --- DR_FSM
        DR_Driver --> DR_Exec
        DR_Driver --> DR_Credit
        DR_Driver --> DR_Mon
    end

    subgraph WorkerPlane["Worker / Execution Plane (Per Rank / Shard)"]
        direction TB
        WK_Worker["SPMDWorker (Ray Actor)<br/>(Sharded Operator Execution)"]
        WK_Lifecycle["WorkerExecutionLifecycle<br/>& StepAdmissionState (Admission Gate)"]
        WK_SourcePipe["SourcePipelineMaintenance<br/>& SourceExecutionScope (Leases & ACKs)"]
        WK_PlanExec["LocalStepPlanExecutor & LocalStepPlan<br/>(WAIT_INPUTS -> CALL_USER -> COMMIT -> SEND -> AWAIT)"]
        WK_Mailbox["WorkerMailbox<br/>(StepBarrier & Dependency Barriers)"]
        WK_Assemble["LocalInputPrepareRuntime & AssemblyRouteResolver<br/>(CompiledAssemblePlan: Gather/De-unique/Concat)"]
        WK_OpMgr["OperatorManager & ExecutionDispatcher<br/>(State, Arguments, Lifecycle)"]
        WK_UserCall["UserCallableExecutor (In-Process Lane)<br/>& WorkerProcessCallRuntime (BaseReader / Subprocess Ops)"]
        WK_Ckpt["WorkerCheckpointCoordinator<br/>& CheckpointActivity (Component Save/Restore)"]
        
        WK_Worker --> WK_Lifecycle
        WK_Lifecycle --> WK_SourcePipe
        WK_SourcePipe --> WK_PlanExec
        WK_Mailbox --> WK_Assemble
        WK_Assemble --> WK_PlanExec
        WK_PlanExec --> WK_OpMgr
        WK_OpMgr --> WK_UserCall
        WK_Worker --> WK_Ckpt
    end

    subgraph DataPlane["Data / Transport Plane (P2P High-Performance)"]
        direction TB
        DP_Out["LocalWorkerDataPlane<br/>& BatchingDataPlaneExecutor (Outbound Queue)"]
        DP_Coherence["CoherenceDirectory (Local MESI Directory)<br/>& CoherenceSequencer (ConsumerLeases)"]
        DP_Retire["CoherenceAdmissionRuntime & BatchRetireRuntime<br/>(Multi-Target Concurrent Enqueue & Child Receipts)"]
        DP_Routing["CompiledSendPlan & CompiledTensorRoute<br/>(Dense, Sparse/MoE, Replicated, Unit Routing)"]
        DP_Wire["TransportTensor ABI & TransportFrame Envelope<br/>(Packed Flags, Headers, DLPack Segments)"]
        DP_Distrib["AsyncDataDistributor & TransportReceiver<br/>(AggregatedTransportGroup)"]
        DP_Backends["DataPlaneCallable Backends:<br/>AxonDataPlane (P2P / Local Deposit) & RayActorDataPlane"]
        DP_Kernels["Assembly & Frame Table Kernels<br/>(FrameTableKernel, Numba JIT, Ragged Tensor)"]
        
        DP_Out --> DP_Coherence
        DP_Coherence --> DP_Retire
        DP_Retire --> DP_Routing
        DP_Routing --> DP_Wire
        DP_Wire --> DP_Distrib
        DP_Distrib --> DP_Backends
        DP_Backends --> DP_Kernels
    end

    subgraph CrossCutting["Diagnostics, Profiling & Cross-Cutting Infrastructure"]
        direction LR
        CC_Diag["fluxway.diagnostics<br/>(prepare_worker_runtime_env, Hook Injection,<br/>Ray SIGTERM Intercept, Pipe2 Emitter, Actor Exit)"]
        CC_Prof["fluxway.profiler<br/>(ProfilerRuntimeFactory, PerfettoTraceWriter,<br/>HLC Clock Sync, Scopes: Async/Function/Launch)"]
        CC_Res["fluxway.common Infrastructure<br/>(ResourceRegistry, LogicalPartitionBinding, msgpack_codec)"]
    end

    %% Inter-plane control and supervision flows
    CP_SPMD ==>|"Supervise Drivers & Distribute Projections"| DR_Driver
    CP_Sched -.->|"Allocate Placement Groups"| DR_Driver
    DR_Exec ==>|"Supervise Workers & Forward Control Admissions"| WK_Lifecycle
    DR_Credit -.->|"Manage Stage Credit & Skew Bounds"| WK_SourcePipe
    WK_SourcePipe -.->|"Async Source ACKs & Step Reports"| CP_Ledger
    WK_Ckpt -.->|"Component Manifests & 2PC Permits"| CP_Ckpt
    DR_Mon -.->|"Driver / Worker Readiness & Telemetry"| CP_HBFc

    %% Worker to Data Plane flows
    WK_PlanExec ==>|"1. Publish Tensor Outputs (SEND_OUTPUT)"| DP_Out
    DP_Backends ==>|"2. P2P Transport (Axon / Ray Carrier) to Remote Workers"| WK_Mailbox
    WK_PlanExec -.->|"3. Return Leases & Retires (InputGrant / MESI)"| DP_Coherence

    %% Cross-cutting observability
    CC_Diag -.-|"Signal Intercept & Exit Attribution"| WorkerPlane
    CC_Prof -.-|"Perfetto Tracing & Unified HLC Clock"| WorkerPlane
    CC_Prof -.-|"Trace Carrier Context"| DataPlane
    CC_Res -.-|"Runtime Handles & DLPack Device Context"| WorkerPlane

```

### Architectural planes and component breakdown

#### 1. Control plane (`fluxway.control_plane`)
- **Controller Core & Lifecycle**: `ControllerActor` / `Controller` acts as the global orchestrator. `ExecutionLifecycle` manages the strict ordering of startup, restore, drain, checkpoint transactions, admission delivery, and failure propagation.
- **State Machine & Effect Scheduler**: `ControllerStateMachine` (`ControllerFSM`) reduces cluster state via pure transition relations. Side-effects are dispatched via `BoundedLifecycleEffectScheduler` across bounded concurrency lanes (`ASYNC` for awaitable I/O, `THREAD` for blocking calls, and `PROCESS` for CPU-intensive compilation).
- **SPMD & Driver Management**: `SPMDManager` and `SPMDHandle` maintain the SPMD mesh topology, supervise driver actor lifecycles, and handle driver registration.
- **Placement & Scheduling**: `PlacementPlanner` solves multi-dimensional placement constraints and lexicographical objectives (`PlacementRecipe`), compiling them into heterogeneous Ray Placement Groups with `STRICT_SPREAD` pinning (`PGManager`, `Scheduler`).
- **Health & Flow Control**: `HeartbeatManager` aggregates driver readiness and monitors node liveness. `FlowControlManager` regulates global step admission and buffer quotas.
- **Data Ingestion Subsystem**: `IngestionLayoutPlanningComponent` isolates data scanning and layout planning on a dedicated `IOLane`. `SourceProgressSupervisor` drives event-driven refills, while `LogicalWorkLedger` manages work epochs, leases, ACK frontiers, and committed cuts. `AlgebraicLayoutCompiler` compiles assignments into `CompiledIngestionLayout` and `WorkerCompiledLayout`.
- **Checkpoint Coordination**: `FluxwayCheckpointer` orchestrates distributed two-phase commit transactions (`STAGED` -> `COMMIT_DECIDED` -> `COMMITTED` / `ABANDONED`) with `CheckpointRepository`. `CheckpointRetentionCoordinator` manages version retention, while `CheckpointComponentRegistry` integrates pluggable storage components (`SafeTensors`, `Orbax`, `WorkerLocalCheckpointComponent`, `StateCheckpointComponent`).

#### 2. Driver / coordination plane (`fluxway.spmd.driver`)
- **SPMD Driver Actor**: `SPMDDriver` serves as the supervisory root for an entire SPMD mesh node.
- **Driver State Machine**: `DriverStateMachine` (`DriverFSM`) processes local mesh state transitions, scheduling effects via `BoundedLifecycleEffectScheduler`.
- **Worker Supervision**: `RayWorkerExecutor` and `DriverComponentRegistry` install graph projections on worker actors and forward control admissions.
- **Credit & Flow Control**: `CreditManager` tracks inter-stage dependency buffers (`CreditWindow`, `CreditEdge`, `UpstreamCreditState`) to prevent buffer bloat.
- **Step Synchronization**: `SPMDSynchronizer` coordinates intra-SPMD step progression via `MaxRankStepSkewSyncPolicy`. Configured via `SPMDConfig.max_rank_step_skew`, it bounds the maximum difference in completed steps between the fastest and slowest ranks in a mesh: `0` enforces strict lockstep synchronization, while `N > 0` allows leading ranks to complete at most `N` additional steps before admission is gated. This rank-level synchronization bound operates independently of operator pipeline depths and inter-SPMD credit flow control. It can be declared statically via `@spmd(max_rank_step_skew=...)` or configured dynamically via `bind_spmd(..., max_rank_step_skew=...)`. Each SPMD configures its synchronization policy independently:

  ```python
  @spmd(max_rank_step_skew=0)  # RecsysBackboneTrain: collective lockstep
  class RecsysBackboneTrain:
      ...

  @spmd(max_rank_step_skew=2)  # Input: bounded rank look-ahead
  class Input:
      ...
  ```
- **Snapshot & Health Monitoring**: `SnapshotOrchestrator` coordinates local worker state snapshots during checkpoints, while `SPMDMonitor` and `HeartbeatManager` aggregate worker health and execution metrics.

#### 3. Worker / execution plane (`fluxway.spmd.worker`)
- **SPMD Worker Actor**: `SPMDWorker` represents a physical rank/shard in the SPMD mesh, executing partitioned user operators.
- **Worker Execution Lifecycle**: `WorkerExecutionLifecycle` manages worker states and enforces execution gates via `StepAdmissionState` (`AdmissionAction`, `AdmissionDecision`).
- **Source Ingestion Pipeline**: `SourcePipelineMaintenance` and `SourceExecutionScope` maintain the worker's active step queue, checking credit and pipeline depth before step acquisition. `SourceWorkRuntime` and `SourceAcknowledgementRuntime` report work leases and submit ACKs asynchronously.
- **Local Step Plan Executor**: `LocalStepPlanExecutor` runs the 5-phase operator micro-step lifecycle:
  1. `WAIT_INPUTS`: Waits for required tensor frames to arrive in `OperatorMailbox` and fulfills `StepBarrier` or direct local completeness requirements.
  2. `CALL_USER`: Prepares operator arguments and invokes user callables.
  3. `COMMIT`: Commits operator outputs and state.
  4. `SEND_OUTPUT`: Submits outputs to outbound transport queues.
  5. `AWAIT_SEND`: Confirms that carrier ingress completed.
- **Operator Mailbox & Input Assembly**: Built on the unified `OperatorMailbox` abstraction, separating `DirectLocalMailbox` (zero-lock fast-path for `local_only` edges) from `DelegatedWorkerMailbox` (delegates distributed ingress and `StepBarrier` synchronization to `WorkerMailbox`). `LocalInputPrepareRuntime` and `AssemblyRouteResolver` use `CompiledAssemblePlan` to unpack, de-duplicate/gather, slice, sort, and concatenate shards into contiguous tensors.
- **Operator Management & Dispatch**: `OperatorManager` and `ExecutionDispatcher` manage operator instances, lifecycle hooks, and dependency route resolution.
- **Execution Lanes**: `UserCallableExecutor` runs in-process operators on `UserCallableLane`. For operators requiring process isolation (such as `BaseReader` and `GrainReader`), `WorkerProcessCallRuntime` coordinates subprocess worker execution.
- **Local Checkpoint Persistence**: `WorkerCheckpointCoordinator` and `CheckpointActivity` persist local component state to storage while holding execution gates closed at the checkpoint cut.

#### 4. Data / transport plane (`fluxway.transport` & `fluxway.rpc`)
- **Outbound Data Plane**: `LocalWorkerDataPlane` and `BatchingDataPlaneExecutor` coalesce outbound frames into batches for efficient transport.
- **MESI Cache Coherence Subsystem**:
  - `CoherenceDirectory`: Endpoint-local MESI directory maintaining entry states (`Modified`, `Exclusive`, `Shared`, `Invalid`).
  - `CoherenceSequencer`: Controls per-subject epoch admission and intent ordering, strictly decoupling the delivery pipeline from underlying buffer retirement, and orchestrating concurrent consumer leases (`ConsumerLease`) with fine-grained ticket IDs.
  - `CoherenceAdmissionRuntime` & `CoherenceBatchRetireRuntime`: Provides atomic admission for multi-target publications, concurrent destination enqueue, and child delivery receipt aggregation.
  - `ParentGrantReturnRuntime`: Returns input grants to upstream producers upon retirement.
- **Routing & Sharding Plans**: `CompiledSendPlan`, `CompiledAssemblePlan`, and `CompiledTensorRoute` lower logical sharding specs: Dense (regular slicing), Sparse (hash/MoE routing), Replicated (broadcast), and Unit routing.
- **Wire Protocol & Physical Transport**:
  - Canonical ABI: `TransportTensor` with DLPack integration.
  - Wire Format: `TransportFrame`, `TransportFrameEnvelope`, and `TransportFrameHeader` with bitpacked `entry_state_flags`.
  - Distribution: `AsyncDataDistributor`, `TransportReceiver`, and `AggregatedTransportGroup`.
  - Transport Backends (`DataPlaneCallable`):
    - `RayActorDataPlane`: Standard Ray actor RPC and object store carrier.
    - `AxonDataPlane`: Direct peer-to-peer high-performance transport supporting CPU and CUDA memory with strict Local Deposit Boundary semantics (`invoke_transport_frames`).
  - Assembly Kernels: `FrameTableKernel`, `ragged.py`, and Numba JIT kernels for accelerated frame table assembly.

#### 5. Diagnostics, profiling & cross-cutting plane (`fluxway.diagnostics`, `fluxway.profiler`, `fluxway.common`)
- **Non-blocking Diagnostics**: `prepare_worker_runtime_env` injects `worker_process_setup_hook` into Ray workers. Intercepts Ray SIGTERM handlers, delegating signals through a non-blocking pipe (`os.pipe2(O_NONBLOCK | O_CLOEXEC)`) to a background daemon emitter (`_SignalEventEmitter`), ensuring zero deadlock in interrupt context. `report_actor_exit` attributes process terminations.
- **High-Resolution Distributed Profiler**: `ProfilerRuntimeFactory` and `PerfettoTraceWriter` output Perfetto `.pftrace` files and Google Model Explorer graph bundles. Synchronizes events across nodes using Hybrid Logical Clocks (`TraceClockSource`), with hierarchical scopes (`async_scope`, `function_scope`) and explicit user launch boundaries via `scope().launched()`.
- **Dynamic Resource Registry**: `ResourceRegistry` and `LogicalPartitionBinding` bind dynamic runtime resources (`ResourceHandle(namespace, resource_id)`) across partitions.
- **Framework & Serialization Support**: Zero-copy DLPack tensor protocol bridging JAX, PyTorch, and NumPy, and binary MessagePack serialization via `msgpack_codec.py`.

## Build and inspect a graph

This complete example compiles two CPU operators without starting Ray. The edge
maps the producer's `value` output to the consumer's `item` input.

```python
import fluxway as fw
from fluxway.common.decorators import spmd


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class Producer:
    @fw.operator(
        name="produce", output_sharding={"value": {"type": "static", "axis": []}}
    )
    def produce(self) -> dict[str, int]:
        return {"value": 1}


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class Consumer:
    @fw.operator(
        name="consume", input_sharding={"item": {"type": "static", "axis": []}}
    )
    def consume(self, item: int) -> None:
        self.last_value = item


builder = fw.Builder()
builder.add_node("source", Producer)
builder.add_node("sink", Consumer)
builder.add_func_edge(
    "source", "produce", "sink", "consume", tensor_map={"value": "item"}
)
graph = builder.build("two-operators")
compiled = graph.require_compiled_graph()

assert len(compiled.dependencies) == 1
assert graph.nodes["source"].spmd_config.graph_view is compiled.spmd("source")
assert compiled.spmd("source").outgoing[0] is compiled.dependencies[0]
```

Both assertions about identity refer to the same compiled objects. Workers and
transfer planning use projections of that graph. This check validates graph
construction; it does not invoke either operator. To run the graph, follow
[execution lifecycle](execution_lifecycle.md#usage).

## Runtime ownership

The compiled graph is the single source of truth for SPMD declarations,
operator identity, dependencies, tensor transfers, placement inventory, and
worker routes. Workers receive immutable graph projections and do not rebuild
sharding or tensor identity metadata.

The Controller owns graph lifecycle, placement, drivers, heartbeats, drain,
checkpoint orchestration, and failure propagation. Drivers supervise workers
and forward control commands. Workers own local execution, admission, pending
step lifecycle, transport delivery, and local plan execution.

`ExecutionLifecycle` centralizes startup, restore, drain/checkpoint continuation,
admission delivery and failure ordering. Driver heartbeats report readiness only.
See [distributed execution lifecycle](execution_lifecycle.md) for the state flow
and ownership table.

Controller and Driver state machines use pure transition relations. A transition commits only
state and failure records; its effect is executed after the lock by the owning lifecycle object.
Source progress has one generation-aware supervisor, and terminal cleanup has one shared future.
Worker close has one release future. These boundaries prevent duplicate tasks and keep Ray,
checkpoint, transport and resource-release work off state-reducer critical sections.

Lifecycle effects use `BoundedLifecycleEffectScheduler`, whose bounded queue and fixed workers
provide explicit capacity. `ASYNC` is reserved for genuinely awaitable I/O; short blocking or
GIL-releasing calls use the fixed thread lane, and pickleable pure-Python CPU work uses the spawn-
based process lane. Same-key effects are ordered, different keys are bounded-concurrent, and a
full queue raises a backpressure error. Outcomes carry `state_revision`; an owner discards stale
outcomes before producing a fact event. CPython's event loop remains single-threaded and
cooperative, so an `async def` declaration alone is not a non-blocking guarantee.

## Coherence and data plane

The endpoint-local MESI directory is the sole coherence state owner. A
`CoherenceInstance(subject, epoch)` is the identity; `CoherenceDirectory.admit`
creates or returns that instance's entry, while `CoherenceSequencer` orders per-subject
intents. Directory operations are event-loop local and do not own worker tasks
or transport queues.

`TransportTensor` is the canonical tensor ABI. Tensor slicing, frame assembly,
batching, routing, and lineage belong to the transport/data-plane layer rather
than the worker scheduler. Exact-step inputs are required; another step cannot
be substituted as a fallback.

## Compiled transfer plans

`TensorTransfer.unit_routing` declares whole-unit incidence independently of source
operators. `CompiledDependency` validates and binds the relation; `CompiledSendPlan`
lowers it alongside ordinary sharding and receipt routing. Execution groups separate
physical identities from logical layout coordinates. All payloads use the existing
DataPlane. See [transfer planning](transfer_planning.md).

## Coherence fanout publication

One worker local step submits its complete immutable target-batch tuple to
`CoherenceBatchRetireRuntime.retire_publication()`. The runtime admits the
publication's instance set once, registers each unique child receipt, and starts
all target enqueues concurrently. `CoherenceSequencer` retains only subject
epoch admission and terminal retirement ordering; it does not serialize sibling
readonly deliveries for one instance.

For the CPU/Ray carrier, receiver completion means the frame was admitted and
deposited into the receiver mailbox. The receiver then owns its materialized
carrier, while the publication keeps the producer outbound references only until
all child results terminate. Same-worker aliases remain protected by their
consumer leases. This avoids a general variable-size buffer ring and does not
claim cross-process zero-copy. A target failure cancels unfinished siblings and
fails the complete publication through its single receipt owner.

## Drain and restore

One `DrainBoundary` reducer owns observed, admitted, drained and committed cuts.
Controller derives `DrainAdmission(revision, target, terminal_cut)` from that reducer
and carries it only on control messages and completion reports. Driver and Worker
reject stale or conflicting admissions before opening a new finite gate. Completion
requires the registered participant set and the active admission identity.

Controller mutation locks contain state reduction, not Driver RPC, source ACK waits
or checkpoint cancellation waits. Source ACK coverage is checked after drain and
before checkpoint preparation. Restore rebuilds canonical plans and physical bindings,
applies the committed cut, and restores model/runtime state with the Worker gate
closed at that cut. Driver `resume` resets checkpoint bookkeeping without clearing
the Worker gate; the next admission supplies its finite target.

## Checkpoint transaction ownership

Controller/Repository checkpoint coordination uses `CheckpointTransactionState`,
distinct from the Worker-local `CheckpointSaveState`:

```mermaid
flowchart LR
    S[STAGED] --> D[COMMIT_DECIDED]
    D --> C[COMMITTED]
    S --> A[ABANDONED]
    D --> A
```

`COMMIT_DECIDED` records that physical publication has started. If a crash occurs
before the global commit marker is written, startup fences the transaction via `abort_save(permit)`
to `ABANDONED`. If the commit marker was already written, startup fencing replays component
`acknowledge_save(permit)` and advances the transaction to `COMMITTED`. Only `COMMITTED` manifests
are restorable and visible. Manifests, transactions, and cleanup intents are persisted
using binary MessagePack via `msgpack_codec`. This format does not imply a fixed
storage size or a latency guarantee.

## Placement and execution

Placement declarations come from compiled SPMD declarations and
`ResourceProfile`. `PlacementRecipe` expresses constraints and lexicographic
objectives without duplicating resource ownership. Each solved NodeID maps to one
composite bundle in a shared heterogeneous placement group. Ray's `ray.io/node-id`
bundle label selector pins the exact identity under `STRICT_SPREAD`; IP resource
markers are not node identities. CPU, GPU and custom accelerator resources retain
the quantities selected by the planner. Workers attest their actual NodeID against
the plan. Multi-node execution requires compatible node environments and a shared
checkpoint repository when recovery is enabled.

Runtime-bound dynamic resources (such as dynamic routing tables, device meshes, or rank
providers) are managed through `ResourceRegistry` (`fluxway.common.resource_registry`)
and referenced via `ResourceHandle(namespace, resource_id)`. The active registry context is established
by `bind_resource_registry`. `LogicalPartitionBinding` selects worker-local values
by logical partition, optionally projecting selected layout axes. It does not define
a physical-rank remapping or authorize topology changes during restore; those remain
part of the component's [checkpoint contract](checkpoint.md#topology).

The worker admission owner contains gating and pending-state promotion.
`SourcePipelineMaintenance.request()` coalesces requests to fill the Worker source
step queue; `fill()` checks running state, pipeline depth, drain boundaries and
the total step target before enqueueing work. This queue maintenance does not
discover source revisions or publish new ingestion windows. The execution lane
consumes owned queue items, evaluates admission, dispatches plans, and completes
`task_done()`.

## Ingestion layout and source coordination

`IngestionSourceAdapter` returns immutable `SourceScanResult` values for bounded
scan requests. The controller's `SourceProgressSupervisor` owns event-driven
scan and activation publication. The compiler first creates canonical assignments,
identified by `SourceWorkKey(epoch, step, assignment_id)`, then binds them to current
reader quotas in transient `WorkerLayoutDescriptor` values. `LogicalWorkLedger`
alone owns leases, ACK frontiers and committed cuts. Driver activation checkpoint
state is empty. Restore reconstructs plans
and bindings before applying the logical committed cut.

`IngestionLayoutPlanningComponent` manages the lifecycle of source bindings and adapters. Binding construction, data scanning, and resource disposal run on a dedicated `IOLane`, while pure compilation uses a persistent process pool. On shutdown, the planner rejects new requests, drains in-flight operations, shuts down the adapter and compiler pool concurrently, and reclaims lane resources.

The planner and `IOLane` use `AsyncOperationSupervisor` and `AsyncCloseOnce` for concurrency safety: cancelling a caller's wait does not leak background tasks or resources, closure executes idempotently, and reentrant self-awaits are prevented from deadlocking. See [source coordination](source_coordination.md) for sequence details.

Planner and ledger publication is installed in
one owner-loop section without suspension; validation and installation costs still
scale with window size. I/O threads isolate blocking calls but do not remove the
GIL for Python CPU work.

`SourceExecutionScope` owns Worker window installation, the active-step index and
transient grants. Acquisition batches active steps up to operator pipeline depth;
the authority returns the complete-step prefix permitted by current credit.
Normal checkpoint continuation retains the scope. Restore uses fresh actors before
source activation. ACK submission uses a bounded asynchronous queue; full capacity
applies backpressure. Controller startup activates one layout. The planner component
supports idle observations and pending tails. The startup API starts this lifecycle;
the controller supervisor performs event-driven adapter scans and replenishment,
while the adapter itself does not own refill. Durable cuts reclaim completed
Controller windows, ledger bindings, and scan observations, retaining a continuity
anchor. Worker-installed windows and Driver activation digests remain retained,
so per-window metadata growth still requires validation.
See [source coordination](source_coordination.md) and
[formal protocol](source_spmd_formal_protocol.md) for the supported contracts.

## Process diagnostics and signal capture

To reliably attribute worker process exits caused by external preemption, evictions, or
operating system signals, Fluxway provides non-blocking process-level diagnostic
interception in `fluxway.diagnostics`:

- **Bootstrap hook injection**: `prepare_worker_runtime_env` injects `worker_process_setup_hook` via Ray's `WORKER_PROCESS_SETUP_HOOK_ENV_VAR`, initializing signal interceptors before worker execution while chaining importable user hooks.
- **Ray SIGTERM interception**: Intercepts Ray's internal `set_sigterm_handler` to record previous handlers and Ray's own SIGTERM callback without altering Ray's teardown protocol.
- **Non-blocking signal delivery**: Signal handlers enqueue events and wake a background emitter daemon thread via a non-blocking pipe (`os.pipe2(O_NONBLOCK | O_CLOEXEC)`), preventing deadlocks or blocking I/O inside OS interrupt context.
- **Faithful delegation**: Handlers always forward signals to prior handlers (`callable`, `SIG_DFL` via `raise_signal`, or `SIG_IGN`), preserving native process termination semantics.
- **Thread safety**: Signal handlers are only installed on the process main thread; asynchronous actor worker threads publish diagnostic metadata without attempting invalid signal registrations.

The worker launcher prepares this environment automatically. The following local
example is for authors integrating a worker launcher; it creates configuration
without registering signal handlers in the submitting process.

```python
from fluxway.diagnostics import prepare_worker_runtime_env

original = {"env_vars": {"PYTHONUNBUFFERED": "1"}}
prepared = prepare_worker_runtime_env(original)

assert original == {"env_vars": {"PYTHONUNBUFFERED": "1"}}
assert prepared["env_vars"]["PYTHONUNBUFFERED"] == "1"
assert len(prepared["env_vars"]) == 2
```

The added variable selects the Fluxway bootstrap hook. Supply user hooks as
importable module paths; a callable object or a manually supplied Ray hook
environment variable is rejected. The reporter preserves existing signal
semantics; it cannot report signals that the operating system does not let a
process intercept, such as `SIGKILL`.

```mermaid
sequenceDiagram
    participant O as Operating system
    participant H as Main-thread signal<br/>handler
    participant E as Background emitter
    participant P as Previous signal<br/>disposition
    O->>H: Catchable system signal
    H->>H: Record event
    H->>E: Wake through a nonblocking pipe
    H->>P: Delegate to previous disposition
    par Diagnostic output
        E->>E: Read event and emit diagnostics
    and Original signal semantics
        P->>P: Terminate, ignore, or invoke user handler
    end
```

The signal handler does not await output. An immediate process exit may still
prevent the diagnostic record from being written.

## Mechanism examples

| Task | Guide |
| --- | --- |
| Start, observe, cancel, and resume execution | [Execution lifecycle](execution_lifecycle.md) |
| Register state, save, restore, and configure retention | [Checkpoints](checkpoint.md) |
| Bound scans and track source assignments | [Source coordination](source_coordination.md) |
| Compile routing and move tensor payloads | [Transfer planning](transfer_planning.md), [data-plane runtime](dataplane_runtime.md) |
| Admit instances and release consumer leases | [Coherence protocol](coherence_protocol.md) |
| Place workers and bind runtime resources | [Placement recipe](placement_recipe.md) |
