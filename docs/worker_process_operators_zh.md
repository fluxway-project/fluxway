# 工作进程算子

[英文版](worker_process_operators.md)

有状态算子实例属于一个 SPMD 工作进程。如果构造函数创建进程本地句柄，
或者需要最终的工作进程拓扑，例如调用 `jax.distributed.initialize`，使用
`@requires_worker_process`。普通有状态类不需要这一标记。

## 声明算子并释放资源

下面将无状态的输入准备与有状态的文件写入分开。
`bind_spmd` 保留原始类供本地检查，并创建用于计算图执行的 SPMD 声明。
写入器只有一个工作进程，因此输出路径只有一个所有者。该路径必须尚不存在。

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

直接调用会在 CPU 上检查文件内容、资源生命周期和编译后的策略，不会运行 Ray 工作进程。
计算图展示了本地交接；启动作业前，需要为 `prepare` 连接实际数据源。
`local_only=True` 表示同一工作进程内的依赖，不是远程调用。

启动时，Fluxway 在每个工作进程内为每个所有者类创建一个实例，并在该类的算子之间共享。
构造参数取自节点的静态参数，例如上面的 `output_path`。
在 `__init__` 之前，Fluxway 还会提供 `rank`、`world_size`、`local_rank`、
`local_world_size`、`node_rank`、`num_nodes`、`spmd_id`、`rank0_address`
和 `ray_component_limits`。`self.ctx` 携带当前调用的上下文，会随调用变化。

正常关闭工作进程时，即使多个算子共享同一实例，Fluxway 也只调用一次该实例的
`on_shutdown()`。该方法可以是同步函数，也可以返回可等待对象。
关闭过程会尝试清理所有实例，并传播清理错误。强制终止进程不能保证执行清理；
缓冲文件也不提供检查点或崩溃恢复保证。注册表不会替资源所有者关闭对象，见
[资源所有权](placement_recipe_zh.md#注册工作进程内创建的资源)。

## 用户代码在哪里运行

| 可调用对象 | 执行方式 |
| --- | --- |
| 普通 Python 函数 | 从第一次调用起，异步提交到工作进程内的命名线程通道 |
| 返回协程对象的普通函数 | 工作线程运行该协程至完成 |
| 协程函数（`async def`） | 使用 `COROUTINE_ASYNC`，在执行器的事件循环上协作运行 |

`@requires_worker_process` 控制构造位置，不改变上表中的选择。
协程等待时必须让出控制权；在 `async def` 内执行阻塞操作会阻塞执行器的事件循环。
进程编号和执行步等上下文变量会随调用传入工作线程。异常会传播到本地执行计划并使工作进程失败。

这样，普通算子运行时，传输接收、信箱更新、输入准备、输出交付和调度器唤醒仍可推进。
标准 CPython 下，计算密集的 Python 代码仍然竞争 GIL。
释放 GIL 的原生代码可以重叠；自由线程版 CPython 还可以并行执行 Python 字节码，
通道和调度规则保持一致。

```mermaid
sequenceDiagram
    participant P as 本地执行计划
    participant E as UserCallableExecutor
    participant T as 命名线程通道
    participant L as 工作进程事件循环
    P->>P: 按状态策略取得所需资源
    P->>E: execute(spec)
    alt 普通 Python 函数
        E->>T: 提交函数及上下文变量
        par 用户函数在线程运行
            T->>T: fn()
            opt 返回协程对象
                T->>T: 在线程中运行协程至完成
            end
        and 事件循环继续推进
            L->>L: 接收、邮箱、交付与调度
        end
        T-->>E: 结果或异常
    else 协程函数
        E->>L: await fn()
        L->>L: 在每次 await 处协作让出控制权
        L-->>E: 结果或异常
    end
    E-->>P: 返回结果或使本地计划失败
    P->>P: 释放本次调用的状态资源
```

线程通道与状态锁是不同边界：`thread_safe()` 允许共用实例的调用重叠，`exclusive()` 保留互斥。协程内的同步阻塞工作仍会阻塞事件循环。

### 关闭时的并发边界

```mermaid
sequenceDiagram
    participant C as 关闭调用方
    participant W as 工作进程唯一关闭任务
    participant T as 用户调用运行时
    participant O as 算子拥有者
    participant P as 检查点协调器
    C->>W: await close()
    W->>W: 停止调度，取消并等待计划、交付与后台任务
    par 结算用户调用
        W->>T: await drain()
        T-->>W: 调用结算结果
    and 清理实例
        W->>O: 每个唯一实例调用一次 on_shutdown()
        O-->>W: 同步返回或等待异步钩子
    and 清理检查点
        W->>P: 回滚活动尝试并关闭
        P-->>W: 清理结果
    end
    W->>W: 关闭执行器，使张量失效并封锁协议
    W-->>C: 共享结果或汇总异常
```

同步 `on_shutdown()` 直接在工作进程事件循环调用；耗时清理应使用异步钩子并自行分派阻塞工作。实例钩子与用户调用排空并发执行，不保证钩子开始前所有线程调用已经结束。拥有者须协调仍在进行的资源访问；取消协程等待也不能终止已运行的 Python 线程。强制结束进程不保证执行钩子。

## 状态与调度策略 (State & Schedule Policy)

在 Fluxway 中，算子的并发执行由**状态互斥策略 (`StatePolicy`)**与**队列调度策略 (`SchedulePolicy`)**共同决定。

### 1. 状态访问策略 (`StatePolicy`)
* `StatePolicy.exclusive()`：**默认策略**。对同一个 SPMD 实例加锁串行化，模型权重修改、有状态存储必须保留此策略。
* `StatePolicy.thread_safe()`：移除实例锁。仅当算子为纯无状态转换或其内部资源完全线程安全时使用，允许同实例多步并行执行。

### 2. 调度策略一分钟极速选型指南 (`SchedulePolicy`)

为了让业务方在 10 秒内明确选择最适合的调度策略，请参考以下决策树与速查表：

```mermaid
flowchart TD
    Q1{"同一 SPMD 内是否包含多个修改共享状态的连续阶段？\n(如 Forward -> Backward -> Update)"}
    Q1 -- "否 (90% 场景: Reader、ETL、无状态变换、单算子训练)" --> R1["使用默认 FIFO (无需显式指定！)\n开箱即用 + 自动规范泳道 + 天然背压"]
    Q1 -- "是 (同一模型包含多阶段更新)" --> Q2{"阶段间是否有复杂的跨步非对称重叠需求？"}
    Q2 -- "否 (标准的步内串行阶段)" --> R2["SchedulePolicy.phase_order\n严格步进门禁，物理阻止跨步脏读"]
    Q2 -- "是 (精细重叠优化)" --> R3["SchedulePolicy.phase_graph\n显式定义跨步因果依赖图"]
    
    Q3{"业务是否要求防长尾/落后批次紧急提权？"} -->|是| R4["SchedulePolicy.deadline_boost\n滞后步数超限时动态爆发插队"]
```

#### 策略速查与代码模板对照表

| 策略类型 | 什么时候用 | 框架默认行为 / 保证 | 核心代码范式 |
| :--- | :--- | :--- | :--- |
| **`fifo` (默认)** | **90% 业务场景**：数据读取、特征物化、无状态前向、单算子训练。 | **默认开箱即用**。自动归入 `io`/`compute` 规范泳道并自动配齐协同优先级 `("compute", "io")`，下游计算优先出队消费。 | `@operator("read", async_policy=AsyncPolicy.pipelined(depth=4))` *(无需写 schedule)* |
| **`phase_order`** | **多阶段模型训练**：同一个 Actor 内有 `train_core` 与 `update` 等多个阶段，必须确保下一步的计算不会读取到正在更新的脏权重。 | 框架编译底层 `PhaseGate` 门禁。在 Step $S$ 优化器更新完成前，Step $S+1$ 计算**物理被拦截在执行队列之外**，绝对无竞态。 | `SchedulePolicy.phase_order(phase="fwd", phase_order=("fwd", "bwd", "opt"))` |
| **`phase_graph`** | **复杂跨步重叠**：模型更新时，允许 Step $S+1$ 的部分辅助工作（如梯度预通信）与 Step $S$ 更新重叠，但核心前向必须等待。 | 用户通过 `(before, after, step_delta)` 显式表达细粒度因果依赖 DAG。 | `SchedulePolicy.phase_graph(phase="fwd", phase_order=..., dependencies=(("opt", "fwd", 1),))` |
| **`deadline_boost`** | **防掉队者 / 周期性刷盘**：平时保持低优先级让出资源，当滞后步数超过 `staleness_limit` 时，临时提权到 `boost_priority` 优先出队。 | 自动监控调度步数滞后差，超限立即在 Min-Heap 队列中提权插队。 | `SchedulePolicy.deadline_boost(base_priority=0, boost_priority=100)` |

#### 代码范例对比

```python
# 1. 默认 FIFO：最简形式，适合绝大多数数据流与单算子
@operator("process", async_policy=AsyncPolicy.pipelined(depth=4))
def process(self, batch: Batch) -> Tensor:
    return self.transform(batch)

# 2. Phase Order：多阶段训练，防止下一步读到未更新的脏参数
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

启用性能分析后，`user_callable.execute` 包含排队时间；
`user_callable.function_run` 只测量函数调用，不包含通道排队和工作进程上下文设置。
`worker_process_call.thread_run` 记录工作线程的执行区间。
分析器观察已经选定的通道，不会再次分发调用。

## 初始化错误

类调用 `jax.distributed.initialize` 却缺少标记时，会被拒绝并报
`Fluxway Worker Process Error`。应使用装饰器，不要直接设置内部字段
`__fluxway_process_affine__`。该标记不会安装 JAX、创建分布式协调器或修正不一致的工作进程拓扑。

## 标准执行泳道与自动协同调度 (Canonical Lanes & Cooperative Scheduling)

在分布式与高性能计算中，算子通常具有不同的计算与 IO 语义特征。若将所有算子混入单一无序队列，或者为每个函数动态分裂出无限个线程池，都会导致系统调度失控、线程上下文切换过载或内存反压失效。

Fluxway 在编译计算图时，基于算子元数据（Metadata）自动进行粗粒度的**标准规范泳道（Canonical Lanes）归类与协同调度规划**：

```mermaid
flowchart TD
    subgraph SPMDWorker["SPMD 工作进程 (Worker Actor)"]
        direction TB
        subgraph EventLoop["AsyncIO 主事件循环 (COROUTINE_ASYNC)"]
            CoroutineOp["async def 算子\n(调度、网络、等待 IPC、数据面流转)"]
            Heartbeat["节点心跳 & 集群协调"]
        end

        subgraph WorkerProcessRuntime["WorkerProcessCallRuntime (工作进程有界泳道)"]
            ComputeLane["'compute' 泳道\n(同步普通计算算子、矩阵运算、特征提取)"]
            IOLane["'io' 泳道\n(BaseReader 来源读取、is_source=True 算子、磁盘拉取)"]
        end

        subgraph PriorityQueue["Min-Heap 协同调度优先队列"]
            ComputeHigher["compute 算子 (优先级更高: 1)"]
            IOLower["io 算子 (优先级更低: 0)"]
            ComputeHigher -->|优先消费就绪数据| IOLower
        end
    end

    EventLoop -. "始终保持响应与网络无阻塞" .-> Heartbeat
    PriorityQueue --> ComputeLane
    PriorityQueue --> IOLane
```

### 1. 标准泳道粗粒度自动规划规则

在 `Builder.build` 编译图阶段，框架根据算子的元数据语义特征自动执行推导（实现位于 `fluxway.common.canonical_lanes`）：

* **`io` 泳道**：凡是算子标记为 `is_source=True`、声明了 `source` 描述符或所属类继承自 `BaseReader` 的算子，默认归入 `"io"` 泳道。
* **`compute` 泳道**：普通的同步计算算子（非 IO 的 `def` 算子）默认归入 `"compute"` 泳道，并在专属有界工作线程池中执行，C 扩展完全释放 GIL，不阻塞主事件循环。
* **异步协程通信算子**：`async def` 协程算子不分配线程池泳道（保持 `lane=None`），直接由 asyncio 主事件循环调度，专职负责跨节点高效通信与异步编排。

### 2. 自动配齐协同优先级与天然背压

当框架在图中同时识别到 `compute` 泳道与 `io` 泳道时，会自动为未显式指定排序的算子注入标准协同优先级：
```python
lane_order = ("compute", "io")
```
在 Worker 调度引擎与优先队列中，`compute` 泳道算子获得更高的调度权重（`priority(compute) > priority(io)`）。这确保了已生成的批次数据被下游计算算子**优先消费**，而不是任由读取算子无限制预取撑爆内存，从而在框架层实现了优雅且高性能的生产-消费协同背压。

### 3. 开闭原则与显式定制

默认规则为开箱即用的高性能标准实践。如果算子有特殊调度需求，可直接在 `SchedulePolicy` 中显式指定，框架严格遵从用户显式配置，绝不覆盖：

```python
# 示例：显式定制算子泳道与优先级
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

## 检查点扩展边界

算子不实现检查点钩子。通过 SPMD 的 `checkpoint_components` 注册
`CheckpointComponentRegistration`。
`SAVE` 组件处理 `prepare_save`、`save`、`finish_save`、`commit_save` 和
`publish_save`；`rollback_save` 处理失败事务。`RESTORE` 组件处理 `restore`。
对象持有和清理预留属于 `OBJECT_LIFECYCLE`，物理删除属于 `OBJECT_CLEANUP`。
控制侧通过 `abort_save` 丢弃未完成的保存尝试，在全局提交成功后调用 `acknowledge_save`。

控制器负责事务顺序和清单发布。组件负责物理 URI、暂存布局、版本和保留操作。
保存方法接收 `SaveSession`，`commit_save` 接收包含全部工作进程结果的
`CommitSaveSession`。`restore` 接收 `RestoreSession`，其中包含保存时和目标拓扑以及组件发布结果。
用 `session.checkpoint_object_ref` 描述由组件管理的对象，再用
`session.component_checkpoint` 发布这些引用。
工作进程的 `save` 返回 `WorkerSaveResult`，`commit_save` 返回
`CheckpointComponentPublication`。变更保存状态的会话携带 `CheckpointCommitPermit`；
协调器换代后，旧许可会被拒绝。注册示例与完整能力约定见[检查点组件](checkpoint_zh.md)。
