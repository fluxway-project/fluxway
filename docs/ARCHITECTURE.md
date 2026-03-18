# 1. 概述
Fluxway 是一个面向大规模分布式训练的深度学习框架，采用直接 SPMD-to-SPMD 数据传输机制，避免了传统的 "SPMD-to-One + One-to-SPMD" 中间聚合步骤，从而提升训练效率。框架在重构后，进一步加强了面向有向无环图（DAG）的数据驱动编排模型、完全异步化的状态机控制以及极高并行度的数据传输分离。

## 1.1 核心特性
**双平面传播协议**：将数据流分为基于高性能网络协议的 Data Plane（处理张量数据本身）与依赖 Ray RPC/Control Agent 广播的 Control Plane（处理标量、状态量和控制流），根据数据切片进行自动路由分发。

**基于 Mailbox 的纯异步汇聚**：Worker 抛弃了传统事件阻塞，采用 `WorkerMailbox` 管理接收到的数据片。配合 `StepBarrier` 检测条件，一旦所期望的张量和状态量就绪，操作将自动进入 `asyncio.PriorityQueue` 执行。

**细粒度 Async 异步控制和反压机制**：新增了 `AsyncStepManager` 来约束不同 SPMD 中执行异步步数差异；并结合了基于信用的 `_busy_worker_ratio` 进行反向压力追踪，自动使得 Reader 节点级联挂起。

**层级架构与灵活资源管理**：使用全局级别的 `Controller` 调度图信息和 Placement Group；使用局部的 `SPMDDriver` 追踪 Worker 心跳；各个 `SPMDWorker` 则专注于高性能异步 Kernel 计算，真正做到职责隔离。

## 1.2 架构概览
```
+----------------------------------------------------------------------------------+
|                           Control Plane (Ray Actor)                              |
|                                                                                  |
|   +------------------+     +------------------+    +-------------------------+   |
|   |  DAG Parser /    |     |    Scheduler     |    |   AsyncStepManager &    |   |
|   |   Graph Builder  |     |   (PG Manager)   |    |    HeartbeatManager     |   |
|   +------------------+     +------------------+    +-------------------------+   |
|                                                                                  |
|                +------------------------------------------------+                |
|                |              ControllerStateMachine            |                |
|                +------------------------------------------------+                |
+----------------------------------------------------------------------------------+
                                        |
+----------------------------------------------------------------------------------+
|                            SPMD Level (Driver Layer)                             |
|                                                                                  |
|   +----------------------+      +--------------------+      +----------------+   |
|   |   Driver FSM         |      | WorkerSynchronizer |      |  SPMD Monitor  |   |
|   +----------------------+      +--------------------+      +----------------+   |
|                                                                                  |
|                +------------------------------------------------+                |
|                |              Stateful Executor                 |                |
|                +------------------------------------------------+                |
+----------------------------------------------------------------------------------+
                                        |
+----------------------------------------------------------------------------------+
|                           Worker Level (Async Actors)                            |
|                                                                                  |
|   +------------------+     +--------------------------+    +-----------------+   |
|   |  Worker Mailbox  | --> | asyncio.PriorityQueue    | -> | OperatorManager |   |
|   |  (StepBarrier)   |     |                          |    | (Dispatcher)    |   |
|   +------------------+     +--------------------------+    +-----------------+   |
|                                                                                  |
|                +------------------------------------------------+                |
|                |       AsyncDataDistributor & Transport         |                |
|                |               (ULP Protocol)                   |                |
|                +------------------------------------------------+                |
+----------------------------------------------------------------------------------+
```

---

# 2. 控制平面 (Control Plane)
控制平面是整个训练框架的核心协调层，对应代码目录为 `fluxway/control_plane`。

## 2.1 主控制器 (Controller)
`Controller` 是由 Ray Actor 实现的主控入口，负责管理任务的整体生命周期和资源调度。

### 2.1.1 拓扑构建与解析 (Graph Builder & Parser)
`GraphParser` 负责处理和生成有向无环图，将用户编写的层级连线提取成了内部任务视图。并在运行时提供节点的前置、后置关联映射。`FunctionalGraph` 和 `NodeProxy` 为前端 API 提供透明抽象。

### 2.1.2 调度层 (Scheduler & PGManager)
**职责**：基于有向无环图的配置结构进行调度。依赖 `PGManager` 包装了基于 Ray Placement Group 的底层创建，实现了 CPU、GPU 内存的静态计算和按需资源划分分配。
支持创建阶段构建整个网络的节点拓扑连接方式。

### 2.1.3 心跳和限流 (HeartbeatManager & AsyncStepManager)
**双重监控与限流**：
1. **HeartbeatManager**：接收各个 SPMD 的 Driver 带来的心跳汇报信息。并在全部 Driver 准备好后发送拓扑注册请求，在探测到出错时切换系统异常状态抛出错误。
2. **AsyncStepManager**：计算操作的最大步数容忍漂移差，执行针对特定 Op（操作）层级的阻止和释放阻塞通知。
3. **反向压力与 Backpressure**：通过探测下游 Driver 发生大量 mailbox full 或 busy 的概率 (`busy_worker_ratio`)，对入图无前驱节点的 Reader 节点触发 pause 停止拉取新批次。

### 2.1.4 状态管理器与 SPMD 配置 (StateManager & SPMDManager)
负责记录图上每个 SPMD 的句柄，以及对节点启动、驱逐做句柄跟踪。保存并规划整个拓扑的状态持久化策略和集群统计。

---

# 3. SPMD 层面
SPMD 在运行时切分为两层，一层是对内做监督的 `SPMDDriver`，另一层是专注于流式运算的极简 `SPMDWorker`。

## 3.1 SPMD 主控端 (SPMDDriver)
位于 `fluxway/spmd/driver` 目录下，作为该 SPMD 系列资源的总管（Master）。每个 SPMD 有一个特定的 Driver，被 SPMDManager 控制启动。
- **DriverStateMachine (FSM)**：控制当前 SPMD 子循环的局部状态（如初始化等待节点汇聚、暂停、完成等）。
- **WorkerSynchronizer**：执行特定细粒度的 Operation Block。能针对该 SPMD 中每个 Operator（操作符）设定统一的 Step Watermark 水位线限制。
- **StatefulExecutor**：有状态执行器，用来响应并维护模型状态向主存储介质或远程主节点的收发。
- **SPMDMonitor**：周期性收集所有所辖内部的 `SPMDWorker` 上报的心跳。

## 3.2 强异步底层执行流 (SPMDWorker)
重构后的 `SPMDWorker` 使用 Python 原生 `asyncio` 并发替代了原先复杂的异步执行树（Event Loop 模式更加单纯可靠）。

### 3.2.1 双流汇聚与邮箱架构 (Dual-Stream Convergence via WorkerMailbox)
为了支持极端的 "Control + Data" 并行发送到达随机性：
Worker 并不会使用阻塞拉取来推进。相反，采用 `WorkerMailbox` 建立异步回调墙（`StepBarrier`）。
- 控制流数据（如 LR 等由 Controller 发出的标量）在经过 Driver 使用 `receive_control` 发进来。
- 数据流张量（如前向输出特征，ULP 通信包）经过 Data Plane 由 `receive_shards` 发进来。
- 一旦某一对 (`op_name`, `step`) 符合了 `StepBarrier` 的预期收到所有的 tensors / states 数目，它将通过回调函数唤起，把该任务放入 `asyncio.PriorityQueue` 执行队列。

### 3.2.2 优先队列调度 (Execution PriorityQueue)
调度执行循环 `_run()` 仅盯着 PriorityQueue 收取内容，通过对图上的操作权重排序，执行更急迫先获得资源的运算，确保关键链路阻塞最少。

### 3.2.3 操作派发与操作管理 (Operator Manager & Dispatcher)
用户函数已被进一步简化剥离为 `OperatorManager` 管理的具体可派发结构。
`ExecutionDispatcher` 分离了 **逻辑节点** 和 **物理节点**。提供了两类主要派发策略：
- **StandardDispatcher**：正常的 DataParallel 和 SequenceParallel。物理 rank == 逻辑 rank，步步执行。
- **InterleavedDispatcher**：处理特殊的虚拟流水线模式，依据 `stride` 数值对计算流程执行空间复用隔绝。

---

# 4. 数据传输层
底层数据传输位于 `fluxway/transport/communication.py` 等文件中。重构后的传输协议极大解耦了网络拓扑，将原本的 "SPMD To One, One to SPMD" 变为了 JIT 编译加速的 ULP (Universal Layout Protocol) 零拷贝网。

## 4.1 数据切片形式 (Transport Layouts)
通信策略依托于 `PhysicalLayout` 中的定义，包括 `is_content_addressable` （依值路由的 Hash 或 Lookup 等散列）与 `routing_dim`（依行列路由的切片 Dense 等）。并根据目标是否开启 `is_referential`（对称流转回写，如 Mirror 反向梯度流），灵活产生网卡路由寻址图。

## 4.2 AsyncDataDistributor
新的双流架构引入了真正的异步发送管理器 `AsyncDataDistributor`。
**分发工作流**：
1. **打包填充 (Plan and Shard)**：使用 JAX `_jit_plan_and_pack` 的向量化静态指令来对即将发送的批次实现切分和数据重组提取，彻底摆脱 Python 动态层缓慢调度。
2. **异步发送 (Distribute)**：获取 `dest_agg` 内所有下游节点切分后的张量、长度计算块和 Global Shape 等，组成多个长程 RPC（使用 Ray Remote 调用）统一挂在 `asyncio.gather` 同时投递，达成全满带宽高传导。

这种完全避免集合通信堵塞，又兼顾了极高的 Python 层扩展性（支持 Sparse 和 Dense 动态组装）。