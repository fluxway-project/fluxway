1. 概述
Fluxway 是一个面向大规模分布式训练的深度学习框架，采用直接 SPMD-to-SPMD 数据传输机制，避免了传统的 "SPMD-to-One + One-to-SPMD" 中间聚合步骤，从而提升训练效率。框架采用分层架构设计，支持异步执行、灵活的资源调度和动态任务编排。其中 DAG 数据驱动编排（DAG-based Data-Driven Orchestration）、双平面传播协议（Dual-Plane Propagation Protocol）和双流汇聚机制（Dual-Stream Convergence）是框架的特色。

1.1 核心特性
**双平面传播协议**：将 SPMD 间的数据传输分为 Data Plane（数据平面）和 Control Plane（控制平面）两条路径，根据数据类型自动选择最优传输策略

**双流汇聚机制**：Worker 采用数据驱动的 `On(Data) AND On(Control)` 执行模型，实现数据流与控制流的异步汇聚

**异步执行**：前向和反向传播支持异步执行，Data Plane 数据流水线化，提升计算资源利用率

**DAG 编排**：基于有向无环图（DAG）的任务编排，节点在所有上游依赖完成后自动执行，实现数据驱动的执行模型

**智能资源管理**：基于 Ray Placement Group 的资源调度机制，支持条件分支共享资源、抢占和驱逐，优化资源利用率

**节点状态恢复**：支持节点驱逐前的状态保存和恢复后的状态加载，确保训练连续性

**Handle 映射机制**：逻辑 Handle 和实际 Handle 的映射层，自动处理节点变更和恢复，提供透明的调用接口

**状态管理**：状态量的监控和持久化，支持增量 Checkpoint 和全量恢复

**监控管理**：分布式集群硬件和调度指标遥测

**框架无关设计**：数据传输层与底层框架（JAX/PyTorch/NumPy）解耦，支持多种 Sharding 策略：Dense（规则切分）、Sparse（Hash/MoE 切分）和 Replicated（广播）
1.2 架构概览

![架构图](docs/pictures/structure.png)
2. 控制平面 (Control Plane)
控制平面是整个训练框架的核心协调层，负责全局视图管理、资源调度和状态监控。
2.1 主控层 (Main Orchestrator)
主控层是框架的最顶层，负责整体训练流程的编排和管理。
2.1.1 DAG 解析器 (DAG Parser)
职责：
解析用户定义的有向无环图（DAG）任务编排
构建节点依赖关系和数据流
识别 SPMD 层级和连接关系
计算拓扑排序和执行层级
验证 DAG 的有效性（无环检验、可达性检查）
核心能力：
数据驱动执行：节点在所有上游依赖完成后自动执行
拓扑排序：计算有效的执行顺序，确保依赖关系正确
并行层级：识别可并行执行的节点层级
Source/Sink 识别：自动识别入口节点（无上游）和出口节点（无下游）
输入：
用户定义的 DAG 配置文件（JSON/YAML）或 Python API
DAG 节点定义（SPMD 配置、上下游依赖）
输出：
内部任务视图（Task View）
SPMD 层级拓扑图
拓扑排序结果
执行层级划分
数据结构：
class DAGNode:
    node_id: str
    spmd_config: SPMDConfig  # worker数量、资源需求等
    upstream_nodes: List[str]  # 上游依赖节点 ID 列表
    downstream_nodes: List[str]  # 下游被依赖节点 ID 列表
    user_forward_fn: Callable
    user_backward_fn: Callable
class DAGEdge:
    source_node: str  # 上游节点
    target_node: str  # 下游节点
    is_gradient_edge: bool  # 是否为梯度传递边
class TaskDAG:
    nodes: Dict[str, DAGNode]
    edges: List[DAGEdge]
    source_nodes: List[str]  # 入口节点（无上游依赖）
    sink_nodes: List[str]  # 出口节点（无下游依赖）
执行模型：
当节点的所有 upstream_nodes 完成执行后，该节点自动变为 ready 状态
ready 状态的节点可以被调度执行
同一执行层级的节点可以并行执行
2.1.2 调度器 (Scheduler)
职责：
基于 Ray Placement Group 进行资源分配
根据 SPMD 的资源需求（CPU、GPU、内存）进行调度
优化资源利用率（考虑亲和性、数据局部性）
处理资源不足和重调度场景
支持抢占和驱逐：基于 Ray API 实现资源抢占和节点驱逐
调度策略：
资源感知调度：根据 SPMD 的计算和通信需求分配资源
亲和性调度（TODO(何嘉)：留接口，先不做）：优先将频繁通信的 SPMD 调度到相近节点
动态调整（TODO(何嘉)：留接口，慢点做）：支持运行时资源调整
条件分支共享 PG（TODO(何嘉)：留接口，慢点做）：不同触发条件下的子节点共享同一个 Placement Group
条件分支资源管理：
共享 Placement Group：同一父节点的多个条件分支子节点共享同一个 PG
资源充足场景：当资源足够时，同时启动多个子节点以应对上游所有可能的请求
资源不足场景（TODO(何嘉)：留接口，先不做）：同一 PG 内触发抢占与驱逐机制
根据优先级和资源需求决定抢占顺序
被抢占的节点保存状态后驱逐
新节点启动时从 Checkpoint 恢复状态
抢占与驱逐机制（TODO(何嘉)：留接口，先不做）：
基于 Ray API：使用 Ray 的 Placement Group 和 Actor 生命周期管理 API
优先级策略：根据节点重要性、资源需求、执行阶段等因素确定优先级
优雅驱逐：驱逐前触发节点状态保存（Checkpoint）
资源回收：驱逐后立即回收资源供新节点使用
接口：
class Scheduler:
    def schedule(self, task_fsm: TaskFSM) -> PlacementPlan:
        """根据FSM生成资源分配计划"""
        pass
    
    def allocate_resources(self, plan: PlacementPlan) -> PlacementGroup:
        """基于Ray Placement Group分配资源"""
        pass
    
    def create_shared_pg(self, node_ids: List[str], resources: ResourceSpec) -> PlacementGroup:
        """为条件分支节点创建共享Placement Group"""
        pass
    
    def preempt_node(self, node_id: str, reason: str) -> bool:
        """抢占节点资源（基于Ray API）"""
        pass
    
    def evict_node(self, node_id: str, checkpoint: bool = True) -> bool:
        """驱逐节点（保存状态后驱逐）"""
        pass
    
    def check_resource_availability(self, resources: ResourceSpec) -> bool:
        """检查资源可用性"""
        pass
2.1.3 SPMD  Manager(全局SPMD 管理器)
职责：
SPMD生命周期：根据调度结果启动各个 SPMD 的 Master 和 Worker，管理 SPMD 的生命周期
SPMD拓扑结构：建立 SPMD 之间的连接关系，确立上下游关系
SPMD初始化：初始化 SPMD 的状态和内部状态机（训练子循环）
状态恢复：通过直接调用或状态机指示各个SPMD从 Checkpoint 恢复节点状态
节点变更：管理节点的创建、驱逐和重建
启动流程：
为每个 SPMD 创建或复用 Placement Group（条件分支共享 PG）
检查是否存在 Checkpoint，如果存在则恢复状态
启动 SPMD Master（Ray Actor）
启动 SPMD Workers（Ray Actors）
建立 Worker 之间的 Handle 连接（逻辑 Handle 映射）
初始化状态管理
状态恢复机制：
SPMD  Manager发起保存把状态所有权State Manager或从State Manage获取状态的所有权
Checkpoint 保存：节点被驱逐前自动保存状态
状态恢复：节点重新启动时从 Checkpoint 恢复
增量恢复：支持增量 Checkpoint 和全量恢复
一致性保证：确保恢复后的状态与驱逐前一致
状态保存与加载指示：能够规划各个SPMD向哪里保存状态/Ckpt，从哪里恢复状态
节点生命周期管理（TODO(何嘉)：留接口，先不做）：
创建：根据调度计划创建新节点
驱逐：资源不足时驱逐低优先级节点（保存状态）
重建：触发条件满足时重建被驱逐的节点（恢复状态）
更新：节点变更时更新 Handle 映射
接口（TODO(何嘉)：接口重新定义）：
class Orchestrator:
    def register_spmd(self, spmd_id: str, Handle: SPMDHandle):
        """注册SPMD"""
        pass

    def start_spmd(self, spmd_id: str, config: SPMDConfig, 
                   checkpoint_id: Optional[str] = None) -> SPMDHandle:
        """启动一个SPMD实例，可选从Checkpoint恢复"""
        pass
    
    def stop_spmd(self, spmd_id: str, save_checkpoint: bool = True):
        """停止SPMD实例，可选保存Checkpoint"""
        pass
    
    def get_spmd_handle(self, spmd_id: str) -> SPMDHandle:
        """获取SPMD句柄"""
        pass
    
    def evict_spmd(self, spmd_id: str) -> str:
        """驱逐SPMD并返回Checkpoint ID"""
        pass
    
    def restore_spmd(self, spmd_id: str, checkpoint_id: str) -> SPMDHandle:
        """从Checkpoint恢复SPMD"""
        pass
    
    def update_Handle_mapping(self, spmd_id: str):
        """更新SPMD的Handle映射"""
        pass
2.1.4 State Manager (全局状态管理器)
职责：
统一状态管理：汇总所有 SPMD 的业务状态信息，向外部提供状态查询接口
持久化策略：持有持久化策略，可定时通过SPMD Manager持久化 Checkpoint
支持节点状态恢复：管理节点驱逐前的状态保存和恢复后的状态加载
状态信息：
业务层训练Metric Dict（当前 step、loss 、auc等）
Profiler层级的资源使用情况
Checkpoint 版本和位置
训练Metric使用Tensorboard格式?
持久化策略：
增量 Checkpoint：定期保存状态快照
全量 Checkpoint：在关键节点（如 epoch 结束）保存完整状态
驱逐前 Checkpoint：节点被驱逐前自动触发状态保存
容错机制：支持从 Checkpoint 恢复训练
Metric后端：先只支持Tensorboard格式
节点状态恢复机制：
自动保存：节点被驱逐前自动触发状态保存
状态一致性：确保保存的状态包含所有必要信息（参数、优化器状态、随机数种子等）
快速恢复：支持快速从 Checkpoint 恢复节点状态
增量恢复：支持增量 Checkpoint 和全量恢复两种模式
接口：
class StateManager:  
    def save_checkpoint(self, checkpoint_id: str, state: GlobalState):
        """保存全局检查点"""
        pass
    
    def load_checkpoint(self, checkpoint_id: str) -> GlobalState:
        """加载检查点"""
        pass

    def get_latest_checkpoint(self, spmd_id: str) -> Optional[str]:
        """获取节点最新的Checkpoint ID"""
        pass
    
    def list_checkpoints(self, spmd_id: str) -> List[str]:
        """列出节点的所有Checkpoint"""
        pass
    
    def get_spmd_state(self, spmd_id: str, state: SPMDState):
        """接收SPMD状态"""
        pass

    def get_global_state(self) -> GlobalState:
        """获取当前全局状态"""
        pass
    
    def save_spmd_state(self, spmd_id: str, state: SPMDState) -> str:
        """保存节点状态，返回Checkpoint ID"""
        pass
    
    def load_spmd_state(self, spmd_id: str, checkpoint_id: str) -> SPMDState:
        """加载节点状态"""
        pass
    
2.1.4 Monitor (集群状态监控)
职责：
汇总所有 SPMD 的集群状态信息
监控集群健康状态
监控集群信息查询接口
向子控制器探活，收集子控制器的集群信息
遥测信息：
SPMD 运行状态（运行中、暂停、错误、被驱逐等）
节点监控的资源使用情况（nvidia-smi等）
节点驱逐和恢复历史
持久化策略：
使用Tensorboard格式还是做在Ray监控层？
接口：
class Monitor:
    def report_telemetry(self, state: dict[str, WorkerTelemetry]):
        """向某个Metric服务上报"""
        pass
    
    def get_spmd_telemetry(self, spmd_id: str) -> WorkerTelemetry:
        """查询Worker遥测结果"""
        pass

3. SPMD 层面
SPMD 控制面是框架的执行层，每个 SPMD 代表一个透明的并行的执行单元，包含 一个常驻的控制器和多个 Worker。
3.1 SPMD Sub Orchestrator
SPMD Master 负责协调该 SPMD 内部的所有 Worker，管理状态同步和分布式锁。
3.1.1 SPMD Sub FSM
职责：
根据Main Orchestrator的下发执行SPMD内部的子循环
在特定条件下阻塞全部 Worker 的执行或调用全部 Worker Actor 的函数
实现同步点（Barrier）
使用场景：
参数更新同步：在第 K+1 个 step 的前向之前，必须等待第 K 个 step 的反向完成（当开启梯度累积时）。假设SPMD0和SPMD1连接，SPMD1做反向的时候是由FSM的loop的执行内容决定SPMD0是否调用SPMD0的优化器参数更新方法。
Checkpoint 同步：所有 Worker 必须同步到达检查点
调试和监控：在需要时暂停所有 Worker
实现机制：
通过Stateful Executor有状态地对SPMD进行执行
状态机支持条件触发（如 step % K == 0）
3.1.2 SPMD Stateful Executor
职责：
执行保存、加载、同步 Worker 的状态的函数
执行SPMD Worker的无状态函数（如计算）
收集 Worker 的业务Metric（loss、throughput 、auc等）
向 Main Orchestrator 的 State Manager 汇报状态
状态信息：
SPMD内部的训练Metric Dict（当前 step、loss 、auc等）
Profiler层级的资源使用情况
模型参数ckpt等大规模State
3.1.2 SPMD Sub Monitor
职责：
监控所有 Worker 的运行状态
遥测信息：
心跳检测：定期接收 Worker 心跳
信息聚合：汇总 Worker 遥测值（平均值、最大值等）
异常检测：识别 Worker 故障或性能异常
接口：
```python
class SPMDMonitor:
    def report_worker_telemetry(self, state: dict[str, WorkerTelemetry]):
        """向Main Orchestrator worker遥测报告"""
        pass

    def report_worker_telemetry(self, worker_id: str, state: WorkerTelemetry):
        """向Main Orchestrator某个worker遥测报告"""
        pass
    
    def get_worker_telemetry(self, worker_id: str) -> WorkerTelemetry:
        """查询Worker遥测结果"""
        pass
```
3.1.2 SPMD Worker Registrar
职责：
注册SPMD内Worker Actor的Handle
注册Worker的名字和RPC Server的UCP地址
状态信息：
注册信息Dict
3.1.3 SPMD Worker Function Signatures
职责：
收集和保存已注册的Worker Actor支持的函数
收集和保存已注册的Worker Actor的UCX RPC支持的函数（TODO(何嘉)：UCX RPC先用Ray实现）
状态信息：
函数签名信息Dict
3.1.3 SPMD Sharded Partition Manager
职责：
声明当前各个SPMD UCX RPC函数的输入输出的Shared Partition信息
从Flax nnx.Module中提取函数并根据SPMD的资源Mesh进行Jit然后提取Dense Shared信息
状态信息：
某个函数对应的Shared信息
3.2 SPMD Worker
SPMD Worker 是实际执行计算任务的单元，每个 Worker 运行用户定义的前后向逻辑。
3.2.1 Worker视图
职责：
在初始化时获取所有前置/后置 SPMD 的逻辑 Handle
维护逻辑 Handle 和实际 Handle 的映射关系
处理节点变更（驱逐、重建）时的 Handle 更新
提供 Handle 查询接口，自动处理节点恢复
创建UCX RPC的Server和Client
核心机制：
逻辑 Handle：节点视角中的稳定 Handle 标识
实际 Handle：底层 Ray Actor 的实际 Handle（可能变更）
自动映射：逻辑 Handle 调用时自动映射到当前实际 Handle
节点恢复：如果节点被驱逐，自动触发恢复流程
初始化流程：
从 Main Orchestrator 获取拓扑信息
获取前置 SPMD 的所有逻辑 Handle（用于接收数据）
获取后置 SPMD 的所有逻辑 Handle（用于发送数据）
建立逻辑 Handle 到实际 Handle 的初始映射
注册 Handle 变更监听器
Handle 映射机制：
映射表：维护 逻辑Handle ID -> 实际Handle 的映射
自动更新：当节点变更时，自动更新映射表
故障检测：检测到 Handle 失效时，触发节点恢复
透明调用：调用逻辑 Handle 时，自动路由到实际 Handle
数据结构：
class LogicalHandle:
    """逻辑Handle，提供稳定的接口"""
    logical_id: str  # 逻辑ID，不随节点变更而改变
    spmd_id: str
    worker_id: str
class ActualHandle:
    """实际Handle，对应Ray Actor"""
    actor_handle: ray.actor.ActorHandle
    worker_id: str
    created_at: float  # 创建时间戳
class NodeView:
    current_spmd_id: str
    current_worker_id: str
    # 逻辑Handle映射
    predecessor_logical_Handles: Dict[str, List[LogicalHandle]]  # spmd_id -> [logical_Handles]
    successor_logical_Handles: Dict[str, List[LogicalHandle]]    # spmd_id -> [logical_Handles]
    # Handle映射表：逻辑Handle -> 实际Handle
    Handle_mapping: Dict[str, ActualHandle]  # logical_id -> actual_Handle
    
    def get_predecessor_Handles(self, spmd_id: str) -> List[LogicalHandle]:
        """获取前置SPMD的所有逻辑Handle"""
        pass
    
    def get_successor_Handles(self, spmd_id: str) -> List[LogicalHandle]:
        """获取后置SPMD的所有逻辑Handle"""
        pass
    
    def call_logical_Handle(self, logical_id: str, method: str, *args, **kwargs):
        """通过逻辑Handle调用，自动映射到实际Handle"""
        # 1. 查找映射表中的实际Handle
        # 2. 如果Handle为空或失效，触发节点恢复
        # 3. 调用实际Handle的方法
        # 4. 如果调用失败，更新映射并重试
        pass
    
    def update_Handle_mapping(self, logical_id: str, actual_Handle: ActualHandle):
        """更新Handle映射"""
        pass
    
    def handle_node_eviction(self, spmd_id: str):
        """处理节点驱逐，触发恢复流程"""
        pass
节点恢复流程：
检测到 Handle 失效（节点被驱逐）
通知 Orchestrator 触发节点恢复
Orchestrator 从 Checkpoint 恢复节点状态
重新创建节点并获取新的实际 Handle
更新 Handle 映射表
继续执行被中断的操作

3.2.2 双流汇聚设计 (Dual-Stream Convergence)
SPMD Worker 本质上是一个事件驱动的事件循环（Event Loop）。为了维持异步性，需要将"控制信息"也视为一种"数据"，与数据流并行到达 Worker 的本地缓存区。

**核心原则**：Worker 不应该主动去"拉取"控制信息，而是控制信息应该"先于"或"并行于"数据信息到达 Worker。

**执行逻辑**：从 `On(Data)` 升级为 `On(Data) AND On(Control)`

**① 控制流先行 (Control Plane Pre-scheduling)**：
- Controller 是全局视角的，它预知接下来的 Batch 需要什么配置（LR, Global Norm 等）
- Controller 主动将配置打包，异步发送到所有 Worker 所在节点（Host 端）
- Worker 节点上有一个轻量级的"控制信箱 (Control Mailbox)"
- 控制信息可能先于数据到达，静候在信箱中等待

**② 数据流到达 (Data Plane Arrival)**：
- Worker 从上游 SPMD 接收到 Tensor 数据
- 数据包必须携带对齐标识 (Alignment ID)，如 `batch_id` 或 `step_id`

**③ 本地匹配与触发 (Local Matching & Trigger)**：
- Worker 拿到数据后，用 ID 去本地"控制信箱"中查询对应的配置
- **情况 A（正常）**：控制信息已到达，Worker 零延迟读取本地内存配置，立即开始计算
- **情况 B（异常）**：控制信息尚未到达，Worker 在本地阻塞等待（Wait on Future），不消耗网络带宽轮询

**关键技术点**：

**1. 唯一对齐标识 (Alignment ID)**
数据流和控制流通过共同的 Key（如 `batch_id + global_step`）在 Worker 处汇合。

**2. 基于信用的流控 (Credit-based Flow Control)**
防止 Controller 和数据流速度不匹配：
- Worker 定期汇报消费进度（`consumed_step`）
- Controller 检查待处理批次数是否超限

**3. 边车代理 (Sidecar Agent)**
物理实现上，不让 GPU Kernel 直接处理网络包：
- Host CPU 上运行轻量级 Agent 接收 Controller 推包
- Agent 将配置写入 Host-Device Shared Memory 或 Pinned Memory
- GPU Worker 只需读取内存中的标志位（Flag），真正实现"异步且高效"

**伪代码实现**：

```
@ray.remote
class SPMDWorker:
    control_mailbox = {}    # batch_id -> config
    control_events = {}     # batch_id -> Event
    
    # Control Plane: Controller 预推送配置
    async set_control(batch_id, config):
        control_mailbox[batch_id] = config
        if batch_id in control_events:
            control_events[batch_id].signal()  # 唤醒等待的数据处理
    
    # Data Plane: 接收数据并等待控制信息
    async receive_data(batch_id, tensor):
        if batch_id not in control_mailbox:
            control_events[batch_id] = new Event()
            await control_events[batch_id].wait()  # 异步等待
        
        config = control_mailbox.pop(batch_id)
        return execute_compute(tensor, config)

class ControlMailbox:
    # 存储预推送的控制信息，支持：
    # - put(batch_id, config): 写入配置，唤醒等待者
    # - get(batch_id, timeout): 获取配置，超时抛异常
    # - 反压机制：超出 max_cache_size 时触发反压

class ControlConfig:
    # 单批次配置：batch_id, learning_rate, global_step, clip_norm, is_training 等
```

3.2.3 前向/反向/更新参数接口
用户接口设计遵循"逻辑上统一，物理上分离"的原则，通过类型注解实现自动路由。

**设计原则**：
- 用户只需要定义纯粹的计算逻辑，不需要关心数据传输
- 用户可以自定义任何函数输入
- 为了方便编译优化，forward/backward 函数禁止使用 Dict、args、kwargs 作为输入
- 通过装饰器注解区分 Data Plane 和 Control Plane 参数

**输入类型注解**：
使用 `@input_type` 装饰器声明参数的传输类型：
- `data_inputs`：Worker 独立/异步数据，走 Data Plane（ULP 协议）
- `control_inputs`：Worker 间共享副本或同步数据，走 Control Plane
- `sync_inputs`：需要聚合的同步数据，在 SPMD 内部先执行 AllReduce

**示例接口**：
```
class UserCustomModule:
    # forward: 前向计算
    @input_type(data_inputs=['input0', 'input1'],       # Sharded Tensors → Data Plane
                control_inputs=['learning_rate', 'step']) # 共享变量 → Control Plane
    def forward(input0, input1, learning_rate, step) -> (output0, output1)
        """
        用户定义的前向逻辑
        inputs: 来自前置SPMD的输入数据
        returns: 输出数据（将传递给后置SPMD）
        """
        pass
    
    # backward: 反向计算
    @input_type(data_inputs=['grad'], control_inputs=['clip_norm'])
    @output_type(data_outputs=['grad_input'], sync_outputs=['loss'])  # loss 自动 AllReduce
    def backward(grad, clip_norm) -> (grad_input, loss)
        """
        用户定义的反向逻辑
        通常对于一个梯度，反向函数计算出两个梯度，一个用来继续反向传播，一个用来更新参数
        returns: 传递给前置SPMD的梯度
        """
        pass
    
    # apply_updates: 参数更新（由 Sub FSM 触发）
    @input_type(control_inputs=['learning_rate'])
    def apply_updates(learning_rate) -> metrics
        """
        用户定义的参数更新逻辑，可能会返回一些更新参数的Metric
        一般由Sub FSM通过Stateful Executor触发，也可以由backward触发。
        注意如果由backward触发则需要在UserCustomModule内进行诸如Step的状态维护
        """
        pass
    
    def save_states(self, *args, **kwargs):
        """
        SPMD保存加载
        """
        pass
        
    def load_states(self, *args, **kwargs):
        """
        SPMD保存加载
        """
        pass
```

**框架自动处理**：
- `data_inputs`：通过 ULP 协议从前置/后置 SPMD Worker 接收
- `control_inputs`：从 SPMD Sub Orchestrator 注入，所有 Worker 相同
- `sync_outputs`：自动在 SPMD 内执行 AllReduce 后传给主控

**类型路由实现**：
框架在 Wrapper 层自动识别参数类型，进行路由：
1. **Data Plane 参数**：通过 ULP 协议从前置/后置 SPMD 的对应 Worker 接收
2. **Control Plane 参数**：从 SPMD Sub Orchestrator 获取，所有 Worker 接收相同的值
3. **Sync Output**：在 SPMD 内部先执行 AllReduce，再传递给下游或主控

**框架特定实现**：
- **JAX**：Control 参数作为 JIT 的 `static_argnums`，或使用 `PartitionSpec(None)` 标记为 Replicated
- **PyTorch**：Control 参数已存在于每个进程内存，直接作为参数传入
- **NumPy/通用**：由 SPMD Sub Orchestrator 在调用前广播给所有 Worker


3.2.4 用户模块Wrapper
职责：
- 包装用户的前向/反向函数
- 实现双平面传播协议的路由逻辑（详见第4节）
- 分离 Data Plane 和 Control Plane 参数
- 管理异步执行和同步点
- 处理数据传输的状态管理
- 自动聚合 Sync Output（如 Loss、Gradient Norm）

**Wrapper 伪代码**：
```
class TypeRouter:
    # 根据装饰器注解分离参数
    route_inputs(all_inputs) -> (data_params, control_params, sync_params)
    route_outputs(outputs) -> (data_outputs, sync_outputs)

class ForwardWrapper:
    def __call__(self, step: int, user_fn: UserForwardFunction):
        1. 异步接收 Data Plane 参数（ULP 协议）
        2. 从 Orchestrator 获取 Control Plane 参数
        3. 等待 Data 参数接收完成
        4. 合并参数，调用用户前向函数
        5. 路由输出：分离 data_outputs 和 sync_outputs
        6. 异步发送 Data Output 到后置 SPMD
        7. 聚合 Sync Output（AllReduce）后发送给主控
        8. 向FSM非阻塞发送完成标识

class BackwardWrapper:
    def __call__(self, step: int, user_fn: UserBackwardFunction):
        1. 异步接收梯度数据（Data Plane）
        2. 获取 Control Plane 参数
        3. 等待梯度接收完成
        4. 调用用户反向函数
        5. 路由输出：分离梯度（Data）和聚合量（Sync）
        6. 异步发送梯度到前置 SPMD
        7. 聚合 Sync Output（如 AllReduce Loss）
        8. 向FSM非阻塞发送完成标识
```

**关键设计**：
- **统一接口**：用户函数签名保持统一，无需区分 Data 和 Control 参数
- **自动路由**：Wrapper 根据装饰器注解自动分流参数到不同平面
- **异步执行**：Data Plane 参数异步接收和发送，Control Plane 参数同步注入
- **自动聚合**：Sync Output 自动在 SPMD 内执行 AllReduce
- **框架无关**：通过抽象接口适配 JAX/PyTorch/NumPy


4. 数据传输层
数据传输层负责 SPMD 之间的数据传播，采用双平面传播协议（Dual-Plane Propagation Protocol），根据数据类型特征选择最优传输路径。
4.1 数据分类 (Data Classification)
SPMD 之间传输的数据根据其在 Worker 间的分布特征，可分为以下三类：
4.1.1 Worker 独立/异步 (Worker-Independent / Local)
**定义**：这类变量在每个 Worker（如每个 GPU 卡）上拥有不同的物理数值，是并行计算的核心。

**特征**：
- 不需要跨 Worker 通信，直接并行计算
- 每个 Worker 持有数据的不同分片或独立副本
- 可以异步执行，无需同步屏障

**典型变量**：
- **Data Tensors (DP)**：数据并行切分后的输入数据（User Logs, Features）
- **Sharded Parameters (MP/ZeRO)**：模型并行或 ZeRO 切分后的权重分片
- **Local Optimizer State**：对应分片权重的动量（Momentum）、方差（Variance）
- **Unique RNG Keys**：在 Dropout 或负采样时，每个 Worker 必须拥有不同的随机种子
- **KV Cache (Sharded)**：每个 Head 或 Partition 上的 Cache

**传输策略**：使用 ULP 协议进行 Worker-to-Worker 的点对点传输

4.1.2 Worker 间共享副本 (Replicated / Shared)
**定义**：这类变量在逻辑上是"全局唯一"的，但在物理上被复制到了每一个 Worker 上。它们必须保持数值严格一致。

**特征**：
- 控制整个训练流的行为
- 所有 Worker 必须持有相同的数值
- 如果 Worker 间不一致，会导致训练发散
- 通常是小数据量的标量或配置

**典型变量**：
- **Hyperparameters**：Learning Rate (LR)、Weight Decay 系数
- **Global Step**：当前训练步数
- **Control Flags**：is_training、use_dropout
- **Global RNG Seed**：用于参数初始化时的种子（所有卡初始化相同的权重）

**传输策略**：通过 SPMD Sub Orchestrator 的控制平面传播，不走 Worker-to-Worker 数据通道

**反模式警告**：
> [!WARNING]
> 不要把共享变量（LR、Config）广播成 Tensor 走 Worker-to-Worker 的数据通道。在大规模集群（如千卡）上，通过数据通道传输标量是浪费带宽和调度开销的，且在 JAX/XLA 中不友好。

4.1.3 SPMD to SPMD 显式同步 (Explicitly Synchronized)
**定义**：这类变量是计算产生的中间结果，需要跨 Worker 聚合（Reduce）后才能传递给下一个 SPMD。

**特征**：
- 这是一个 Barrier 点，前一个 SPMD 的所有 Worker 必须在此处握手
- 需要执行 AllReduce、AllGather 等集合通信操作
- 聚合后的结果需要传递给下游 SPMD

**典型变量**：
- **Global Loss**：所有卡 Loss 的平均值
- **Gradient Norm**：用于 Gradient Clipping（所有分片梯度的 L2 范数）
- **Batch Norm Statistics**：全局的 Running Mean/Var（需要 AllReduce）

**传输策略**：先在 SPMD 内部执行集合通信聚合，再通过控制平面或数据平面传递给下游 SPMD

4.2 双平面传播协议 (Dual-Plane Propagation Protocol)
为了高效处理上述三类数据，框架采用双平面传播协议，在逻辑上提供统一 API，在物理上根据数据类型自动选择传输路径。

**核心原则**：逻辑上统一（Unified API），物理上分离（Dual-Plane Propagation）

4.2.1 Path A: Data Plane (Worker-to-Worker)
**传输对象**：
- Worker 独立/异步数据（Sharded Tensors, Local States, Unique RNGs）

**传输协议**：
- 使用 ULP（Universal Layout Protocol）协议
- Worker-to-Worker 点对点传输
- 支持异步执行和流水线

**实现机制**：
- 直接映射到底层框架的集合通信或并行计算指令
- 零拷贝传输，最小化数据复制
- 向量化路由计算

4.2.2 Path B: Control Plane (Controller-to-Controller)
**传输对象**：
- Worker 间共享副本（Shared Configs, LR, Steps, Global Flags）
- SPMD to SPMD 显式同步数据（聚合后的结果）

**传输协议**：
- 由 SPMD Sub Orchestrator 维护和传播
- 仅在 Kernel 启动时作为"参数"注入
- 或通过轻量级广播发送（避免走数据通道）

**实现机制**：
- **JAX**：作为 JIT 的 static 参数，或使用 `jax.sharding.NamedSharding(Replicated)`
- **PyTorch**：控制变量已存在于每个 Python 进程的内存中（SPMD-on-Process），直接作为函数参数传入
- **框架职责**：确保在多机启动时，这些参数在所有 Rank 上初始化一致

4.2.3 Type Router（类型路由器）
框架在 SPMD 的 function call 抽象层实现类型路由，自动将输入参数分流到不同的传输平面。

**路由规则**：
1. 检查输入参数的类型和 Sharding 属性
2. **Sharded Tensor** → Data Plane（ULP 协议）
3. **Global Scalar / Config** → Control Plane（Orchestrator 传播）
4. **Synchronized Result** → Control Plane（聚合后传播）

**用户接口**：
- 用户只需定义统一的函数签名
- 框架自动根据类型注解进行路由
- 无需显式调用不同的 call 方法

4.3 ULP 协议 (Universal Layout Protocol)
**适用范围**：仅用于 Data Plane（Path A），处理 Worker 独立/异步数据的传输

4.3.1 概述
ULP 是一种高性能、去 Python 化、与框架无关（Framework-Agnostic）的分布式数据传输范式。其核心目标是统一 Dense（规则切分）、Sparse（Hash/MoE 切分）和 Replicated（广播）的传输逻辑，通过向量化算子在发送端静态计算路由，并通过票据计数（Ticket Barrier）在接收端实现零同步开销的组装。
核心特性：
统一传输范式：统一处理 Dense、Sparse 和 Replicated 三种数据传输模式
向量化路由计算：在发送端使用向量化算子静态计算路由，避免 Python 循环
零同步开销：接收端采用票据计数机制，实现无锁、O(1) 复杂度的数据组装
框架无关：与底层深度学习框架（JAX/Torch/NumPy）解耦
弱状态机：Ticket Barrier是一种单worker私有的弱状态机
4.2 核心数据结构
4.2.1 发送端视图：LayoutSpec
LayoutSpec 定义如何将数据映射到物理节点。所有字段均为静态 Native Array（Numpy/JAX/Torch Tensor），禁止使用 Python List/Dict。
字段说明：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`op_type`** | `PartitionOp (IntEnum)` | 决定切分算法的枚举值（`DIV_MOD`、`HASH_MOD`、`LOOKUP`、`REPLICATE`） |
| **`op_params`** | `Array<Int>` | 算子所需的静态参数（如 `[divisor, modulus]`） |
| **`topology_map`** | `Array<Int> (Optional)` | 逻辑分区 (PID) -> 物理 Rank 的映射表。用于解耦数据切分与物理部署（处理 Dense 多级切分的关键） |
| **`target_ranks`** | `Array<Int> (Optional)` | 仅用于 `REPLICATE` 模式，指定目标节点列表 |

关键设计：
所有字段必须是静态 Native Array，禁止使用 Python 动态数据结构
topology_map 用于处理多级切分场景，将逻辑分区映射到物理节点
在训练开始前一次性生成，运行时只读
4.2.2 接收端视图：DependencySpec
DependencySpec 定义当前节点在某个 Batch 中需等待的信号量。
字段说明：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`expected_shards`** | `Int` | 票据阈值。通常等于上游并行度（Sender World Size） |

关键设计：
接收方不关心数据来自哪个发送方，只关心收到了多少次（票据计数）
实现了接收端的去中心化，无需维护复杂的发送方列表
4.3 算子定义 (Partition Operators)
所有路由计算必须通过底层框架（JAX/Torch/NumPy）的向量化指令执行，严禁使用 Python 循环。
PartitionOp 枚举
DIV_MOD (Dense/Slicing)
公式: PID = (val // \text{params}[0]) \% \text{params}[1]
场景: Data Parallel, Tensor Parallel, Sequence Sharding.
HASH_MOD (Sparse/Embedding)
公式: PID = \text{hash}(val) \% \text{params}[0]
场景: Embedding Lookup, Parameter Server.
LOOKUP (MoE/Irregular)
公式: PID = \text{params}[val]
场景: MoE Router 结果分发（Params 即为 Routing Table）。计算出Index当成PID直接返回是它的一种特例。
REPLICATE (Broadcast)
公式: PID = \text{All Target Ranks}
场景: 广播全局变量，模型同步。
实现要求：
所有算子必须使用向量化实现（如 jnp.argsort、torch.gather 等）
禁止使用 Python for 循环
计算复杂度由 C++/CUDA 承担，Python 仅负责 Launch Kernel
4.4 发送端工作流 (Sharded Sender)
发送端工作流遵循以下原则：计算复杂度由 C++/CUDA 承担，Python 仅负责 Launch Kernel。
4.4.1 计算目标 (Route Compute)
步骤：
根据 LayoutSpec 执行向量化算子，计算每个元素的 TargetRank
输出：Tensor<Int32> [N]，表示每个元素的目标 Rank ID
实现：
根据 op_type 选择对应的向量化算子
使用 op_params 作为算子参数
如果存在 topology_map，将逻辑 PID 映射到物理 Rank
4.4.2 分组与重排 (Group & Sort)
步骤：
执行 argsort(TargetRank) 对数据索引进行重排，使相同 TargetRank 的元素连续
执行 unique_consecutive 或 bucketize 获取切分点（Split Offsets）
输出：有序的 DataView 和 IndexView
目的：
将相同目标的数据聚合在一起，减少通信次数
为后续的数据切片和发送做准备
4.4.3 空包生成 (Empty Packet Generation)
关键协议：
遍历所有潜在的 TargetRank
如果某 Rank 在本 Batch 中分配到的数据量为 0，必须构造一个 Len=0 的 Payload
目的：防止接收端死锁（Deadlock Prevention）
重要性：
确保接收端的 Ticket Barrier 计数器永远能达到 limit
即使某个发送方没有数据要发送，也必须发送空包
4.4.4 RPC 发射 (Launch Async)
步骤：
对每个 Target Rank 发起异步 RPC 调用
发送对应分片的数据（或空包）
不等待响应，继续处理下一个批次
优化：
使用异步 RPC，实现数据传输与计算的流水线
支持批量发送，减少 RPC 调用开销
4.5 接收端工作流 (Ticket Barrier Receiver)
接收端工作流遵循以下原则：无锁（Lock-free）、无复杂消息交互、O(1) 复杂度。
4.5.1 状态初始化 (Register)
步骤：
在计算图构建或 Batch 开始时，建立状态条目：
State[BatchID] = {
    count: 0,
    shards: [],
    limit: DependencySpec.expected_shards
}
说明：
每个 Batch 有独立的状态条目
limit 来自 DependencySpec，表示期望接收的分片数量
count 为原子计数器，记录已收到的分片数
4.5.2 接收函数 (On Recv)
步骤：
原子计数：State[BatchID].count += 1
数据存储：如果 Payload.len > 0，存入 shards 列表（Zero-Copy Reference）
屏障触发检查：检查是否满足触发条件
关键设计：
使用原子操作更新计数器，避免锁开销
非空数据使用零拷贝引用，避免不必要的内存复制
不关心发送方 ID，只关心计数
4.5.3 屏障触发 (Barrier Trigger)
触发条件：
if count == limit：所有分片已接收完成
触发操作：
Finalize：将 shards 列表拼接（Concat）为完整的 JAX/Numpy Array
Execute：触发下游计算任务
Clean：删除 State[BatchID]
等待状态：
else：返回（不做任何事），继续等待更多分片
性能特点：
O(1) 复杂度的屏障检查
无锁实现，避免线程阻塞
仅在触发时进行一次数据拼接
4.6 通信载荷 (RPC Payload)
传输过程应保持极致轻量，仅包含必要数据。
RPCPayload 结构：
```python
class RPCPayload:
    batch_id: str           # 唯一批次标识
    data: Array             # 数据分片 (Tensor)
    routing_indices: Array   # 原始全局索引 (Tensor, 用于 Sparse->Dense 还原顺序)
    # 无需包含 sender_id 或 complex metadata
```
设计原则：
最小化元数据，减少传输开销
routing_indices 用于支持 Sparse 数据的顺序还原
不包含发送方 ID，实现接收端的去中心化
4.7 性能优化 Checklist
关键优化点：
 Zero-Copy：发送端使用 View/Slice 而非 Copy；接收端仅在 Barrier 触发时进行一次 Concat
 Static IndexDescriptor：LayoutSpec 和 DependencySpec 应在训练前一次性生成，运行时只读
 Vectorized Implementation：确保路由计算使用 jnp.argsort、jnp.take、torch.gather 等原语
 Keep-Alive：严格执行"空包保活"策略，确保 Barrier 计数器永远能达到 limit

5. 关键流程详解
5.1 训练启动流程

```
1. 用户提交FSM定义（包含条件分支和触发条件）
   ↓
2. Main Orchestrator解析FSM
   ├─ 解析状态转换图
   ├─ 解析触发条件
   └─ 构建条件分支映射
   ↓
3. Scheduler生成资源分配计划
   ├─ 为条件分支创建共享Placement Group
   ├─ 评估资源需求
   └─ 确定抢占优先级
   ↓
4. Orchestrator启动各个SPMD
   ├─ 创建/复用Placement Group（条件分支共享）
   ├─ 检查Checkpoint（如果存在则恢复）
   ├─ 启动SPMD Master（Ray Actor）
   └─ 启动SPMD Workers（Ray Actors）
   ↓
5. 建立Worker之间的逻辑Handle映射
   ├─ 创建逻辑Handle
   ├─ 建立逻辑Handle到实际Handle的映射
   └─ 注册Handle变更监听
   ↓
5. 初始化State Backend
   ↓
6. 开始训练循环（根据触发条件动态选择执行路径）
```

5.2 单Step执行流程
```
Step K (K % accumulation_steps != 0, 异步执行):
SPMD1 Worker Ni:
  1. execute_forward(K) → 执行前向
  2. next_forward_call(K) → 异步发送到SPMD2
  3. [无需等待] 继续执行 execute_forward(K+1)
SPMD2 Worker Mj:
  1. on_forward_called(K) → 接收数据
  2. execute_forward(K) → 执行前向
  3. execute_backward(K) → 执行反向
  4. next_backward_call(K) → 异步发送梯度到SPMD1
SPMD1 Worker Ni:
  5. on_backward_called(K) → 接收梯度
  5. execute_backward(K) → 执行反向
  6. [累积梯度，不更新参数]
Step K+1 (K+1 % accumulation_steps == 0, 需要同步):
SPMD1 Worker Ni:
  1. [等待] SPMD Locker.wait_for_backward_complete(K)
  2. execute_forward(K+1)
  3. ...
  4. execute_backward(K+1)
  5. update_parameters() → 更新参数
  5. SPMD Locker.release_forward_lock(K+2)
```
5.3 容错与恢复流程
5.3.1 Worker 故障恢复
```
1. State Manager或Main FSM定时发动保存Checkpoint
   ↓
2. 检测到Worker故障
   ↓
3. SPMD Master报告异常状态
   ↓
4. Main Orchestrator决定恢复策略
   ├─ 重启失败的Worker
   ├─ 从Checkpoint恢复状态
   └─ 重新建立连接
   ↓
5. 继续训练
```
5.3.2 节点驱逐与恢复流程
```
1. 资源不足，触发抢占机制
   ↓
2. Scheduler根据优先级选择要驱逐的节点
   ↓
3. Orchestrator触发节点状态保存
   ├─ 保存节点参数
   ├─ 保存优化器状态
   ├─ 保存训练进度
   └─ 生成Checkpoint ID
   ↓
4. Orchestrator驱逐节点（基于Ray API）
   ├─ 停止Ray Actor
   ├─ 释放Placement Group资源
   └─ 标记Handle为失效
   ↓
5. 资源释放，新节点启动
   ↓
5. 触发条件满足，需要恢复被驱逐的节点
   ↓
6. Orchestrator从Checkpoint恢复节点
   ├─ 加载节点状态
   ├─ 重新创建Ray Actor
   ├─ 恢复Placement Group
   └─ 更新Handle映射
   ↓
7. 继续执行被中断的操作
```
5.3.3 Handle 变更处理流程
```
1. 节点调用逻辑Handle
   ↓
2. Handle映射层查找实际Handle
   ↓
3. 检测到Handle失效（节点被驱逐）
   ↓
4. 触发节点恢复流程
   ├─ 获取最新Checkpoint ID
   ├─ 恢复节点状态
   └─ 更新Handle映射
   ↓
5. 重新调用实际Handle
   ↓
5. 如果调用失败，重试（最多3次）
   ↓
6. 成功执行或报告错误
```

6. 实现细节
6.1 Ray Actor Handle 管理
6.1.1 逻辑 Handle 和实际 Handle 映射
设计原则：
逻辑 Handle：提供稳定的接口，不随节点变更而改变
实际 Handle：对应底层 Ray Actor，可能因驱逐和重建而变更
自动映射：调用逻辑 Handle 时自动路由到当前实际 Handle
故障恢复：检测到节点失效时自动触发恢复流程

6.1.2 Worker Handle 封装
```python
class WorkerHandle:
    """Worker的Handle封装（逻辑Handle接口）"""
    def __init__(self, logical_id: str, spmd_id: str, worker_id: str, 
                 Handle_mapping: HandleMapping):
        self.logical_id = logical_id
        self.spmd_id = spmd_id
        self.worker_id = worker_id
        self.Handle_mapping = Handle_mapping
    
    def receive_forward_data(self, step: int, data: Tensor, source_id: str):
        """远程调用：接收前向数据（自动处理Handle变更）"""
        return self.Handle_mapping.call_with_retry(
            self.logical_id, 'receive_forward_data', step, data, source_id
        )
    
    def receive_backward_data(self, step: int, grad: Tensor, source_id: str):
        """远程调用：接收反向数据（自动处理Handle变更）"""
        return self.Handle_mapping.call_with_retry(
            self.logical_id, 'receive_backward_data', step, grad, source_id
        )
    
    def mark_forward_complete(self, step: int, source_id: str):
        """远程调用：标记前向完成（自动处理Handle变更）"""
        return self.Handle_mapping.call_with_retry(
            self.logical_id, 'mark_forward_complete', step, source_id
        )
    
    def mark_backward_complete(self, step: int, source_id: str):
        """远程调用：标记反向完成（自动处理Handle变更）"""
        return self.Handle_mapping.call_with_retry(
            self.logical_id, 'mark_backward_complete', step, source_id
        )
```
6.2 数据传输优化
批量传输：将多个小张量合并为一次传输
压缩：对于稀疏数据使用压缩算法
流水线：数据传输与计算重叠
负载均衡：动态调整Sharding策略
6.3 Ray内存管理？
缓冲区管理：限制同时缓存的Python Object数量
及时释放：step完成后立即释放数据

7. 扩展性设计
7.1 支持更多Sharding策略
框架设计支持插件式的Sharding策略扩展：
实现 ShardingSpec 接口
注册到框架中
在配置中指定使用
7.2 支持多种传输后端
当前使用 Ray Actor Handle，未来可以扩展：
UCX RPC
7.3 支持动态图
当前设计主要针对静态图，可以扩展支持：
动态图执行
条件分支
循环结构

8. 总结
Fluxway 框架通过直接 SPMD-to-SPMD 传输机制、异步执行和灵活的状态管理，实现了高效的大规模分布式训练。框架采用分层架构设计，职责清晰，易于扩展和维护。
8.1 核心优势
高效的传输机制：避免中间聚合，减少通信开销
异步执行：提升计算资源利用率
灵活的状态管理：支持多种同步和异步场景
可扩展设计：支持多种Sharding策略和传输后端
8.2 下一步工作
实现核心组件
添加单元测试和集成测试
性能优化和调优
文档和示例完善