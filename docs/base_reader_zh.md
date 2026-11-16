# 读取接口

[英文版](base_reader.md)

`BaseReader` 将来源记录转换成带有全局行坐标的 `TransportTensor`。受控读取器只读取
`self.ctx.source_work_items` 中的精确区间。规划器固定记录顺序与逻辑批次，
`LogicalWorkLedger` 管理租约与进度，分发器选择工作进程并路由数据。

## 读取已分配的记录

每次受控读取前，运行时检查当前训练步的 `SourceWorkItem` 非空元组。来源身份、结构和
全局批次大小必须一致，重复区间直接报错。读取器使用任务中固定的修订和存储定位符（storage locator），
不能临时换成最新修订。

下面的本地示例手动提供运行时通常会注入的上下文。一个读取器持有同一任务中两个
分开的区间。这里构造任务只是为了说明坐标，不代表已经注册来源或启动分布式执行。

```python
import numpy as np

from fluxway.common.decorators import operator, operator_context
from fluxway.common.ingestion import (
    LayoutUnitDescriptor,
    SourceWorkAssignment,
    SourceWorkItem,
)
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.configs import OperatorContext
from fluxway.common.types.tensor import TransportTensor
from fluxway.reader import BaseReader


class FeatureReader(BaseReader):
    @operator("read")
    def read(self) -> tuple[TransportTensor, TransportTensor, TransportTensor]:
        if self.ctx is None or not self.ctx.source_work_items:
            raise RuntimeError("acquired source work is required")
        revisions = self.static_args["revision_rows"]
        rows = np.concatenate(
            [
                revisions[
                    (
                        item.assignment.descriptor.revision_identity,
                        item.assignment.descriptor.storage_locator,
                    )
                ][item.logical_start : item.logical_end]
                for item in self.ctx.source_work_items
            ]
        )
        return (
            self.create_tensor(rows, uniform_dense_partition=True),
            self.create_row_sparse_tensor(rows.tolist(), width=2, dtype=np.int64),
            self.create_tensor(
                [row[: index % 2 + 1].tolist() for index, row in enumerate(rows)],
                dtype=np.int64,
            ),
        )


set_backend("hybrid")
unit = LayoutUnitDescriptor(
    unit_id="part-0:2:8",
    revision_identity="revision-7",
    cardinality=6,
    cardinality_metric="rows",
    split_policy="SplittableSpan",
    locality_key="part-0",
    storage_locator="part-0",
    logical_start=2,
    opaque_token=b"pinned-array",
)
assignment = SourceWorkAssignment.create(
    epoch=0,
    step=3,
    source_identity="features",
    view_identity="view-7",
    schema_digest="two-columns",
    lineage_digest="revision-7",
    descriptor=unit,
    global_start=24,
    global_batch_size=8,
    global_valid_count=8,
)
context = OperatorContext(
    physical_rank=1,
    physical_world_size=4,
    logical_rank=1,
    logical_world_size=4,
    step=3,
    source_work_items=(
        SourceWorkItem(assignment, 2, 4),
        SourceWorkItem(assignment, 6, 8),
    ),
)
revisions = {("revision-7", "part-0"): np.array([[i, i + 1] for i in range(10)])}
with operator_context(
    static_args={"revision_rows": revisions},
    rank=1,
    world_size=4,
    operator_ctx=context,
):
    dense, sparse, nested = FeatureReader.read()

assert dense.global_shape == sparse.global_shape == (8, 2)
assert dense.origin_indices.materialize().tolist() == [24, 25, 28, 29]
assert sparse.origin_indices.materialize().tolist() == [24, 25, 28, 29]
assert nested.origin_indices.materialize().tolist() == [24, 25, 28, 29]
assert nested.logical_shape == (4, None)
assert nested.global_shape is None
```

读取的文件行号为 `2、3、6、7`，全局行号为 `24、25、28、29`。物理进程编号 `1`
不决定这两组坐标。接入存储时应根据原始来源编码解析区间；文件片段的局部下标
不一定等于文件内的物理行号。嵌套列表保留行坐标并报告本地不定长结构，不推测全局
不定长形状。

## 张量辅助方法

| 接口 | 用途 |
| --- | --- |
| `create_tensor(data, dtype=None, uniform_dense_partition=False)` | 封装密集或嵌套数据，并附上来源坐标。 |
| `create_row_sparse_tensor(rows, *, width, dtype=None)` | 编码每行数量可变的值，并声明行宽。 |
| `self.ctx.source_work_items` | 本次调用的精确区间，不能跨调用缓存。 |
| `self.rank`, `self.world_size` | 查看物理放置，不用于推导来源所有权。 |

数据外层行数必须等于已领取区间的记录数之和，所有条目必须属于当前训练步和同一
逻辑批次。均匀密集和行稀疏辅助方法使用任务的 `global_batch_size` 作为全局行数。
缺少上下文、行数错误或元数据冲突时直接抛出异常。

没有任务条目、但由调用方显式协调的来源必须提供 `global_source_offset`。下面是
这种模式的独立本地示例：

```python
import numpy as np

from fluxway.common.decorators import operator, operator_context
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.configs import OperatorContext
from fluxway.common.types.tensor import TransportTensor
from fluxway.reader import BaseReader


class ExplicitReader(BaseReader):
    @operator("read")
    def read(self) -> TransportTensor:
        return self.create_tensor(
            self.static_args["rows"], uniform_dense_partition=True
        )


set_backend("hybrid")
context = OperatorContext(
    physical_rank=3,
    physical_world_size=4,
    logical_rank=1,
    logical_world_size=2,
    step=3,
    global_source_offset=14,
)
with operator_context(
    static_args={"rows": np.array([[10.0], [20.0]])},
    rank=3,
    world_size=4,
    operator_ctx=context,
):
    tensor = ExplicitReader.read()

assert tensor.global_shape == (4, 1)
assert tensor.origin_indices.materialize().tolist() == [14, 15]
```

这里的全局行数等于本地行数乘以 `logical_world_size`。调用方负责协调这些显式偏移；
这种模式本身不提供检查点重放，也不保证拓扑变化后的所有权。

## 读取已编译布局

`IngestionReadSession.read` 接受 `ScheduledSegmentRead` 和显式 `batch_size`，为请求区间
提供 `SegmentReadChunk`。每块数据记录实际行数、物理字节数，以及是否打开了对象。
`IngestionLayoutSession` 在线程中推进阻塞迭代器，并验证总记录数。

下面的集成函数需要运行时发出的激活信息、存储读取会话和异步消费函数。全部读完后
才能返回布局完成信息；提前退出或记录数不符时不能报告完成。

```python
from collections.abc import Awaitable, Callable
from typing import Any

from fluxway.common.ingestion import (
    IngestionLayoutActivation,
    SourceLayoutCompletion,
)
from fluxway.reader import IngestionLayoutSession, IngestionReadSession


async def consume_layout(
    activation: IngestionLayoutActivation,
    reader: IngestionReadSession,
    consume: Callable[[Any], Awaitable[None]],
    *,
    batch_size: int,
) -> SourceLayoutCompletion:
    session = IngestionLayoutSession(activation, reader, batch_size=batch_size)
    try:
        while True:
            batch = await session.read()
            if batch is None:
                return session.acknowledgement
            await consume(batch)
    finally:
        await session.close()
```

布局完成信息与 `SourceWorkAcknowledgement` 不同。正常受控执行由工作进程领取租约、
注入任务条目、调用读取器，再通过有界异步队列确认租约。应用不能替其他工作进程
确认整个描述区间。

```mermaid
sequenceDiagram
    participant A as 异步消费方
    participant S as IngestionLayoutSession
    participant T as 读取线程
    participant R as IngestionReadSession
    loop 直到完整耗尽
        A->>S: await read()
        S->>S: 取得会话异步锁
        S->>T: to_thread(next)
        T->>R: 推进阻塞区间迭代器
        R-->>T: SegmentReadChunk
        T->>T: 校验实际记录数与区间覆盖
        T-->>S: 本批数据或流结束
        alt 仍有数据
            S-->>A: batch
            A->>A: await consume(batch)
        else 全部区间耗尽且计数一致
            S->>S: 创建 SourceLayoutCompletion
            S-->>A: None
        end
    end
    A->>S: await close()
    S->>T: to_thread(stream.close)
    T->>R: close()
    S-->>A: 读取资源已关闭
```

提前退出或读取异常不会生成布局完成信息；计数不符直接报错。调用方需等待消费完成并在 `finally` 中关闭会话，不能把 `close()` 当作租约确认或检查点提交。

## 来源注册与恢复

来源适配器实现 `IngestionSourceAdapter.scan_next(cursor, budget)` 和纯位置运算
`advance_cursor(cursor, records)`，并提供 `close()`，即使没有外部资源也必须实现。
来源描述提供 `build_source_scan_binding` 与 `build_component`。控制器将绑定创建和缓存
交给规划器，统一保存读取拓扑、适配器、尾部策略和初始不透明位置；绑定成功后由规划器
负责适配器关闭，首次扫描前取消等待也不会丢弃资源。构造、初始位置、扫描和关闭可返回
可等待对象，在适配器专用通道执行；位置运算保持同步。工作进程的 `IngestionReadSession.close()`
仍是独立的同步读取会话契约，不替代控制侧适配器关闭。可运行的扫描与生命周期示例见[来源协调](source_coordination_zh.md)，
完整接入见[推荐来源描述](../fluxway/example/recommendation/source_adapter.py)与
[读取器](../fluxway/example/recommendation/reader.py)。

来源确认只表示生产完成，不代表模型已经消费或状态已经持久化。规划器按共同检查点
的切点生成消费书签，恢复后从书签重新规划；读取器不另存独立持久位置，也不恢复旧窗口。
外部副作用需要独立事务或幂等契约。仅实现读取器不能证明补充窗口、整图排空和恢复
流程已经完整接通，这些边界由[来源协调](source_coordination_zh.md)定义。

## 高性能执行范式与流水线重叠

`BaseReader` 作为 Source SPMD，是 MPMD 计算图的一等生产者节点，完全复用框架生命周期管理、步进编排和分布式数据面能力。

```mermaid
flowchart LR
    subgraph ControlLane["控制面泳道 (AsyncIO Loop)"]
        direction TB
        Sched["步进准入调度器"]
        Mailbox["数据面邮箱与网络"]
    end

    subgraph WorkerLanes["受控工作线程泳道 (WorkerProcess Threads)"]
        direction TB
        IO["来源数据读取 (I/O)"]
        ArrowConcat["BaseReader.concatenate_batches"]
        ArrowParse["BaseReader.create_tensors_from_batch"]
        Tensor["TransportTensor 物化"]
        IO --> ArrowConcat --> ArrowParse --> Tensor
    end

    subgraph DownstreamLanes["下游算子计算泳道 (InputSPMD)"]
        direction TB
        AsyncUnique["GPU/网络通信 async_unique"]
        CpuFinalize["同步计算算子 (compute 泳道)\n密集特征与张量物化"]
        AsyncUnique --> CpuFinalize
    end

    Sched -- "pipelined(depth=K) 并发派发" --> IO
    Tensor -- "跨节点流水发送" --> Mailbox
    Mailbox -- "输入邮箱交付" --> AsyncUnique
```

### 1. 同步执行契约与非阻塞泳道隔离
* **普通同步函数（`def read(self)`）**：用户编写纯粹的同步业务逻辑。运行时调度器（`UserCallableExecutor`）自动将普通函数分发到 `WORKER_PROCESS` 线程泳道。
* **硬件级多核并行**：PyArrow、NumPy、Numba 等底层计算在进入 C/C++ 核心时会自动释放 GIL，实现真正的多核物理加速，同时彻底解放主事件循环，消除对集群心跳与网络路由的延迟阻塞。
* **同步与异步分层隔离**：普通的同步读取与计算算子在各自的规范泳道（`io` 与 `compute`）中有界并发，C 扩展完全释放 GIL；跨节点异步通信在 `async def` 协程中进行，主事件循环零阻塞。

### 2. 零成本流水预取
* 无需额外实现专门的 `Prefetcher`、队列或后台线程环形缓冲区。
* 直接在算子上声明 `async_policy=AsyncPolicy.pipelined(depth=K)`，框架调度引擎会自动前瞻推进 $K$ 个步进的租约获取与数据生产，将已物化的 `TransportTensor` 通过数据面自动存入下游节点的输入邮箱，实现天然的分布式流水线预取。

### 3. 一等公民的 Arrow 批处理物化与拼接

`BaseReader` 内建了开箱即用的大规模特征物化与拼接能力，无需在业务代码中反复手写 Arrow 到 NumPy/Tensor 的转换逻辑：

| 接口 | 输入类型 | 输出类型 | 特性与底层优化 |
| :--- | :--- | :--- | :--- |
| `create_tensors_from_batch(batch, ...)` | `pa.Table`, `pa.RecordBatch`, 字典映射 | `dict[str, TransportTensor]` | 自动解析全局坐标与批次维度；密集列自动转张量；List/LargeList 变长列自动提取 offsets 生成带 `nested_splits` 的稀疏张量。 |
| `create_tensor(data, ...)` | `pa.Array`, `pa.ChunkedArray`, `ndarray`, `list` | `TransportTensor` | 统一抽象，支持原生 Arrow 数组零拷贝解析，支持指定 `dtype` 与均匀密集切分标记。 |
| `concatenate_batches(batches)` | `Sequence[pa.Table]`, `Sequence[RecordBatch]`, `Sequence[Mapping]` | `pa.Table` 或拼接字典 | 范围读取器（Range Worker）与分片合并的首选，采用 Permissive Schema 快速合并，避免反复内存碎片化。 |

```python
# 示例：直接在 Reader 中消费 Arrow Table
class ParquetRecommendationReader(BaseReader):
    @operator("read", async_policy=AsyncPolicy.pipelined(depth=4))
    def read(self) -> dict[str, TransportTensor]:
        table: pa.Table = self.load_next_parquet_table()
        # 自动将稠密与非均匀 List 列转换为标准的 TransportTensor 映射
        return self.create_tensors_from_batch(
            table,
            column_dtypes={"user_id": np.int32, "item_id": np.int32},
            uniform_dense_partition=True,
        )
```

### 4. 运行时范式对照表

| 算子/任务编写模式 | 声明形式 | 运行物理宿主 | 适用场景 | 架构优势 |
| :--- | :--- | :--- | :--- | :--- |
| **同步来源算子 (IO)** | `def read(self)` (继承自 `BaseReader` 或 `is_source=True`) | `WORKER_PROCESS` 线程池 (`io` 泳道) | 来源读取、数据解包、磁盘拉取、Arrow 解析 | 编译期自动归入 `io` 泳道，零异步样板代码，C 扩展完全释放 GIL。 |
| **同步计算算子 (Compute)** | `def op(self)` | `WORKER_PROCESS` 线程池 (`compute` 泳道) | 密集矩阵计算、特征交叉、模型前向、张量处理 | 编译期自动归入 `compute` 泳道，与 `io` 自动配齐协同优先级（`compute` 优先执行）。 |
| **纯异步通信算子** | `async def op(self)` | `COROUTINE_ASYNC` 事件循环 (无泳道) | 跨节点通信调度、异步分布式搜寻、事件流驱动 | 极致上下文切换性能，高并发 IO，不占用工作线程池。 |


