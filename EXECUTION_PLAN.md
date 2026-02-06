# _assemble_input 重构执行计划

## 目标
修改 `fluxway/spmd/worker/worker.py:785` 的 `_assemble_input` 方法，在方法内完成输入数据的组装，直接返回 op 期望的 data_type 对象，而不是返回字典。

## 核心设计理念
- **SPMD-to-SPMD all-to-all 传输**：单个 worker 从多个前置 SPMD 的多个 worker 接收 tensor shards
- **框架职责**：只负责单个 worker 的输入组装，不考虑当前 SPMD 的 worker 间交互（这是用户 op 的职责）
- **输入类型预计算**：在 OperatorManager 初始化时，从 operator 函数签名提取参数类型并缓存

## 执行步骤

### 步骤 1: 在 OperatorManager 初始化时预计算输入类型

**位置**: `fluxway/spmd/worker/operator_manager/operator_manager.py`

**任务**:
1. 在 `__init__` 方法中，解析每个 operator 的函数签名
2. 提取第一个参数（排除 self）的类型注解
3. 存储到 `_operator_input_types: Dict[OpType, Type]` 中

**实现要点**:
```python
import inspect
from typing import Type

# 在 __init__ 中添加
self._operator_input_types: Dict[OpType, Optional[Type]] = {
    OpType.FORWARD: None,
    OpType.BACKWARD: None,
    OpType.UPDATE: None,
}

# 在 _parse_operator_metadata 后添加
self._precompute_operator_input_types()
```

### 步骤 2: 重构 _assemble_input 方法

**位置**: `fluxway/spmd/worker/worker.py:785`

**需要完成的5件事**:

#### 2.1 获取 static_args
- 调用 `self._op_manager.get_static_args()` 获取静态参数

#### 2.2 获取 state_args
- 调用 `self._op_manager.get_expected_state_names(op_type)` 获取期望的 state 名称列表
- 从 mailbox 查询: `await self._mailbox.get_all_states(op_type, step)`
- 按名称提取对应的 state 值

#### 2.3 组装 tensor_args（核心算法）

**前提条件**:
- 从 mailbox 获取: `await self._mailbox.get_all_tensor_shards(op_type, step)`
- 返回格式: `{tensor_name: {rank: (data, meta)}}`
- 获取 input_sharding: `self._op_manager.get_input_sharding(op_type)`
- 获取 predecessor_topology: `self._op_manager.get_predecessor_topology(op_type)`
- 获取 input_routing_strategies: `self._op_manager.get_input_routing_strategy(op_type, pre_spmd_id, tensor_name)`

**三种情况的组装算法**:

##### 情况 1: static -> static (Dense-to-Dense)

**算法**:
1. 对于每个 tensor_name:
   - 获取 input_spec: `input_sharding[tensor_name]` (StaticShardingSpec)
   - 遍历所有 predecessor SPMD:
     - 获取 pre_sharding: `pre_info.tensor_sharding[tensor_name]` (StaticShardingSpec)
     - 获取 input_strategy: `_input_routing_strategies[(op_type, pre_spmd_id, tensor_name)]`
     - 从 input_strategy.dependent_ranks 知道哪些前置 rank 会发送数据给当前 rank
   - 从 mailbox 中提取这些 rank 的 shards
   - 按照 input_spec 的 sharding 规则组装（concatenate/stack）

**组装逻辑**:
- 使用 DenseShardingStrategy 的 dim_infos 和 src_coords/dst_coords 信息
- 对于每个维度：
  - 如果 src_num_shards == dst_num_shards: 直接 concatenate
  - 如果 src_num_shards != dst_num_shards: 需要 reshape/transpose 后 concatenate
- 最终得到一个完整的 tensor，形状符合 input_spec 的要求

**实现要点**:
```python
def _assemble_static_to_static_tensor(
    self,
    tensor_name: str,
    shards: Dict[int, Tuple[Any, Optional[TensorMeta]]],
    input_spec: StaticShardingSpec,
    pre_spmd_id: str,
    pre_info: SPMDTopologyInfo,
) -> Any:
    # 获取 input routing strategy
    strategy = self._op_manager.get_input_routing_strategy(
        op_type, pre_spmd_id, tensor_name
    )
    
    # 只取 dependent_ranks 的 shards
    relevant_shards = {
        rank: shards[rank][0] 
        for rank in strategy.dependent_ranks 
        if rank in shards
    }
    
    # 按照 strategy 的 dim_infos 组装
    # 使用 numpy/torch 的 concatenate/stack 操作
    # ...
```

##### 情况 2: dynamic -> static (Sparse-to-Dense)

**算法**:
1. 对于每个 tensor_name:
   - 获取 input_spec: `input_sharding[tensor_name]` (StaticShardingSpec)
   - 遍历所有 predecessor SPMD:
     - 获取 pre_sharding: `pre_info.tensor_sharding[tensor_name]` (DynamicShardingSpec)
     - 从 mailbox 获取所有 rank 的 shards（sparse 数据）
   - 从每个 shard 的 meta.global_indices 复原全局索引
   - 根据 global_indices 和 input_spec，将 sparse 数据组装成 dense tensor

**关键问题**:
- 当前 TensorMeta 有 `global_indices` 字段，但可能不够完整
- 需要知道：
  - 全局 tensor 的形状（从 input_spec 可以推断？）
  - sparse 数据的值（在 data 中）
  - 如何填充到 dense tensor 中

**实现要点**:
```python
def _assemble_dynamic_to_static_tensor(
    self,
    tensor_name: str,
    shards: Dict[int, Tuple[Any, Optional[TensorMeta]]],
    input_spec: StaticShardingSpec,
) -> Any:
    # 收集所有 sparse shards
    all_values = []
    all_indices = []
    
    for rank, (data, meta) in shards.items():
        if meta is None or meta.global_indices is None:
            raise ValueError(f"Missing global_indices in meta for sparse tensor {tensor_name}")
        
        all_values.append(data)
        all_indices.append(meta.global_indices)
    
    # 合并所有 indices 和 values
    # 根据 input_spec 推断全局形状
    # 创建 dense tensor 并填充
    # ...
```

##### 情况 3: to dynamic (Any -> Sparse)

**算法**:
1. 对于每个 tensor_name:
   - 获取 input_spec: `input_sharding[tensor_name]` (DynamicShardingSpec)
   - 遍历所有 predecessor SPMD:
     - 从 mailbox 获取所有 rank 的 shards
   - 直接合并所有 shards（concatenate values）
   - 合并所有 meta（合并 global_indices，保留 source tracking 信息）

**实现要点**:
```python
def _assemble_to_dynamic_tensor(
    self,
    tensor_name: str,
    shards: Dict[int, Tuple[Any, Optional[TensorMeta]]],
) -> Tuple[Any, TensorMeta]:
    # 合并所有 shards 的 data
    all_data = [shards[rank][0] for rank in sorted(shards.keys())]
    merged_data = np.concatenate(all_data, axis=0)  # 或其他合并方式
    
    # 合并所有 meta
    all_metas = [shards[rank][1] for rank in sorted(shards.keys()) if shards[rank][1] is not None]
    
    # 合并 global_indices
    all_indices = []
    for meta in all_metas:
        if meta.global_indices is not None:
            all_indices.append(meta.global_indices)
    
    merged_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    
    # 创建合并后的 meta
    merged_meta = TensorMeta(
        tensor_name=tensor_name,
        shape=merged_data.shape,
        global_indices=merged_indices,
        # 保留 source tracking（取第一个？）
        source_spmd_id=all_metas[0].source_spmd_id if all_metas else "",
        source_rank=all_metas[0].source_rank if all_metas else -1,
        forward_step=all_metas[0].forward_step if all_metas else -1,
        is_sparse=True,
    )
    
    return merged_data, merged_meta
```

#### 2.4 组装 tensor meta
- 对于 static -> static: 从第一个 shard 的 meta 提取（或创建新的）
- 对于 dynamic -> static: 创建新的 meta（因为已经转换为 dense）
- 对于 to dynamic: 合并所有 meta（见情况 3）

#### 2.5 构造最终的 data_type 对象
- 获取预计算的输入类型: `self._op_manager.get_operator_input_type(op_type)`
- 使用 static_args、state_args、tensor_args 构造对象
- 返回构造好的对象（而不是字典）

### 步骤 3: 修改 execute 方法

**位置**: `fluxway/spmd/worker/operator_manager/operator_manager.py:791`

**任务**:
- 恢复为简单的 `return operator(input_data)`
- 因为 `input_data` 已经是正确的 data_type 对象了

## 需要添加的辅助方法

### 在 OperatorManager 中:
1. `get_operator_input_type(op_type: OpType) -> Optional[Type]`
2. `get_input_sharding(op_type: OpType) -> Dict[str, ShardingSpec]`
3. `get_predecessor_topology(op_type: OpType) -> Dict[str, SPMDTopologyInfo]`
4. `get_input_routing_strategy(op_type: OpType, pre_spmd_id: str, tensor_name: str) -> Optional[DenseShardingStrategy]`
5. `_precompute_operator_input_types() -> None`

### 在 Worker 中:
1. `_assemble_static_to_static_tensor(...) -> Any`
2. `_assemble_dynamic_to_static_tensor(...) -> Any`
3. `_assemble_to_dynamic_tensor(...) -> Tuple[Any, TensorMeta]`
4. `_merge_tensor_metas(...) -> TensorMeta`

## 注意事项

1. **Eager 模式**: 如果 `is_eager_trigger(op_type)` 为 True，不需要组装 tensor，直接使用第一个 shard
2. **多前置 SPMD**: 一个 tensor 可能来自多个前置 SPMD，需要合并所有来源
3. **Meta 合并**: 对于 to dynamic 的情况，需要正确合并所有 meta 信息
4. **错误处理**: 如果缺少必要的 shards 或 meta，应该抛出清晰的错误

## 实现顺序

1. 先在 OperatorManager 中添加输入类型预计算
2. 实现 tensor 组装的三种情况（先实现 static->static，再实现其他）
3. 重构 _assemble_input 方法
4. 简化 execute 方法
5. 测试和调试
