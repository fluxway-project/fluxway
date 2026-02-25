import fluxway
import torch
from typing import Optional

from fluxway import FunctionalGraph

## 用户构建 graph 的相关 api 示例
# 1. 定义数据类型：
#   ActionState1/ActionState2/Gradient2/ActionState3/Gradient3 均为用户自定义的数据传输类，
#   通过注解 @flux.data_type ，来定义参数类型，包含两种类型：
#   其中 state_argument 表示状态参数，tensor_argument 表示张量参数。分别对应两种传输模式，对应文档 ARCHITECTURE.md 3.2.2 所说的双流汇聚传输策略。
#   state_argument 指定的参数通过控制流传递，tensor_argument 指定的参数通过数据流传递。


@fluxway.data_type(
    state_argument=["id"], tensor_argument=["tensor"], static_argument=["static"]
)
class ActionState1:
    id: int
    tensor: torch.Tensor


@fluxway.data_type(state_argument=["id"], tensor_argument=["tensor"])
class ActionState2:
    id: int
    tensor: torch.Tensor


@fluxway.data_type(state_argument=["id"], tensor_argument=["tensor"])
class Gradient2:
    id: int
    tensor: torch.Tensor


@fluxway.data_type(state_argument=["id"], tensor_argument=["tensor"])
class ActionState3:
    id: int
    tensor: torch.Tensor


@fluxway.data_type(state_argument=["id"], tensor_argument=["tensor"])
class Gradient3:
    id: int
    tensor: torch.Tensor


# 2. 定义 SPMD
#   SPMD1/SPMD2/SPMD3 均为用户自定义的 SPMD 模块，
#   通过注解 @flux.spmd，来定义 SPMD 模块，包含四种类型：
#   其中 ：
#       resource 表示资源需求，
#       sharding 表示数据切分，
#       mesh 表示计算网格，
#       max_async_steps 表示相同 spmd 的不同 worker 间，最大的异步步数，默认为 1，
#       params_update_steps 表示参数更新步数，每执行 params_update_steps 步，前向就会阻塞，直到所有的反向完成后才继续。默认为 1。
#
#   此外，SPMD 模块需要实现 forward 和 backward 两个方法，分别对应前向和反向传播。缺省则表示没有对应的操作，如 SPMD1 是数据源，没有 backward 操作。
#   forward 和 backward 的输出，即为需要在 SPMD 之间传输的数据。
#   如果是用 @flux.data_type 注解的数据类型，则会根据注解的 state_argument 和 tensor_argument 来确定传输模式，
#   state_argument 指定的参数通过控制流传递，tensor_argument 指定的参数通过数据流传递。
#   如果不是用 @flux.data_type 注解的数据类型，则默认通过控制流传递。


@fluxway.spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 1, "cnt": 1},
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(1,),
    max_async_steps=10,
    params_update_steps=1,
)
class DataSource:
    pass


def read_from_datasource(ds: DataSource) -> ActionState1:
    return ActionState1(id=1, tensor=torch.tensor([1.0]))


class SPMD1:
    @fluxway.operator(name="produce")
    def produce(self, datasource: DataSource) -> ActionState1:
        return read_from_datasource(datasource)


@fluxway.spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 1, "cnt": 1},
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(1,),
)
class SPMD2:
    @fluxway.operator(name="compute")
    def compute(self, state: ActionState1) -> ActionState2:
        return ActionState2(id=state.id, tensor=state.tensor)

    @fluxway.operator(name="optimize")
    def optimize(self, state: ActionState2) -> Optional[Gradient2]:
        pass


@fluxway.spmd(
    resource={"nums_cpu": 4, "nums_gpu": 2, "memory": 16, "cnt": 1},
    mesh_axis_names=("dp", "tp"),
    mesh_axis_shapes=(4, 2),
)
class SPMD3:
    @fluxway.operator(
        name="compute",
        input_sharding={
            "id": {"type": "dense", "axis": [None]},
            "tensor": {"type": "dense", "axis": ["dp"]},
        },
        output_sharding={
            "id": {"type": "dense", "axis": [None]},
            "tensor": {"type": "dense", "axis": ["dp"]},
        },
    )
    def compute(self, state: ActionState1) -> ActionState3:
        return ActionState3(id=state.id, tensor=state.tensor)

    @fluxway.operator(name="optimize")
    def optimize(self, state: ActionState3) -> Optional[Gradient3]:
        pass


def check(state: ActionState1) -> bool:
    return state.id == 1


# 使用 FunctionalGraph 构造 graph
fg = FunctionalGraph()

# 注册节点
node1 = fg.node("Node1", SPMD1)
node2 = fg.node("Node2", SPMD2)
node3 = fg.node("Node3", SPMD3)

# 假设 datasource 是通过某种方式注入的或者在 SPMD1 内部处理
# 这里简化为直接调用
h1 = node1.produce(datasource=DataSource())

# 条件分支
h2 = node2.compute(h1, condition_fn=check)
h3 = node3.compute(h1, condition_fn=lambda state: not check(state))

# 反向链路（优化）
node2.optimize(h2)
node3.optimize(h3)

# 自循环或其他复杂拓扑
node2.optimize(h2)  # SPMD2 反向触发自身的下一次操作等（示意）

graph = fg.compile("functional_test")
