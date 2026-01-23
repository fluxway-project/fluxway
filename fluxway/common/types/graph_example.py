## 用户构建 graph 的相关 api 示例
# 1. 定义数据类型：
#   ActionState1/ActionState2/Gradient2/ActionState3/Gradient3 均为用户自定义的数据传输类，
#   通过注解 @flux.data_type ，来定义参数类型，包含两种类型：
#   其中 state_argument 表示状态参数，tensor_argument 表示张量参数。分别对应两种传输模式，对应文档 ARCHITECTURE.md 3.2.2 所说的双流汇聚传输策略。
#   state_argument 指定的参数通过控制流传递，tensor_argument 指定的参数通过数据流传递。

@fluxway.data_type{
    state_argument: ["id"],
    tensor_argument: ["tensor"]
    static_argument: ["static"]
}
class ActionState1:
    id: int
    tensor: torch.Tensor

@fluxway.data_type{
    state_argument: ["id"],
    tensor_argument: ["tensor"]
}
class ActionState2:
    id: int
    tensor: torch.Tensor

@fluxway.data_type{
    state_argument: ["id"],
    tensor_argument: ["tensor"]
}
class Gradient2:
    id: int
    tensor: torch.Tensor

@fluxway.data_type{
    state_argument: ["id"],
    tensor_argument: ["tensor"]
}
class ActionState3:
    id: int
    tensor: torch.Tensor

@fluxway.data_type{
    state_argument: ["id"],
    tensor_argument: ["tensor"]
}
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

@fluxway.spmd{
    resource: {
        nums_cpu: int,
        nums_gpu: int,
        memory: int,
        cnt: int
    },
    mesh: Dict[PartitionType, int]
    input_sharding: Dict[PartitionType, int],
    output_sharding: Dict[PartitionType, int],
    max_async_steps: 10,
    params_update_steps: 1,
}
class SPMD1:
    @fluxway.operator(name="forward")
    def forward(self, datasource: DataSource) -> ActionState1:
        return read_from_datasource(datasource)


@fluxway.spmd{
    resource: {
        nums_cpu: int,
        nums_gpu: int,
        memory: int,
        cnt: int
    },
    mesh: Dict[PartitionType, int]
    max_async_steps: 10,
    params_update_steps: 1,
}
class SPMD2:
    @fluxway.operator(name="forward")
    def forward(self, state: ActionState1) -> ActionState2:
        return ActionState2(id=state.id, tensor=state.tensor)

    @fluxway.operator(name="backward")
    def backward(self, state: ActionState2) -> Optional[Gradient2]:
        pass


class PartitionType(str, Enum):
    """The type of task to run on the vLLM engine."""

    """Generate text."""
    DP = "dp"

    """Generate embeddings."""
    TP = "tp"


@fluxway.spmd(
    resource={
        "nums_cpu": 4,
        "nums_gpu": 2,
        "memory": 16,
        "cnt": 1
    },
    mesh_axis_names=("dp", "tp"),
    mesh_axis_shapes=(4, 2),
    max_async_steps=10,
    params_update_steps=1,
)
class SPMD3:
    @fluxway.operator(
        name="forward",
        input_sharding={
            "id": {"type": "dense", "axis": [None]},  # Replicate
            "tensor": {"type": "dense", "axis": ["dp"]}  # Shard along dp axis
        },
        output_sharding={
            "id": {"type": "dense", "axis": [None]},  # Replicate
            "tensor": {"type": "dense", "axis": ["dp"]}  # Shard along dp axis
            # For sparse sharding, use fn:
            # "tensor": {"type": "sparse", "axis": ["dp"], "fn": my_routing_fn}
            # where fn(tensor, tensor_meta, post_spec) -> List[(slice, rank)]
        }
    )
    def forward(self, state: ActionState1) -> ActionState3:
        return ActionState3(id=state.id, tensor=state.tensor)

    @fluxway.operator(name="backward")
    def backward(self, state: ActionState3) -> Optional[Gradient3]:
        pass

def check(state: ActionState1) -> bool:
    return state.id == 1

def check1(state: ActionState1) -> bool:
    return state.id != 1

# 构造器方法构造 graph
builder = Builder()

builder.add_node("SPMD1", SPMD1, "description1")
builder.add_node("SPMD2", SPMD2, "description2")
builder.add_node("SPMD3", SPMD3, "description3")

# 定义 SPMD1 和 SPMD2、SPMD3 之间的前向条件边
builder.add_forward_conditional_edge("SPMD1", check, {
    True: "SPMD2",
    False: "SPMD3"
})
# 等价于
builder.add_forward_conditional_edge("SPMD1", check, {
    True: "SPMD2"
})
builder.add_forward_conditional_edge("SPMD1", check, {
    False: "SPMD3"
})
# 等价于
builder.add_forward_conditional_edge("SPMD1", check, "SPMD2")
builder.add_forward_conditional_edge("SPMD1", check1, "SPMD3")

# 定义 SPMD2 和 SPMD1 之间的反向条件边
builder.add_backward_conditional_edge("SPMD2", None, "SPMD1")
# 等价于
builder.add_backward_edge("SPMD2", "SPMD1")

# 定义 SPMD3 和 SPMD1 之间的反向条件边
builder.add_backward_conditional_edge("SPMD3", None, "SPMD1")
# 等价于
builder.add_backward_edge("SPMD3", "SPMD1")

builder.add_backward_edge("SPMD2", "SPMD2")
builder.add_backward_edge("SPMD3", "SPMD3")

builder.build(name="test")

class Graph:
    name: str
    nodes: List[Node]
    edges: List[Edge]

class Node:
    name: str
    description: str
    resource: ResourceSpec
    spmd_config: SPMDConfig
    forward_edges: Optional[List[Edge]]
    backward_edges: Optional[List[Edge]]

class Edge:
    source_node: str
    target_node: str
    condition_fn: Optional[Callable[..., bool]]

class SPMDConfig:
    sharding: ShardingSpec
    mesh: MeshConfig
    user_forward_fn: Optional[Callable]
    user_backward_fn: Optional[Callable]
    max_async_steps: int = 1
    params_update_steps: int = 1
