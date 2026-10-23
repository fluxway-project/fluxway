# 放置数学

[英文版](placement_mathematics.md)

规划器将每个逻辑工作进程分配到一个物理节点。工作进程由 `(spmd_id, rank)` 标识；
其中的编号不是节点索引、GPU 索引或节点内的进程索引。配方约束分配方案，并规定优化目标的顺序。
作业启动方式见[放置配方](placement_recipe_zh.md)。

## 选择工作进程和节点

| 表达式 | 含义 |
| --- | --- |
| `workers(S)` | 计算图中类 `S` 的所有实例的工作进程 |
| `instance("left")` | 一个计算图实例的工作进程 |
| `rank(W, 0)`、`ranks(W, (0, 1))` | 编号符合指定值的工作进程 |
| `all_workers()`、`union(A, B)` | 全部工作进程，或选中工作进程的并集 |
| `occupancy(W)` | 每个集群节点上的进程数量，包含零值 |
| `support(W)` | 至少存在一个选中进程的节点集合 |
| `union(A, B)`、`intersection(A, B)`、`difference(A, B)` | 节点集合的并集、交集和差集 |
| `cardinality(N)` | 节点集合中的节点数 |

同一个类有多个计算图实例时，`rank(workers(S), 0)` 会选中每个实例的零号进程。
只选择一个实例时，使用 `rank(instance("left"), 0)`。
工作进程集合和节点集合是不同类型，不能混用。

下面的完整 CPU 示例让 `left` 的两个进程位于不同节点，并让 `right` 使用相同的节点集合。
代码不会启动 Ray 执行器。

<!-- test:placement-logical-mapping -->
```python
import fluxway as fw
from fluxway.placement import ClusterNode, ClusterSnapshot, PlacementPlanner
from fluxway.placement.inventory import PlacementInventory


class Task:
    def run(self) -> None:
        return None


TaskSPMD = fw.bind_spmd(
    Task,
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(2,),
    run={},
)
builder = fw.Builder()
builder.add_node("left", TaskSPMD)
builder.add_node("right", TaskSPMD)
builder.add_func_edge("left", "run", "right", "run")
graph = builder.build("logical-mapping")

left = fw.instance("left")
right = fw.instance("right")
recipe = fw.PlacementRecipe(
    recipe_id="separate-ranks",
    constraints=(
        fw.disjoint(fw.support(fw.rank(left, 0)), fw.support(fw.rank(left, 1))),
        fw.equal(fw.support(left), fw.support(right)),
    ),
    objective=fw.lexicographic(fw.max_min(fw.occupancy(fw.workers(TaskSPMD)))),
)
snapshot = ClusterSnapshot.from_nodes(
    (
        ClusterNode(node_id="node-a", resources={"CPU": 2}),
        ClusterNode(node_id="node-b", resources={"CPU": 2}),
    )
)
plan = PlacementPlanner(solve_timeout_s=10).resolve(
    PlacementInventory.from_compiled_graph(graph.require_compiled_graph()),
    recipe,
    snapshot,
)

mapping = {(item.spmd_id, item.rank): item.node_id for item in plan.assignments}
assert mapping["left", 0] != mapping["left", 1]
assert plan.node_ids_for("left") == plan.node_ids_for("right")
assert plan.worker_counts_by_node("right") == [1, 1]
assert all(result.satisfied for result in plan.constraint_results)
```

## 硬约束与目标顺序

`subseteq(A, B)`、`disjoint(A, B)` 和 `equal(A, B)` 比较节点集合。
`equal(x, y)` 和 `less_equal(x, y)` 也可以比较标量。
`all_of`、`any_of` 和 `negate` 用于组合约束。所有顶层约束都满足后，方案才有效。

`lexicographic(first, second)` 先求出 `first` 的最优值并固定，再优化 `second`。
后面的目标不能牺牲前面已经得到的结果。

| 目标 | 行为 |
| --- | --- |
| `minimize(x)`、`maximize(x)` | 最小化或最大化标量 |
| `max_min(v)` | 将向量升序排列，从最小分量开始依次最大化 |
| `maximize_majorization(v)` | 将向量降序排列，依次最大化各分量，使分布更集中 |

`AFFINITY_FIRST` 是预定义的目标序列：先公平扩展参与节点亲和约束的 SPMD，
再扩展其他 SPMD；减少占用节点；集中目标进程的分布；提高亲和关系的覆盖度和同节点进程对数；
最后减少资源碎片。节点亲和关系直接取自配方中的约束，不另建关系注册表。

## 资源核算

对于工作进程 `w` 和节点 `n`，`x[w, n]` 为一表示该进程放在该节点。
规划器要求 `sum_n x[w, n] = 1`。节点占用数为 `sum_{w in W} x[w, n]`；
这一结果为正的节点组成 `support` 集合。每个节点上，每种资源的总分配量都不能超过容量。

`ResourceScale` 为同一 SPMD 实例的每个进程增加相同整数份的资源。
份数记作 `q` 时，单个进程的需求为 `base + q * step`，且 `0 <= q <= max_steps`。
资源量使用整数或精确的 `Decimal`，不能使用浮点数。内存以字节为单位。

<!-- test:placement-resource-scaling -->
```python
from decimal import Decimal

from fluxway import ResourceProfile, ResourceScale

profile = ResourceProfile(
    base={"CPU": 1, "memory": 1024 * 1024},
    scale=ResourceScale(step={"CPU": 1}, max_steps=2, weight=Decimal("1.5")),
    ray_component_limits={"CPU": Decimal("0.25")},
)

assert profile.minimum["CPU"] == 1
assert profile.maximum["CPU"] == 3
assert profile.resources_at(1)["CPU"] == 2
assert profile.placement_resources_at(1)["CPU"] == Decimal("2.25")
```

`ray_component_limits` 为 Ray 组件预留额外容量，不计入 `resources_at()`，
而是加在工作进程的基础资源和扩展资源之上。规划器也会计入配置的驱动器预留量。

`resource_usage(W, "CPU")` 累加选中进程的分配量及其 Ray 组件预留量。
`resource_capacity(N, "CPU")` 累加节点容量。
`allocated_quanta(W)` 按实例累加扩展份数，每个实例只计一次，不按进程数量重复计数。
`weighted_dominant_share(W)` 是用于加权资源公平分配的向量目标，不能作为标量硬约束。
`resource_fragmentation()` 是标量，可以用于约束或最小化目标。

## 编译与错误

从求解到资源预留、工作进程节点校验的时序见[放置配方](placement_recipe_zh.md)。本节的模型求解本身不启动工作进程。

编译器冻结表达式树，并将其转换为 CP-SAT 变量。资源配置来自编译后的 SPMD 声明。
求解结束后，独立的求值器会重新检查分配结果是否满足约束。

不支持的表达式、未知类或实例、不存在的编号以及无效操作数类型都会报错。
无解时抛出 `PlacementInfeasibleError`，求解超时抛出 `PlacementSolveTimeoutError`。
模型规模和目标复杂度也会在求解前检查。失败不会放宽硬约束或改用另一种放置策略。
