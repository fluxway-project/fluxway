# Placement mathematics

[Chinese edition](placement_mathematics_zh.md)

The planner assigns every logical worker to one physical node. A worker is
identified by `(spmd_id, rank)`; its rank is not a node index, GPU index, or
local process index. A recipe constrains that assignment and orders its goals.
See [placement recipes](placement_recipe.md) for job startup.

## Select workers and nodes

| Expression | Meaning |
| --- | --- |
| `workers(S)` | Workers of every graph instance of class `S` |
| `instance("left")` | Workers of one graph instance |
| `rank(W, 0)`, `ranks(W, (0, 1))` | Workers whose rank matches the requested values |
| `all_workers()`, `union(A, B)` | All workers, or the union of selected workers |
| `occupancy(W)` | Worker count on each cluster node, including zero counts |
| `support(W)` | Nodes with at least one selected worker |
| `union(A, B)`, `intersection(A, B)`, `difference(A, B)` | Set operations on node sets |
| `cardinality(N)` | Number of nodes in a node set |

For a class with several graph instances, `rank(workers(S), 0)` selects rank
zero from each instance. Use `rank(instance("left"), 0)` to select just one.
Worker and node sets have different types and cannot be mixed.

This complete CPU example separates the two ranks of `left` and makes `right`
use the same node set. No Ray actors are started.

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

## Hard constraints and ordered objectives

`subseteq(A, B)`, `disjoint(A, B)`, and `equal(A, B)` compare node sets.
`equal(x, y)` and `less_equal(x, y)` also compare scalars.
`all_of`, `any_of`, and `negate` combine constraints. Every top-level constraint
must hold before a solution can be selected.

`lexicographic(first, second)` optimizes `first`, fixes its optimum, then optimizes
`second`. A later objective cannot trade away an earlier result.

| Objective | Behavior |
| --- | --- |
| `minimize(x)`, `maximize(x)` | Optimize a scalar |
| `max_min(v)` | Sort values ascending and maximize them in order, starting with the smallest |
| `maximize_majorization(v)` | Sort values descending and maximize them in order, favoring concentration |

`AFFINITY_FIRST` is a predefined objective sequence: fairly expand SPMDs involved
in support-affinity constraints, then isolated SPMDs; reduce occupied nodes;
concentrate target occupancy; improve affinity coverage and colocated worker
pairs; then reduce resource fragmentation. Its support-affinity relations come
from the recipe's constraints, not a separate relation registry.

## Resource accounting

For worker `w` and node `n`, let `x[w, n]` be one when the worker is placed there.
The planner enforces `sum_n x[w, n] = 1`. Occupancy is
`sum_{w in W} x[w, n]`; support contains the nodes where this sum is positive.
For each node and resource, total allocated demand cannot exceed capacity.

`ResourceScale` adds the same integer number of resource quanta to every worker
of an SPMD instance. If that number is `q`, worker demand is `base + q * step`,
with `0 <= q <= max_steps`. Resource quantities use integers or exact `Decimal`
values, not floats. Memory quantities are in bytes.

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

`ray_component_limits` reserve capacity for Ray components in addition to the
worker's base and scaled resources. They are not part of `resources_at()`.
The planner also accounts for configured driver reservations.

`resource_usage(W, "CPU")` sums allocated worker resources and their Ray
component limits. `resource_capacity(N, "CPU")` sums node capacity.
`allocated_quanta(W)` sums the selected instances' scale counts once per
instance, rather than once per rank. `weighted_dominant_share(W)` is a vector
objective for weighted resource fairness; it is not a scalar hard constraint.
`resource_fragmentation()` is a scalar that can be constrained or minimized.

## Compilation and errors

See [placement recipes](placement_recipe.md) for the sequence from solving to resource
reservation and Worker attestation. Solving the model itself does not start Workers.

The compiler freezes the expression tree and lowers it to CP-SAT variables.
Resource profiles come from compiled SPMD declarations. After solving, an
independent evaluator checks the materialized assignment against the constraints.

Unsupported expressions, unknown classes or instances, nonexistent ranks, and
invalid operand types raise errors. An infeasible problem raises
`PlacementInfeasibleError`; a solver timeout raises `PlacementSolveTimeoutError`.
Model size and objective-complexity limits are enforced before solving. Neither
failure relaxes a hard constraint or chooses another placement strategy.
