from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import heapq
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import quote

from fluxway.common.compiled_graph import (
    CompiledDependency,
    CompiledOperator,
    CompiledGraph,
    OperatorKey,
    SPMDGraphView,
)
from fluxway.common.operator_policy import ArgumentType
from fluxway.common.schedule_semantics import compile_phase_schedule
from fluxway.common.types import Graph, Node


_NODE_STYLES = {
    "source": ("#E6F4EA", "#137333"),
    "transform": ("#E8F0FE", "#1967D2"),
    "sink": ("#FEF7E0", "#B06000"),
    "cyclic": ("#F3E8FD", "#8430CE"),
    "isolated": ("#F1F3F4", "#5F6368"),
}
_CROSS_STEP_PHASE_COLOR = "#007B83"
_LAYOUT_NODE_SEPARATION = 36
_LAYOUT_RANK_SEPARATION = 96
_LAYOUT_ENTRY_SEPARATION = 24
_PROJECTION_FORMAT = "fluxway.model_explorer.v1"
_PROJECTION_FILE = "model_explorer.json"
_VIEW_IDS = ("Dependencies", "Potential overlap")


@dataclass(frozen=True, slots=True)
class _OperatorLayout:
    dependency_ranks: dict[OperatorKey, int]
    roles: dict[OperatorKey, str]
    cycle_groups: dict[OperatorKey, str]
    lanes: dict[OperatorKey, str]


@dataclass(frozen=True, slots=True, order=True)
class _PhaseRelation:
    source: OperatorKey
    target: OperatorKey
    step_delta: int


@dataclass(frozen=True, slots=True)
class _CausalRelation:
    source: OperatorKey
    target: OperatorKey
    dependencies: tuple[CompiledDependency, ...]
    phase_step_deltas: tuple[int, ...]

    @property
    def blocks_same_step(self) -> bool:
        return bool(self.dependencies or 0 in self.phase_step_deltas)


def visualize_graph(
    graph: Graph,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    plan = _require_compiled_graph(graph)
    destination = Path(directory)
    destination_exists = destination.exists()
    if destination_exists and not overwrite:
        raise FileExistsError(f"Visualization directory already exists: {destination}")

    model_explorer = _model_explorer_module()
    projection = _model_explorer_projection(graph, plan, model_explorer)
    _validate_projection(projection)
    if destination_exists:
        _validate_bundle(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_json(temporary / _PROJECTION_FILE, projection)
        if destination_exists:
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup.", dir=destination.parent
                )
            )
            backup.rmdir()
            destination.replace(backup)
            try:
                temporary.replace(destination)
            except OSError:
                backup.replace(destination)
                raise
            shutil.rmtree(backup)
        else:
            temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def view_graph(
    bundle: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    directory = Path(bundle)
    model_explorer = _model_explorer_module()
    _validate_bundle(directory)
    model_path = (directory / _PROJECTION_FILE).resolve()
    config = model_explorer.config()
    config.add_model_from_path(str(model_path), adapterId="builtin_json")
    model_explorer.visualize_from_config(
        config=config,
        host=host,
        port=port,
        no_open_in_browser=not open_browser,
    )


def _require_compiled_graph(graph: Graph) -> CompiledGraph:
    plan = graph.require_compiled_graph()
    if plan.name != graph.name:
        raise ValueError(
            f"Graph name {graph.name!r} does not match compiled graph "
            f"name {plan.name!r}."
        )
    plan_spmd_ids = {spmd.spmd_id for spmd in plan.spmds}
    if set(graph.nodes) != plan_spmd_ids:
        raise ValueError("Graph nodes do not match compiled graph SPMDs.")
    for spmd in plan.spmds:
        node = graph.nodes[spmd.spmd_id]
        config = node.spmd_config
        if config.spmd_id != spmd.spmd_id:
            raise ValueError(
                f"Graph node {spmd.spmd_id!r} has SPMD config ID {config.spmd_id!r}."
            )
        node_plan = config.require_graph_view()
        if node_plan.graph_id != plan.graph_id:
            raise ValueError(
                f"Graph node {spmd.spmd_id!r} does not reference compiled graph "
                f"{plan.graph_id!r}."
            )
        if node_plan.mesh != spmd.mesh:
            raise ValueError(
                f"Graph node {spmd.spmd_id!r} mesh does not match its compiled graph."
            )
    return plan


def _model_explorer_module():
    try:
        import model_explorer  # type: ignore[import-untyped]  # pyrefly: ignore [missing-import]
    except ImportError as exc:
        raise ImportError(
            "Graph visualization requires `ai-edge-model-explorer`. "
            "Install the `fluxway[visualization]` extra."
        ) from exc
    return model_explorer


def _validate_bundle(directory: Path) -> None:
    if not directory.is_dir():
        raise ValueError(f"Visualization bundle is not a directory: {directory}")
    expected_files = {_PROJECTION_FILE}
    actual_files = {path.name for path in directory.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            f"Visualization bundle must contain exactly {sorted(expected_files)}, "
            f"got {sorted(actual_files)}."
        )
    projection = json.loads((directory / _PROJECTION_FILE).read_text(encoding="utf-8"))
    _validate_projection(projection)


def _validate_projection(projection: object) -> None:
    if not isinstance(projection, dict):
        raise ValueError("Model Explorer projection must be a JSON object.")
    label = projection.get("label")
    if not isinstance(label, str):
        raise ValueError("Model Explorer projection requires a string label.")
    graphs = projection.get("graphs")
    if not isinstance(graphs, list):
        raise ValueError("Model Explorer projection requires a graphs list.")
    graph_ids = tuple(
        graph.get("id") if isinstance(graph, dict) else None for graph in graphs
    )
    if graph_ids != _VIEW_IDS:
        raise ValueError(
            f"Model Explorer projection requires views {_VIEW_IDS!r}, got "
            f"{graph_ids!r}."
        )
    for graph in graphs:
        root_attributes = graph.get("groupNodeAttributes", {}).get("")
        if not isinstance(root_attributes, dict):
            raise ValueError("Model Explorer view requires root graph attributes.")
        if root_attributes.get("format") != _PROJECTION_FORMAT:
            raise ValueError(
                f"Model Explorer view requires format {_PROJECTION_FORMAT!r}."
            )
        graph_id = root_attributes.get("graph_id")
        if not isinstance(graph_id, str) or not graph_id:
            raise ValueError("Model Explorer view requires a non-empty graph_id.")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _node_id(operator: CompiledOperator) -> str:
    encoded = f"{operator.key.spmd_id}\0{operator.key.op_id}".encode()
    return f"op_{sha256(encoded).hexdigest()}"


def _namespace(spmd_id: str) -> str:
    return quote(spmd_id, safe="")


def _key_values(graph_builder, values: Mapping[str, str]):
    return [
        graph_builder.KeyValue(key=key, value=value) for key, value in values.items()
    ]


def _spmd_class_id(spmd: SPMDGraphView) -> str:
    spmd_class = spmd.spmd_class
    return f"{spmd_class.__module__}.{spmd_class.__qualname__}"


def _mesh_value(spmd: SPMDGraphView) -> str:
    return json.dumps(
        tuple(zip(spmd.mesh.axis_names, spmd.mesh.shape, strict=True)),
        ensure_ascii=False,
    )


def _resource_attributes(spmd: SPMDGraphView, *, prefix: str) -> dict[str, str]:
    profile = spmd.resource_profile
    return {
        f"{prefix}resources_minimum": json.dumps(
            dict(profile.minimum), sort_keys=True, default=str
        ),
        f"{prefix}resources_maximum": json.dumps(
            dict(profile.maximum), sort_keys=True, default=str
        ),
    }


def _operator_attributes(
    graph_id: str,
    graph_generation: int,
    node: Node,
    spmd: SPMDGraphView,
    operator: CompiledOperator,
    *,
    dependency_rank: int,
    role: str,
    cycle_group: str | None,
) -> dict[str, str]:
    policy = operator.spec.async_policy
    schedule = policy.schedule
    values = {
        "graph_id": graph_id,
        "graph_generation": str(graph_generation),
        "spmd_id": operator.key.spmd_id,
        "spmd_class_id": _spmd_class_id(spmd),
        "spmd_description": node.description,
        "spmd_mesh": _mesh_value(spmd),
        "op_id": operator.key.op_id,
        "function_id": operator.function_id,
        "input_signature_id": operator.input_signature_id,
        "output_signature_id": operator.output_signature_id,
        "dependency_rank": str(dependency_rank),
        "dependency_role": role,
        "async_mode": policy.mode,
        "async_depth": str(policy.depth),
        "async_commit": policy.commit,
        "state_access": policy.state.access,
        "schedule_kind": schedule.kind,
        "schedule_base_priority": str(schedule.base_priority),
        "schedule_phase_order": json.dumps(schedule.phase_ordering),
        "schedule_phase_dependencies": json.dumps(schedule.phase_dependencies),
        "schedule_lane_order": json.dumps(schedule.lane_ordering),
        "schedule_lane_priorities": json.dumps(dict(schedule.lane_priorities)),
        "schedule_phase_priorities": json.dumps(dict(schedule.phase_priorities)),
    }
    if cycle_group is None:
        values["potential_overlap_group"] = f"rank:{dependency_rank}"
    else:
        values["dependency_cycle_group"] = cycle_group
    if policy.max_staleness is not None:
        values["async_max_staleness"] = str(policy.max_staleness)
    if schedule.phase_ordering:
        values["async_phases"] = json.dumps(schedule.phase_ordering)
    if policy.state.resource is not None:
        values["state_resource"] = policy.state.resource
    if schedule.phase is not None:
        values["schedule_phase"] = schedule.phase
    if schedule.lane is not None:
        values["schedule_lane"] = schedule.lane
    values.update(_resource_attributes(spmd, prefix="spmd_"))
    return values


def _callable_id(value: Any) -> str:
    return (
        f"{value.__module__}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


def _dependency_attributes(
    dependency: CompiledDependency,
    *,
    peer_operator: OperatorKey,
) -> dict[str, str]:
    spec = dependency.spec
    values = {
        "dependency_code": str(dependency.code),
        "peer_spmd_id": peer_operator.spmd_id,
        "peer_op_id": peer_operator.op_id,
        "local_only": str(spec.local_only).lower(),
    }
    if spec.condition is not None:
        values["condition"] = _callable_id(spec.condition)
    if spec.symmetric_to is not None:
        values["symmetric_to"] = ".".join(spec.symmetric_to)
    return values


def _transfer_attributes(
    dependency: CompiledDependency,
    *,
    source_side: bool,
    peer_operator: OperatorKey,
) -> dict[str, str]:
    transfers = [
        {
            "transfer_code": transfer.transfer_code,
            "logical_tuuid": transfer.logical_tuuid,
            "source_port": transfer.spec.source.name,
            "target_port": transfer.spec.target.name,
        }
        for transfer in dependency.transfers
    ]
    ports = [
        transfer.spec.source.name if source_side else transfer.spec.target.name
        for transfer in dependency.transfers
    ]
    return {
        **_dependency_attributes(dependency, peer_operator=peer_operator),
        "__tensor_tag": ports[0] if len(ports) == 1 else f"{len(ports)} tensors",
        "kind": "data",
        "relation": "data_transfer",
        "transfer_count": str(len(transfers)),
        "ports": json.dumps(ports, ensure_ascii=False),
        "transfers": json.dumps(transfers, ensure_ascii=False, sort_keys=True),
    }


def _ordering_attributes(
    dependency: CompiledDependency,
    *,
    peer_operator: OperatorKey,
) -> dict[str, str]:
    return {
        **_dependency_attributes(dependency, peer_operator=peer_operator),
        "__tensor_tag": "ordering",
        "kind": "control",
        "relation": "ordering_dependency",
        "transfer_count": "0",
    }


def _operators(plan: CompiledGraph) -> dict[OperatorKey, CompiledOperator]:
    return {
        operator.key: operator for spmd in plan.spmds for operator in spmd.operators
    }


def _phase_relations(plan: CompiledGraph) -> tuple[_PhaseRelation, ...]:
    relations = []
    for spmd in plan.spmds:
        schedule = compile_phase_schedule(
            {
                operator.key.op_id: operator.spec.async_policy.schedule
                for operator in spmd.operators
            }
        )
        relations.extend(
            _PhaseRelation(
                source=OperatorKey(spmd.spmd_id, source_op),
                target=OperatorKey(spmd.spmd_id, target_op),
                step_delta=step_delta,
            )
            for source_op, target_op, step_delta in schedule.relations
        )
    return tuple(sorted(relations))


def _causal_relations(
    dependencies: tuple[CompiledDependency, ...],
    phase_relations: tuple[_PhaseRelation, ...],
) -> tuple[_CausalRelation, ...]:
    dependencies_by_pair: dict[
        tuple[OperatorKey, OperatorKey], list[CompiledDependency]
    ] = {}
    phase_deltas_by_pair: dict[tuple[OperatorKey, OperatorKey], set[int]] = {}
    for dependency in dependencies:
        pair = (dependency.spec.source, dependency.spec.target)
        dependencies_by_pair.setdefault(pair, []).append(dependency)
    for relation in phase_relations:
        pair = (relation.source, relation.target)
        phase_deltas_by_pair.setdefault(pair, set()).add(relation.step_delta)
    pairs = sorted(set(dependencies_by_pair) | set(phase_deltas_by_pair))
    return tuple(
        _CausalRelation(
            source=source,
            target=target,
            dependencies=tuple(
                sorted(
                    dependencies_by_pair.get((source, target), ()), key=lambda d: d.code
                )
            ),
            phase_step_deltas=tuple(
                sorted(phase_deltas_by_pair.get((source, target), ()))
            ),
        )
        for source, target in pairs
    )


def _operator_layout(
    operators: Mapping[OperatorKey, CompiledOperator],
    edges: tuple[tuple[OperatorKey, OperatorKey], ...],
) -> _OperatorLayout:
    identities = sorted(operators)
    adjacency: dict[OperatorKey, set[OperatorKey]] = {
        identity: set() for identity in identities
    }
    reverse: dict[OperatorKey, set[OperatorKey]] = {
        identity: set() for identity in identities
    }
    for source, target in edges:
        adjacency[source].add(target)
        reverse[target].add(source)

    finish_order: list[OperatorKey] = []
    visited: set[OperatorKey] = set()
    for root in identities:
        if root in visited:
            continue
        visited.add(root)
        dfs_stack = [(root, False)]
        while dfs_stack:
            identity, expanded = dfs_stack.pop()
            if expanded:
                finish_order.append(identity)
                continue
            dfs_stack.append((identity, True))
            for target in sorted(adjacency[identity], reverse=True):
                if target in visited:
                    continue
                visited.add(target)
                dfs_stack.append((target, False))

    components: list[tuple[OperatorKey, ...]] = []
    component_by_identity: dict[OperatorKey, int] = {}
    for root in reversed(finish_order):
        if root in component_by_identity:
            continue
        component_index = len(components)
        members = []
        component_stack = [root]
        component_by_identity[root] = component_index
        while component_stack:
            identity = component_stack.pop()
            members.append(identity)
            for source in sorted(reverse[identity], reverse=True):
                if source in component_by_identity:
                    continue
                component_by_identity[source] = component_index
                component_stack.append(source)
        components.append(tuple(sorted(members)))

    component_adjacency: dict[int, set[int]] = {
        index: set() for index in range(len(components))
    }
    indegree = dict.fromkeys(range(len(components)), 0)
    for source, targets in adjacency.items():
        source_component = component_by_identity[source]
        for target in targets:
            target_component = component_by_identity[target]
            if source_component == target_component:
                continue
            if target_component in component_adjacency[source_component]:
                continue
            component_adjacency[source_component].add(target_component)
            indegree[target_component] += 1

    component_rank = dict.fromkeys(range(len(components)), 0)
    ready = [
        (components[index][0], index)
        for index, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    processed_components = 0
    while ready:
        _, source_component = heapq.heappop(ready)
        processed_components += 1
        for target_component in sorted(
            component_adjacency[source_component],
            key=lambda index: components[index][0],
        ):
            component_rank[target_component] = max(
                component_rank[target_component],
                component_rank[source_component] + 1,
            )
            indegree[target_component] -= 1
            if indegree[target_component] == 0:
                heapq.heappush(
                    ready,
                    (components[target_component][0], target_component),
                )
    if processed_components != len(components):
        raise RuntimeError("Dependency condensation graph must be acyclic.")

    ranks = {
        identity: component_rank[component_by_identity[identity]]
        for identity in identities
    }
    roles = {}
    cycle_groups = {}
    for identity in identities:
        component = components[component_by_identity[identity]]
        cyclic = len(component) > 1 or identity in adjacency[identity]
        if cyclic:
            role = "cyclic"
            encoded = repr(
                tuple((key.spmd_id, key.op_id) for key in component)
            ).encode()
            cycle_groups[identity] = f"cycle:{sha256(encoded).hexdigest()[:12]}"
        elif not reverse[identity] and not adjacency[identity]:
            role = "isolated"
        elif not reverse[identity]:
            role = "source"
        elif not adjacency[identity]:
            role = "sink"
        else:
            role = "transform"
        roles[identity] = role
    return _OperatorLayout(
        dependency_ranks=ranks,
        roles=roles,
        cycle_groups=cycle_groups,
        lanes={
            key: operator.spec.async_policy.schedule.lane or "default"
            for key, operator in operators.items()
        },
    )


def _cross_step_phase_label(relation: _CausalRelation) -> str:
    deltas = tuple(delta for delta in relation.phase_step_deltas if delta != 0)
    if not deltas:
        raise ValueError(
            f"Causal relation {relation.source!r}->{relation.target!r} "
            "has no cross-step phase facts."
        )
    return f"step_delta={','.join(str(delta) for delta in deltas)}"


def _add_native_dependency_edges(
    graph_builder,
    nodes: Mapping[OperatorKey, Any],
    dependencies: tuple[CompiledDependency, ...],
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    connected_inputs: set[tuple[str, str, str]] = set()
    connected_outputs: set[tuple[str, str, str]] = set()
    for dependency in dependencies:
        source_key = dependency.spec.source
        target_key = dependency.spec.target
        source = nodes[source_key]
        target = nodes[target_key]
        output_id = f"dependency_{dependency.code}_output"
        input_id = f"dependency_{dependency.code}_input"
        target.incomingEdges.append(
            graph_builder.IncomingEdge(
                sourceNodeId=source.id,
                sourceNodeOutputId=output_id,
                targetNodeInputId=input_id,
            )
        )
        if dependency.transfers:
            source_attributes = _transfer_attributes(
                dependency,
                source_side=True,
                peer_operator=target_key,
            )
            target_attributes = _transfer_attributes(
                dependency,
                source_side=False,
                peer_operator=source_key,
            )
        else:
            source_attributes = _ordering_attributes(
                dependency,
                peer_operator=target_key,
            )
            target_attributes = _ordering_attributes(
                dependency,
                peer_operator=source_key,
            )
        source.outputsMetadata.append(
            graph_builder.MetadataItem(
                id=output_id,
                attrs=_key_values(graph_builder, source_attributes),
            )
        )
        target.inputsMetadata.append(
            graph_builder.MetadataItem(
                id=input_id,
                attrs=_key_values(graph_builder, target_attributes),
            )
        )
        for transfer in dependency.transfers:
            connected_outputs.add(
                (source_key.spmd_id, source_key.op_id, transfer.spec.source.name)
            )
            connected_inputs.add(
                (target_key.spmd_id, target_key.op_id, transfer.spec.target.name)
            )
    return connected_inputs, connected_outputs


def _add_unconnected_port_metadata(
    graph_builder,
    nodes: Mapping[OperatorKey, Any],
    operators: Mapping[OperatorKey, CompiledOperator],
    connected_inputs: set[tuple[str, str, str]],
    connected_outputs: set[tuple[str, str, str]],
) -> None:
    for key, operator in operators.items():
        node = nodes[key]
        for argument_type in ArgumentType:
            for index, name in enumerate(operator.get_arguments(False, argument_type)):
                if (key.spmd_id, key.op_id, name) in connected_inputs:
                    continue
                node.inputsMetadata.append(
                    graph_builder.MetadataItem(
                        id=f"{argument_type.value}_input_{index}",
                        attrs=_key_values(
                            graph_builder,
                            {"__tensor_tag": name, "kind": argument_type.value},
                        ),
                    )
                )
        for argument_type in (ArgumentType.DATA, ArgumentType.CONTROL):
            for index, name in enumerate(operator.get_arguments(True, argument_type)):
                if (key.spmd_id, key.op_id, name) in connected_outputs:
                    continue
                node.outputsMetadata.append(
                    graph_builder.MetadataItem(
                        id=f"{argument_type.value}_output_{index}",
                        attrs=_key_values(
                            graph_builder,
                            {"__tensor_tag": name, "kind": argument_type.value},
                        ),
                    )
                )


def _add_native_layout_edges(
    graph_builder,
    nodes: Mapping[OperatorKey, Any],
    causal_relations: tuple[_CausalRelation, ...],
) -> None:
    for relation in causal_relations:
        if not relation.blocks_same_step:
            continue
        nodes[relation.target].incomingEdges.append(
            graph_builder.IncomingEdge(
                sourceNodeId=nodes[relation.source].id,
            )
        )


def _model_explorer_projection(
    graph: Graph,
    plan: CompiledGraph,
    model_explorer,
) -> dict[str, Any]:
    operators = _operators(plan)
    phase_relations = _phase_relations(plan)
    causal_relations = _causal_relations(plan.dependencies, phase_relations)
    dependency_edges = tuple(
        (dependency.spec.source, dependency.spec.target)
        for dependency in plan.dependencies
    )
    overlap_edges = tuple(
        (relation.source, relation.target)
        for relation in causal_relations
        if relation.blocks_same_step
    )
    dependency_layout = _operator_layout(operators, dependency_edges)
    overlap_layout = _operator_layout(operators, overlap_edges)
    collection = model_explorer.graph_builder.GraphCollection(
        label=graph.name,
        graphs=[
            _model_explorer_graph(
                graph,
                plan,
                model_explorer.graph_builder,
                dependency_layout,
                operators,
                causal_relations,
                view="dependencies",
            ),
            _model_explorer_graph(
                graph,
                plan,
                model_explorer.graph_builder,
                overlap_layout,
                operators,
                causal_relations,
                view="potential_overlap",
            ),
        ],
        graphSorting="name_asc",
    )
    return asdict(collection)


def _operator_namespace(
    identity: OperatorKey,
    layout: _OperatorLayout,
    *,
    view: str,
) -> str:
    spmd_namespace = _namespace(identity.spmd_id)
    if view == "dependencies":
        return spmd_namespace
    if view != "potential_overlap":
        raise ValueError(f"Unsupported visualization view: {view!r}.")
    group = layout.cycle_groups.get(identity)
    if group is None:
        group = f"rank:{layout.dependency_ranks[identity]}"
    lane = quote(layout.lanes[identity], safe="")
    return f"{group}/lane:{lane}/{spmd_namespace}"


def _lane_sort_key(operator: CompiledOperator) -> tuple[int, str]:
    schedule = operator.spec.async_policy.schedule
    lane = schedule.lane or "default"
    if schedule.lane is not None and schedule.lane in schedule.lane_ordering:
        return schedule.lane_ordering.index(schedule.lane), lane
    return len(schedule.lane_ordering), lane


def _model_explorer_graph(
    graph: Graph,
    plan: CompiledGraph,
    graph_builder,
    layout: _OperatorLayout,
    operators: Mapping[OperatorKey, CompiledOperator],
    causal_relations: tuple[_CausalRelation, ...],
    *,
    view: str,
):
    dependency_ranks = layout.dependency_ranks
    spmds = {spmd.spmd_id: spmd for spmd in plan.spmds}
    nodes = {}
    for key, operator in sorted(
        operators.items(),
        key=lambda item: (
            dependency_ranks[item[0]],
            _lane_sort_key(item[1]),
            item[0],
        ),
    ):
        role = layout.roles[key]
        background_color, border_color = _NODE_STYLES[role]
        nodes[key] = graph_builder.GraphNode(
            id=_node_id(operator),
            label=key.op_id,
            namespace=_operator_namespace(key, layout, view=view),
            attrs=_key_values(
                graph_builder,
                _operator_attributes(
                    plan.graph_id,
                    plan.generation,
                    graph.nodes[key.spmd_id],
                    spmds[key.spmd_id],
                    operator,
                    dependency_rank=dependency_ranks[key],
                    role=role,
                    cycle_group=layout.cycle_groups.get(key),
                ),
            ),
            style=graph_builder.GraphNodeStyle(
                backgroundColor=background_color,
                borderColor=border_color,
                hoveredBorderColor=border_color,
            ),
        )

    cross_step_phase_edges: list[Any] = []
    if view == "dependencies":
        connected_inputs, connected_outputs = _add_native_dependency_edges(
            graph_builder,
            nodes,
            plan.dependencies,
        )
        _add_unconnected_port_metadata(
            graph_builder,
            nodes,
            operators,
            connected_inputs,
            connected_outputs,
        )
    elif view == "potential_overlap":
        _add_native_layout_edges(graph_builder, nodes, causal_relations)
        for relation in causal_relations:
            if not any(delta != 0 for delta in relation.phase_step_deltas):
                continue
            cross_step_phase_edges.append(
                graph_builder.Edge(
                    sourceNodeId=nodes[relation.source].id,
                    targetNodeId=nodes[relation.target].id,
                    label=_cross_step_phase_label(relation),
                )
            )
    else:
        raise ValueError(f"Unsupported visualization view: {view!r}.")

    group_attributes = {
        "": {
            "format": _PROJECTION_FORMAT,
            "graph_name": graph.name,
            "graph_id": plan.graph_id,
            "graph_generation": str(plan.generation),
            "view": view,
        }
    }
    for key in operators:
        node = graph.nodes[key.spmd_id]
        spmd = spmds[key.spmd_id]
        values = {
            "spmd_id": key.spmd_id,
            "class_id": _spmd_class_id(spmd),
            "description": node.description,
            "mesh": _mesh_value(spmd),
            **_resource_attributes(spmd, prefix=""),
        }
        namespace = _operator_namespace(key, layout, view=view)
        group_attributes[namespace] = values
        if view == "potential_overlap":
            outer_group, lane_segment, _ = namespace.split("/", 2)
            lane_group = f"{outer_group}/{lane_segment}"
            group_attributes[lane_group] = {"schedule_lane": layout.lanes[key]}
            if key in layout.cycle_groups:
                group_attributes[outer_group] = {
                    "dependency_cycle_group": layout.cycle_groups[key]
                }
            else:
                group_attributes[outer_group] = {
                    "dependency_rank": str(layout.dependency_ranks[key]),
                    "potential_overlap_group": outer_group,
                }

    overlay_data = []
    if view == "potential_overlap" and cross_step_phase_edges:
        overlay_data.append(
            graph_builder.EdgeOverlaysData(
                name="Fluxway cross-step constraints",
                overlays=[
                    graph_builder.EdgeOverlay(
                        name="Cross-step phase",
                        edgeColor=_CROSS_STEP_PHASE_COLOR,
                        edgeWidth=1,
                        edges=cross_step_phase_edges,
                    )
                ],
            )
        )

    tasks_data = graph_builder.TasksData(
        edgeOverlaysDataListLeftPane=overlay_data,
    )
    operator_namespaces = sorted(
        {_operator_namespace(key, layout, view=view) for key in operators}
    )
    if view == "dependencies":
        group_configs = [
            graph_builder.GroupNodeConfig(
                namespaceRegex="^$",
                layoutDirection=graph_builder.LayoutDirection.LEFT_RIGHT,
            ),
            *(
                graph_builder.GroupNodeConfig(
                    namespaceRegex=f"^{re.escape(namespace)}$",
                    layoutDirection=graph_builder.LayoutDirection.TOP_BOTTOM,
                )
                for namespace in operator_namespaces
            ),
        ]
        graph_id = _VIEW_IDS[0]
    elif view == "potential_overlap":
        outer_namespaces = sorted(
            {namespace.split("/", 1)[0] for namespace in operator_namespaces}
        )
        lane_namespaces = sorted(
            {namespace.rsplit("/", 1)[0] for namespace in operator_namespaces}
        )
        group_configs = [
            graph_builder.GroupNodeConfig(
                namespaceRegex="^$",
                layoutDirection=graph_builder.LayoutDirection.LEFT_RIGHT,
            ),
            *(
                graph_builder.GroupNodeConfig(
                    namespaceRegex=f"^{re.escape(namespace)}$",
                    layoutDirection=graph_builder.LayoutDirection.TOP_BOTTOM,
                )
                for namespace in (
                    *outer_namespaces,
                    *lane_namespaces,
                    *operator_namespaces,
                )
            ),
        ]
        graph_id = _VIEW_IDS[1]
    else:
        raise ValueError(f"Unsupported visualization view: {view!r}.")

    return graph_builder.Graph(
        id=graph_id,
        nodes=list(nodes.values()),
        groupNodeAttributes=group_attributes,
        groupNodeConfigs=group_configs,
        tasksData=tasks_data,
        layoutConfigs=graph_builder.LayoutConfigs(
            nodeSep=_LAYOUT_NODE_SEPARATION,
            rankSep=_LAYOUT_RANK_SEPARATION,
            edgeSep=_LAYOUT_ENTRY_SEPARATION,
        ),
    )
