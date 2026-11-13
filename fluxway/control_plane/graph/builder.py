"""
Graph Builder for constructing task graphs programmatically.

Based on graph_example.py API design.

Usage:
    builder = Builder()
    builder.add_node("SPMD1", SPMD1, "description1")
    builder.add_node("SPMD2", SPMD2, "description2")
    builder.add_func_edge("SPMD1", "forward", "SPMD2", "forward")
    builder.add_func_edge("SPMD2", "backward", "SPMD1", "backward")
    graph = builder.build(name="my_graph")
"""

import copy
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from fluxway.common.transfer import UnitRouting
from fluxway.common.types import (
    Graph,
    Node,
)
from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort as ExecutionOperatorPort,
    SPMDGraphSpec,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.operator_spec import OperatorSpec
from fluxway.resources import ResourceProfile
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.sharding import (
    ShardingSpec,
    AlgebraicRouting,
)
from fluxway.common.operator_policy import ArgumentType
from fluxway.control_plane.checkpointer.component import CheckpointComponentBinding


class Builder:
    """
    Builder class for constructing task graphs programmatically.

    Provides a fluent interface for adding nodes and edges to build
    a complete task graph for SPMD orchestration.
    """

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: list[DependencySpec] = []
        self._spmd_classes: dict[str, type] = {}

    def add_node(
        self,
        name: str,
        spmd_class: type,
        description: str = "",
        mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> "Builder":
        """
        Add a node to the graph.

        Args:
            name: Unique identifier for the node
            spmd_class: The SPMD class decorated with @fluxway.spmd
            description: Human-readable description of the node
            mesh: Optional DeviceMesh override. If provided, it takes precedence
                  over the mesh defined in the @spmd decorator.
            resource: Optional resource dict override (e.g. {"num_gpus": 1}).
            runtime_env: Optional Ray runtime_env dict override.
            operator_configs: Optional dict of {op_name: {config_key: value}}
                              to override or register operator configurations.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node with same name already exists
            ValueError: If spmd_class is not decorated with @fluxway.spmd
            ValueError: If no valid DeviceMesh is found or provided
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists in the graph")

        # Validate that spmd_class is decorated with @fluxway.spmd
        if not getattr(spmd_class, "is_fluxway_spmd", lambda: False)():
            raise ValueError(
                f"Class '{spmd_class.__name__}' is not a valid Fluxway SPMD. "
                f"Did you forget to add the @spmd decorator?"
            )

        if "resource" in kwargs:
            raise TypeError(
                "resource was removed; pass resource_profile=ResourceProfile(...)"
            )
        resource_override = kwargs.pop("resource_profile", None)
        runtime_env_override = kwargs.pop("runtime_env", None)
        operator_configs = kwargs.pop("operator_configs", None)
        if "priority" in kwargs or "placement_strategy" in kwargs:
            raise TypeError(
                "priority and placement_strategy were removed; use placement constraints"
            )

        resource_profile = spmd_class.get_resource_profile()
        spmd_config = copy.deepcopy(spmd_class.get_spmd_config())
        unbound = [
            name
            for name, spec in spmd_config.operator_specs.items()
            if spec.owner is None
        ]
        if unbound:
            raise ValueError(
                f"SPMD class {spmd_class.__qualname__!r} has operators without "
                f"an owner class: {sorted(unbound)}"
            )

        if resource_override is not None:
            if not isinstance(resource_override, ResourceProfile):
                raise TypeError("resource_profile must be a ResourceProfile")
            resource_profile = resource_override
        if not isinstance(resource_profile, ResourceProfile):
            raise TypeError(
                f"SPMD class {spmd_class.__qualname__!r} must declare a "
                "ResourceProfile or receive resource_profile= when added to a graph"
            )

        if runtime_env_override is not None:
            spmd_config.runtime_env = runtime_env_override

        if operator_configs is not None:
            from fluxway.common.bind import _apply_operator_configs

            _apply_operator_configs(spmd_config, operator_configs, spmd_class)

        if mesh is not None:
            for op_name, spec in spmd_config.operator_specs.items():
                for kind, sharding in [
                    ("input", spec.input_sharding),
                    ("output", spec.output_sharding),
                ]:
                    if not sharding:
                        continue
                    for k, v in sharding.items():
                        required_axes = []
                        if isinstance(v, dict):
                            required_axes = v.get("axis", [])
                            if isinstance(required_axes, str):
                                required_axes = [required_axes]
                        elif isinstance(v, ShardingSpec) and isinstance(
                            v.routing, AlgebraicRouting
                        ):
                            required_axes = [
                                ax
                                for sub in v.routing.partition.partitions.values()
                                for ax in sub
                            ]

                        for ax in required_axes:
                            if ax not in mesh.axis_names:
                                raise ValueError(
                                    f"\n[Fluxway Configuration Error] Node '{name}' (class '{spmd_class.__name__}') "
                                    f"uses axis '{ax}' for {kind} sharding of tensor '{k}'.\n"
                                    f"  However, the provided DeviceMesh for this node does NOT have an axis named '{ax}'.\n"
                                    f"  Current Mesh Axis Names: {mesh.axis_names}\n"
                                    f"  Fix: Ensure your DeviceMesh has matching axis names, or adjust the @operator sharding."
                                )

            spmd_config.mesh = mesh

        if not isinstance(spmd_config.mesh, DeviceMesh):
            raise ValueError(
                f"\n[Fluxway Graph Error] Node '{name}' (class '{spmd_class.__name__}') has no valid mesh defined.\n"
                f"  Every SPMD node in Fluxway must run on a DeviceMesh.\n"
                f"  Fix: Provide a mesh in @spmd decorator or pass it to add_node().\n\n"
                f"       @spmd(\n"
                f'           mesh_axis_names=("data",),\n'
                f"           mesh_axis_shapes=(4,),\n"
                f"           ...\n"
                f"       )\n"
                f"       class {spmd_class.__name__}:"
            )

        spmd_config.static_args.update(kwargs)
        spmd_config.spmd_id = name
        for registration in spmd_config.checkpoint_components:
            if isinstance(registration.component, CheckpointComponentBinding):
                registration.component.bind_to_spmd(name)

        node = Node(
            name=name,
            description=description,
            resource_profile=resource_profile,
            spmd_config=spmd_config,
            spmd_class=spmd_class,
        )

        self._nodes[name] = node
        self._spmd_classes[name] = spmd_class

        return self

    # -------------------------------------------------------------------------
    # Reader Node API
    # -------------------------------------------------------------------------

    def add_reader(
        self,
        name: str,
        data_source: Any = None,
        reader_class: type | None = None,
        num_workers: int = 1,
        mesh_shape: tuple[int, ...] | None = None,
        mesh_axis_names: tuple[str, ...] | None = None,
        resource_profile: ResourceProfile | None = None,
        description: str = "",
        **reader_kwargs: Any,
    ) -> "Builder":
        """
        Add a Reader node to the graph with a generic data source.

        This is the modern, source-driven API for adding data input nodes.
        It defaults to GrainReader which uses Google Grain for distributed loading.

        Args:
            name: Unique name for the reader node
            data_source: The underlying data source object (e.g., Grain DataSource)
            reader_class: Custom reader class (defaults to GrainReader)
            num_workers: Number of parallel workers if no mesh is provided
            mesh_shape: Optional multi-dimensional mesh shape
            mesh_axis_names: Names for mesh axes
            resource_profile: Per-worker minimum and scalable resources.
            description: Optional description
            **reader_kwargs: Additional configuration (batch_size, shuffle, etc.)
                             that will be passed to the reader as static_args.
        """
        from fluxway.reader.grain_reader import GrainReader
        from fluxway.common.mesh import DeviceMesh

        reader_cls = reader_class or GrainReader

        # Establish parallel execution mesh BEFORE adding node
        if mesh_shape and mesh_axis_names:
            reader_mesh = DeviceMesh(
                device_shape=mesh_shape, axis_names=mesh_axis_names
            )
        else:
            # Default to flat 1D mesh for simple data parallelism
            reader_mesh = DeviceMesh(
                device_shape=(num_workers,), axis_names=("worker",)
            )

        # Add as an SPMD node with the explicitly created mesh
        # This triggers the mandatory mesh check in add_node
        self.add_node(
            name,
            reader_cls,
            description,
            mesh=reader_mesh,
            resource_profile=resource_profile,
        )

        # Inject additional configuration
        node = self._nodes[name]

        # Merge data_source into static_args
        all_static_args = {**reader_kwargs}
        if data_source is not None:
            all_static_args["data_source"] = data_source
        node.spmd_config.static_args.update(all_static_args)

        return self

    # -------------------------------------------------------------------------
    # Generic func edge (core API)
    # -------------------------------------------------------------------------

    def add_func_edge(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        condition_fn: Callable[..., bool | Awaitable[bool]] | None = None,
        source_output: str | None = None,
        target_input: str | None = None,
        tensor_map: dict[str, str] | None = None,
        symmetric_to: tuple[str, str, str, str] | None = None,
        local_only: bool = False,
        unit_routing: UnitRouting | None = None,
    ) -> "Builder":
        """
        Add a generic edge specifying source and target op names.

        This is the core edge API. All other edge methods are syntactic sugar
        built on top of this method.

        Args:
            source_node: Source node name
            source_op: Source node's op name (e.g., "forward", "rollout")
            target_node: Target node name
            target_op: Target node's op name (e.g., "forward", "reward")
            condition_fn: Optional condition function
            source_output: Optional specific output field name from source op
            target_input: Optional specific input field name for target op

        Returns:
            Self for method chaining

        Examples:
            # forward -> forward
            builder.add_func_edge("A", "forward", "B", "forward")

            # forward -> backward self-loop (sink node triggers its own backward)
            builder.add_func_edge("Sink", "forward", "Sink", "backward")

            # rollout -> reward (arbitrary user-defined ops)
            builder.add_func_edge("Actor", "rollout", "RewardModel", "compute_reward")
        """
        if source_node not in self._nodes:
            raise ValueError(f"Source node '{source_node}' not found")
        if target_node not in self._nodes:
            raise ValueError(f"Target node '{target_node}' not found")
        if local_only and source_node != target_node:
            raise ValueError(
                "local_only edges require source_node == target_node, "
                f"got {source_node!r} -> {target_node!r}."
            )
        if condition_fn is not None and not callable(condition_fn):
            raise TypeError("condition_fn must be callable")
        condition = condition_fn

        dependency = self._validate_and_link_edge(
            source_node,
            source_op,
            target_node,
            target_op,
            condition,
            source_output,
            target_input,
            tensor_map=tensor_map,
            symmetric_to=symmetric_to,
            local_only=local_only,
            unit_routing=unit_routing,
        )

        self._merge_or_append_edge(dependency)

        return self

    def _merge_or_append_edge(
        self,
        dependency: DependencySpec,
    ) -> None:
        for index, edge in enumerate(self._edges):
            if edge.key != dependency.key:
                continue
            if (
                edge.local_only != dependency.local_only
                or edge.symmetric_to != dependency.symmetric_to
                or edge.condition is not dependency.condition
            ):
                raise ValueError(
                    f"Conflicting dependency metadata: {dependency.key!r}."
                )
            existing_transfers = {
                (transfer.source.name, transfer.target.name)
                for transfer in edge.transfers
            }
            new_transfers = {
                (transfer.source.name, transfer.target.name)
                for transfer in dependency.transfers
            }
            duplicates = existing_transfers & new_transfers
            if not dependency.transfers or duplicates:
                raise ValueError(
                    f"Duplicate dependency transfers for {dependency.key!r}: "
                    f"{sorted(duplicates)!r}."
                )
            self._edges[index] = replace(
                dependency,
                transfers=edge.transfers + dependency.transfers,
            )
            return
        self._edges.append(dependency)

    def add_symmetric_path(
        self,
        node1: str,
        op1_fwd: str,
        op1_bwd: str,
        node2: str,
        op2_fwd: str,
        op2_bwd: str,
        tensor_map: dict[str, str] | None = None,
    ) -> "Builder":
        """
        Create a pair of symmetric edges (forward/backward) between two nodes.

        This enables 'Zero-Declaration' routing in the backward pass. The backward
        pass will automatically use the provenance information captured during
        the forward pass to route data (gradients) back to its origin.

        Args:
            node1: Source node for forward (Target for backward)
            op1_fwd: Forward op on node1
            op1_bwd: Backward op on node1
            node2: Target node for forward (Source for backward)
            op2_fwd: Forward op on node2
            op2_bwd: Backward op on node2
            tensor_map: Mapping for the backward edge tensors (e.g., {"grad_out": "grad_in"}).
                        If None, matching names are mapped.
        """
        fwd_key = (node1, op1_fwd, node2, op2_fwd)
        bwd_key = (node2, op2_bwd, node1, op1_bwd)

        # 1. Forward Edge: node1:op1_fwd -> node2:op2_fwd
        self.add_func_edge(node1, op1_fwd, node2, op2_fwd, symmetric_to=bwd_key)

        # 2. Backward Edge: node2:op2_bwd -> node1:op1_bwd
        self.add_func_edge(
            node2, op2_bwd, node1, op1_bwd, tensor_map=tensor_map, symmetric_to=fwd_key
        )

        return self

    # -------------------------------------------------------------------------
    # Internal helpers for operator metadata & sharding linking
    # -------------------------------------------------------------------------

    def _require_operator_spec(
        self,
        node_name: str,
        method_name: str,
        spec: OperatorSpec | None,
        ctx: str,
    ) -> OperatorSpec:
        if spec is None:
            spmd_class = self._spmd_classes[node_name]
            class_name = spmd_class.__name__

            op_name_by_fn = {}
            for attr_name in dir(spmd_class):
                # Use getattr on the class to avoid triggering property descriptors if any
                attr = getattr(spmd_class, attr_name, None)
                operator_spec = getattr(attr, "_fluxway_operator_spec", None)
                if operator_spec is not None:
                    op_name_by_fn[attr_name] = operator_spec.name

            if method_name in op_name_by_fn:
                actual_op_name = op_name_by_fn[method_name]
                raise ValueError(
                    f"\n[Fluxway Graph Error] In {ctx}:\n"
                    f"  You referred to operator '{method_name}', which is the name of a function in '{class_name}'.\n"
                    f"  However, this function is decorated as @operator(name='{actual_op_name}').\n"
                    f"  Fix: Change your code to use '{actual_op_name}' instead of '{method_name}' in the builder (add_func_edge)."
                )

            if getattr(spmd_class, method_name, None) is not None:
                raise ValueError(
                    f"\n[Fluxway Graph Error] In {ctx}:\n"
                    f"  Method '{method_name}' exists in class '{class_name}' but it is NOT decorated with @operator.\n"
                    f"  Every method used in the graph must have the @operator decorator to define its sharding and IO.\n"
                    f"  Fix: Add @operator to '{class_name}.{method_name}':\n\n"
                    f"       @operator(name='{method_name}', ...)\n"
                    f"       def {method_name}(self, ...):"
                )

            available_ops = list(
                self._nodes[node_name].spmd_config.operator_specs.keys()
            )
            raise ValueError(
                f"\n[Fluxway Graph Error] In {ctx}:\n"
                f"  SPMD '{node_name}' (class {class_name}) has no operator named '{method_name}'.\n"
                f"  Available operators for this node: {available_ops}\n"
                f"  Fix: Check for typos in your add_func_edge call, or ensure the method exists and is decorated with @operator."
            )

        return spec

    def _validate_and_link_edge(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        condition_fn: Callable[..., bool | Awaitable[bool]] | None,
        source_output: str | None = None,
        target_input: str | None = None,
        tensor_map: dict[str, str] | None = None,
        symmetric_to: tuple[str, str, str, str] | None = None,
        local_only: bool = False,
        unit_routing: UnitRouting | None = None,
    ) -> DependencySpec:
        ctx = f"edge {source_node}.{source_op} -> {target_node}.{target_op}"
        resolved_tensor_map = self._resolve_edge_tensor_map(
            source_node,
            source_op,
            target_node,
            target_op,
            source_output,
            target_input,
            ctx,
            tensor_map=tensor_map,
        )
        if unit_routing is not None and (local_only or not resolved_tensor_map):
            raise ValueError("unit routing requires non-local tensor transfers")
        return DependencySpec(
            source=OperatorKey(source_node, source_op),
            target=OperatorKey(target_node, target_op),
            transfers=tuple(
                TensorTransfer(
                    source=ExecutionOperatorPort(name=source_name),
                    target=ExecutionOperatorPort(name=target_name),
                    unit_routing=unit_routing,
                )
                for source_name, target_name in resolved_tensor_map.items()
            ),
            local_only=local_only,
            condition=condition_fn,
            symmetric_to=symmetric_to,
        )

    def _resolve_edge_tensor_map(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        source_output: str | None,
        target_input: str | None,
        ctx: str,
        tensor_map: dict[str, str] | None = None,
    ) -> dict[str, str]:
        src_cfg = self._nodes[source_node].spmd_config
        dst_cfg = self._nodes[target_node].spmd_config

        # 1. Ensure Metadata Exists
        src_meta = self._require_operator_spec(
            source_node, source_op, src_cfg.operator_specs.get(source_op), ctx
        )
        dst_meta = self._require_operator_spec(
            target_node, target_op, dst_cfg.operator_specs.get(target_op), ctx
        )
        if tensor_map is not None:
            return dict(tensor_map)

        # This implementation treats parameter names as placeholders and resolves
        # the underlying flattened tensor fields automatically.
        src_args = src_meta.get_arguments(is_output=True, arg_type=ArgumentType.DATA)
        dst_args = dst_meta.get_arguments(is_output=False, arg_type=ArgumentType.DATA)

        effective_src_names = src_args
        effective_dst_names = dst_args

        # If target_input refers to a logical @data_type parameter,
        # map all its constituent fields instead of the name itself.
        explicit_target = False
        if target_input and target_input in (
            getattr(dst_meta, "input_param_map", {}) or {}
        ):
            effective_dst_names = [
                f for f in dst_meta.input_param_map[target_input] if f in dst_args
            ]
            target_input = None  # Consumed
            explicit_target = True

        # Similarly for source placeholders/operators
        explicit_source = False
        if source_output and source_output in (
            getattr(src_meta, "output_param_map", {}) or {}
        ):
            effective_src_names = [
                f for f in src_meta.output_param_map[source_output] if f in src_args
            ]
            source_output = None  # Consumed
            explicit_source = True

        # If positional source/target provided, use them directly
        if not source_output and len(effective_src_names) == 1:
            source_output = effective_src_names[0]
            explicit_source = True
        if not target_input and len(effective_dst_names) == 1:
            target_input = effective_dst_names[0]
            explicit_target = True

        tensor_map = {}

        # Positional mapping if both endpoints were explicitly specified and their structures match
        if (
            not tensor_map
            and (source_output is None and target_input is None)
            and explicit_source
            and explicit_target
        ):
            if len(effective_src_names) == len(effective_dst_names):
                for s, d in zip(effective_src_names, effective_dst_names, strict=True):  # type: ignore
                    tensor_map[s] = d
            else:
                raise ValueError(
                    f"\n[Fluxway Topology Error] In {ctx}:\n"
                    f"  Cannot perform structural mapping because lengths differ.\n"
                    f"  Source has {len(effective_src_names)} fields: {effective_src_names}\n"
                    f"  Target has {len(effective_dst_names)} fields: {effective_dst_names}\n"
                    f"  Fix: Ensure the data structures are identical or extract specific fields."
                )
        # Simple 1-to-1 explicit mapping of primitive names
        elif not tensor_map and source_output and target_input:
            tensor_map[source_output] = target_input
        elif not tensor_map and source_output:
            tensor_map[source_output] = source_output
        elif not tensor_map and target_input:
            tensor_map[target_input] = target_input
        elif not tensor_map:
            # Fallback: Intersection (Map fields with matching names)
            common = [n for n in effective_src_names if n in effective_dst_names]
            for n in common:
                tensor_map[n] = n

        # 1-to-1 fallback for single-tensor streams if still no map
        if (
            not tensor_map
            and len(effective_src_names) == 1
            and len(effective_dst_names) == 1
        ):
            tensor_map[effective_src_names[0]] = effective_dst_names[0]

        if not tensor_map and (effective_src_names and effective_dst_names):
            raise ValueError(
                f"\n[Fluxway Topology Error] In {ctx}:\n"
                f"  Could not automatically map outputs from '{source_node}.{source_op}' to inputs of '{target_node}.{target_op}'.\n"
                f"  Both operators have data fields, but no names matched.\n"
                f"  Available outputs: {list(src_args)}\n"
                f"  Available inputs: {list(dst_args)}\n"
                f"  Fix: Explicitly specify target arguments using kwargs handling or ensure argument names intersect."
            )

        return tensor_map

    def build(self, name: str = "") -> Graph:
        """
        Build and return the complete graph.

        Args:
            name: Name for the graph

        Returns:
            Constructed Graph object

        Raises:
            ValueError: If graph validation fails
        """
        # Validate that all edge references exist
        for edge in self._edges:
            if edge.source_node not in self._nodes:
                raise ValueError(
                    f"Edge references non-existent source node: {edge.source_node}"
                )
            if edge.target_node not in self._nodes:
                raise ValueError(
                    f"Edge references non-existent target node: {edge.target_node}"
                )

        active_ops_by_node: dict[str, set[str]] = {
            node_name: set() for node_name in self._nodes
        }
        for edge in self._edges:
            active_ops_by_node[edge.source_node].add(edge.source_op)
            active_ops_by_node[edge.target_node].add(edge.target_op)

        for node_name, node in self._nodes.items():
            active_ops = active_ops_by_node[node_name]
            unknown_ops = active_ops - set(node.spmd_config.operator_specs)
            if unknown_ops:
                raise ValueError(
                    f"Active operators for node {node_name!r} are missing "
                    f"operator metadata: {sorted(unknown_ops)}"
                )
            if not active_ops:
                raise ValueError(
                    f"Node {node_name!r} has no active operators in graph {name!r}."
                )

        spmd_specs = []
        for spmd_id in sorted(self._nodes):
            resource_profile = self._nodes[spmd_id].resource_profile
            assert resource_profile is not None, (
                f"Node {spmd_id!r} lacks a resource profile"
            )
            spmd_specs.append(
                SPMDGraphSpec.create(
                    spmd_id=spmd_id,
                    spmd_class=self._nodes[spmd_id].spmd_class,
                    mesh=self._nodes[spmd_id].spmd_config.mesh,
                    resource_profile=resource_profile,
                    operators={
                        op_id: self._nodes[spmd_id].spmd_config.operator_specs[op_id]
                        for op_id in sorted(active_ops_by_node[spmd_id])
                    },
                )
            )

        compiled_graph = compile_graph(
            name=name,
            spmds=tuple(spmd_specs),
            dependencies=tuple(self._edges),
            canonical_lane_presets=True,
        )
        graph_nodes = {}
        for spmd_id, node in self._nodes.items():
            spmd_view = compiled_graph.spmd(spmd_id)
            updated_operator_specs = dict(node.spmd_config.operator_specs)
            for op in spmd_view.operators:
                updated_operator_specs[op.key.op_id] = op.spec
            graph_nodes[spmd_id] = node.model_copy(
                update={
                    "spmd_config": node.spmd_config.model_copy(
                        update={
                            "operator_specs": updated_operator_specs,
                            "static_args": dict(node.spmd_config.static_args),
                            "graph_view": spmd_view,
                        }
                    )
                }
            )
        graph = Graph(
            name=name,
            nodes=graph_nodes,
            compiled_graph=compiled_graph,
        )
        return graph

    def reset(self) -> "Builder":
        """
        Reset the builder to empty state.

        Returns:
            Self for method chaining
        """
        self._nodes.clear()
        self._spmd_classes.clear()
        self._edges.clear()
        return self

    def get_spmd_class(self, node_name: str) -> type | None:
        """
        Get the SPMD class associated with a node.

        Args:
            node_name: Name of the node

        Returns:
            The SPMD class or None if not found
        """
        return self._spmd_classes.get(node_name)
