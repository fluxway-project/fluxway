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
from typing import Any, Dict, List, Optional, Callable, Type, Tuple
import logging

from fluxway.common.types import (
    Graph,
    Node,
    Edge,
    OperatorMeta,
    EdgeShardingInfo,
    ResourceSpec,
)
from fluxway.common.mesh import DeviceMesh


logger = logging.getLogger(__name__)


class Builder:
    """
    Builder class for constructing task graphs programmatically.

    Provides a fluent interface for adding nodes and edges to build
    a complete task graph for SPMD orchestration.
    """

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._spmd_classes: Dict[str, Type] = {}

    def add_node(
        self,
        name: str,
        spmd_class: Type,
        description: str = "",
        mesh: Optional[DeviceMesh] = None,
    ) -> "Builder":
        """
        Add a node to the graph.

        Args:
            name: Unique identifier for the node
            spmd_class: The SPMD class decorated with @fluxway.spmd
            description: Human-readable description of the node
            mesh: Optional DeviceMesh override. If provided, it takes precedence
                  over the mesh defined in the @spmd decorator.

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

        # Get resource and config from the decorated class
        resource = copy.deepcopy(spmd_class.get_resource())
        spmd_config = copy.deepcopy(spmd_class.get_spmd_config())

        if mesh is not None:
            # Structural Check: Ensure new mesh provides all axes required by operators.
            # (Note: operator_metas currently stores raw dicts, so we'll check axis names)
            for op_name, meta in spmd_config.operator_metas.items():
                for kind, sharding in [
                    ("input", meta.input_sharding),
                    ("output", meta.output_sharding),
                ]:
                    if not sharding:
                        continue
                    for k, v in sharding.items():
                        # Extract axis names from raw dict or ShardingSpec
                        required_axes = []
                        if isinstance(v, dict):
                            required_axes = v.get("axis", [])
                            if isinstance(required_axes, str):
                                required_axes = [required_axes]
                        elif hasattr(v, "partition"):
                            # It's a ShardingSpec-like object
                            required_axes = [
                                ax
                                for sub in v.partition.partitions.values()
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

            # Re-bind the mesh at the config and spec level
            spmd_config.mesh = mesh
            from fluxway.common.types.sharding import ShardingSpec

            for meta in spmd_config.operator_metas.values():
                for sharding_dict in [meta.input_sharding, meta.output_sharding]:
                    if not sharding_dict:
                        continue
                    for k, v in sharding_dict.items():
                        if isinstance(v, ShardingSpec):
                            sharding_dict[k] = v.with_mesh(mesh)

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

        node = Node(
            name=name,
            description=description,
            resource=resource,
            spmd_config=spmd_config,
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
        reader_class: Optional[Type] = None,
        num_workers: int = 1,
        mesh_shape: Optional[Tuple[int, ...]] = None,
        mesh_axis_names: Optional[Tuple[str, ...]] = None,
        resource: Optional[Dict[str, Any]] = None,
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
            resource: Resource requirements (num_cpus, num_gpus, etc.)
            description: Optional description
            **reader_kwargs: Additional configuration (batch_size, shuffle, etc.)
                             that will be passed to the reader as static_args.
        """
        from ...reader.grain_reader import GrainReader
        from fluxway.common.mesh import DeviceMesh

        reader_cls = reader_class or GrainReader

        # 1. Establish parallel execution mesh BEFORE adding node
        if mesh_shape and mesh_axis_names:
            reader_mesh = DeviceMesh(
                device_shape=mesh_shape, axis_names=mesh_axis_names
            )
        else:
            # Default to flat 1D mesh for simple data parallelism
            reader_mesh = DeviceMesh(
                device_shape=(num_workers,), axis_names=("worker",)
            )

        # 2. Add as an SPMD node with the explicitly created mesh
        # This triggers the mandatory mesh check in add_node
        self.add_node(name, reader_cls, description, mesh=reader_mesh)

        # 3. Inject additional configuration
        node = self._nodes[name]

        # Merge data_source into static_args
        all_static_args = {**reader_kwargs}
        if data_source is not None:
            all_static_args["data_source"] = data_source
        node.static_args.update(all_static_args)

        # Handle resource specification
        if resource:
            node.resource = ResourceSpec(
                num_cpus=resource.get("num_cpus", 1),
                num_gpus=resource.get("num_gpus", 0),
                memory_gb=resource.get("memory_gb", 0.0),
            )

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
        condition_fn: Optional[Callable[..., bool]] = None,
        source_output: Optional[str] = None,
        target_input: Optional[str] = None,
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

        edge = Edge(
            source_node=source_node,
            target_node=target_node,
            source_op=source_op,
            target_op=target_op,
            condition_fn=condition_fn,
        )

        # All edges go into _edges (no semantic distinction anymore)
        self._edges.append(edge)

        # Validate and link sharding between source and target
        self._validate_and_link_edge(
            source_node,
            source_op,
            target_node,
            target_op,
            condition_fn,
            source_output,
            target_input,
        )

        return self

    # -------------------------------------------------------------------------
    # Internal helpers for operator metadata & sharding linking
    # -------------------------------------------------------------------------

    def _ensure_meta_exists(
        self, node_name: str, method_name: str, meta: Optional[OperatorMeta], ctx: str
    ) -> OperatorMeta:
        """Ensure that an operator meta exists for the given node/method; otherwise raise."""
        if meta is None:
            spmd_class = self._spmd_classes[node_name]
            class_name = spmd_class.__name__

            op_name_by_fn = {}
            for attr_name in dir(spmd_class):
                # Use getattr on the class to avoid triggering property descriptors if any
                attr = getattr(spmd_class, attr_name, None)
                m = getattr(attr, "_fluxway_operator_meta", None)
                if m is not None:
                    op_name_by_fn[attr_name] = m.name

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
                self._nodes[node_name].spmd_config.operator_metas.keys()
            )
            raise ValueError(
                f"\n[Fluxway Graph Error] In {ctx}:\n"
                f"  SPMD '{node_name}' (class {class_name}) has no operator named '{method_name}'.\n"
                f"  Available operators for this node: {available_ops}\n"
                f"  Fix: Check for typos in your add_func_edge call, or ensure the method exists and is decorated with @operator."
            )

        return meta

    def _validate_and_link_edge(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        condition_fn: Optional[Callable[..., bool]],
        source_output: Optional[str] = None,
        target_input: Optional[str] = None,
    ) -> None:
        """
        Validate and link operator metadata between two nodes.
        Refactored into discrete validation and linking phases.
        """
        ctx = f"edge {source_node}.{source_op} -> {target_node}.{target_op}"

        # 1. Validation Phase (Strict checks on metadata, IO, and Sharding)
        src_meta, dst_meta, effective_sharding, tensor_map = self._validate_edge_io(
            source_node,
            source_op,
            target_node,
            target_op,
            source_output,
            target_input,
            ctx,
        )

        # 2. Linking Phase (Update topology and sharding info in SPMDConfigs)
        self._link_topology(
            source_node, source_op, target_node, target_op, condition_fn, tensor_map
        )
        self._link_sharding(
            source_node,
            source_op,
            target_node,
            target_op,
            src_meta,
            dst_meta,
            effective_sharding,
            condition_fn,
            tensor_map,
        )

    def _validate_edge_io(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        source_output: Optional[str],
        target_input: Optional[str],
        ctx: str,
    ) -> Tuple[OperatorMeta, OperatorMeta, Dict[str, Any], Dict[str, str]]:
        """
        Unified validation for edge IO.
        Returns (source_meta, target_meta, { 'src_specs': ..., 'dst_specs': ..., 'src_names': ..., 'dst_names': ... }, tensor_map)
        """
        src_cfg = self._nodes[source_node].spmd_config
        dst_cfg = self._nodes[target_node].spmd_config

        # 1. Ensure Metadata Exists
        src_meta = self._ensure_meta_exists(
            source_node, source_op, src_cfg.operator_metas.get(source_op), ctx
        )
        dst_meta = self._ensure_meta_exists(
            target_node, target_op, dst_cfg.operator_metas.get(target_op), ctx
        )

        # 2. Validate State Metadata
        src_cls = self._spmd_classes[source_node]
        src_meta.validate_and_update_state_output_match_dest_input(
            dst_meta=dst_meta,
            src_spmd_class=src_cls,
            src_method_name=source_op,
            ctx=ctx,
        )

        from fluxway.common.types.sharding import ShardingConfig
        from fluxway.common.types.configs import ArgumentType

        # This implementation treats parameter names as placeholders and resolves
        # the underlying flattened tensor fields automatically.
        src_args = src_meta.get_arguments(is_output=True, arg_type=ArgumentType.TENSOR)
        dst_args = dst_meta.get_arguments(is_output=False, arg_type=ArgumentType.TENSOR)

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
            (source_output is None and target_input is None)
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
        elif source_output and target_input:
            tensor_map[source_output] = target_input
        elif source_output:
            tensor_map[source_output] = source_output
        elif target_input:
            tensor_map[target_input] = target_input
        else:
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

        if not tensor_map:
            raise ValueError(
                f"\n[Fluxway Topology Error] In {ctx}:\n"
                f"  Could not automatically map outputs from '{source_node}.{source_op}' to inputs of '{target_node}.{target_op}'.\n"
                f"  Available outputs: {list(src_args)}\n"
                f"  Available inputs: {list(dst_args)}\n"
                f"  Fix: Explicitly specify target arguments using kwargs handling or ensure argument names intersect."
            )

        # 4. Validate and Extract Sharding Specs
        def to_spec(raw, mesh):
            if isinstance(raw, dict):
                try:
                    return ShardingConfig.model_validate(raw).to_spec(mesh)
                except Exception as e:
                    raise ValueError(
                        f"\n[Fluxway Sharding Error] In {ctx}:\n"
                        f"  Invalid sharding configuration: {raw}\n"
                        f"  Validation Error: {str(e)}\n"
                        f"  Fix: Ensure the dictionary contains 'type' (e.g. 'static') and 'axis'. Refer to ShardingConfig documentation."
                    ) from e
            return raw

        src_out_specs = {
            k: to_spec(v, src_cfg.mesh)
            for k, v in (src_meta.output_sharding or {}).items()
        }
        dst_in_specs = {
            k: to_spec(v, dst_cfg.mesh)
            for k, v in (dst_meta.input_sharding or {}).items()
        }

        src_names = list(tensor_map.keys())
        dst_names = list(tensor_map.values())

        for name in src_names:
            if name not in src_out_specs:
                raise ValueError(
                    f"\n[Fluxway Sharding Error] In {ctx}:\n"
                    f"  Output tensor '{name}' in operator '{source_node}.{source_op}' has no sharding defined.\n"
                    f"  Fix: Add output_sharding={{'{name}': ...}} to the @operator decorator."
                )

        for name in dst_names:
            if name not in dst_in_specs:
                raise ValueError(
                    f"\n[Fluxway Sharding Error] In {ctx}:\n"
                    f"  Input tensor '{name}' in operator '{target_node}.{target_op}' has no sharding defined.\n"
                    f"  Fix: Add input_sharding={{'{name}': ...}} to the @operator decorator."
                )

        return (
            src_meta,
            dst_meta,
            {
                "src_out_specs": src_out_specs,
                "dst_in_specs": dst_in_specs,
                "src_names": src_names,
                "dst_names": dst_names,
            },
            tensor_map,
        )

    def _link_topology(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        condition_fn: Optional[Callable[..., bool]],
        tensor_map: Dict[str, str],
    ) -> None:
        """Link nodes within the SPMDConfig topology dictionaries."""
        src_cfg = self._nodes[source_node].spmd_config
        dst_cfg = self._nodes[target_node].spmd_config
        from fluxway.common.types.configs import SPMDTopology, SPMDTopologyInfo

        # Ensure entries
        src_cfg.topology.setdefault(source_op, SPMDTopology())
        dst_cfg.topology.setdefault(target_op, SPMDTopology())

        # Link Successor
        if target_node not in src_cfg.topology[source_op].successors:
            src_cfg.topology[source_op].successors[target_node] = SPMDTopologyInfo(
                spmd_id=target_node, condition_fn=condition_fn, tensor_map=tensor_map
            )
        else:
            src_cfg.topology[source_op].successors[target_node].tensor_map.update(
                tensor_map
            )

        # Link Predecessor
        if source_node not in dst_cfg.topology[target_op].predecessors:
            dst_cfg.topology[target_op].predecessors[source_node] = SPMDTopologyInfo(
                spmd_id=source_node, tensor_map=tensor_map
            )
        else:
            dst_cfg.topology[target_op].predecessors[source_node].tensor_map.update(
                tensor_map
            )

    def _link_sharding(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        src_meta: OperatorMeta,
        dst_meta: OperatorMeta,
        effective_sharding: Dict[str, Any],
        condition_fn: Optional[Callable[..., bool]],
        tensor_map: Dict[str, str],
    ) -> None:
        """Inject edge sharding information into OperatorMeta objects."""
        src_out_specs = effective_sharding["src_out_specs"]
        dst_in_specs = effective_sharding["dst_in_specs"]

        if tensor_map:
            effective_src_out = {
                target_name: src_out_specs[src_name]
                for src_name, target_name in tensor_map.items()
            }
            effective_dst_in = {
                src_name: dst_in_specs[target_name]
                for src_name, target_name in tensor_map.items()
            }
        else:
            src_names = effective_sharding["src_names"]
            dst_names = effective_sharding["dst_names"]
            effective_src_out = {n: src_out_specs[n] for n in src_names}
            effective_dst_in = {n: dst_in_specs[n] for n in dst_names}

        # Source -> Successor Sharding

        dst_meta.input_pre_sharding.setdefault(source_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tensor_sharding=effective_src_out,
                target_op=target_op,
                tensor_map=tensor_map,
            )
        )
        src_meta.output_post_sharding.setdefault(target_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tensor_sharding=effective_dst_in,
                target_op=target_op,
                tensor_map=tensor_map,
            )
        )

        logger.debug(
            "Builder linked sharding: %s.%s <-> %s.%s, tensor_keys=%s",
            source_node,
            source_op,
            target_node,
            target_op,
            list(effective_src_out.keys()),
        )

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

        # Sync static_args to spmd_config
        for node in self._nodes.values():
            if node.static_args:
                node.spmd_config.static_args.update(node.static_args)

        graph = Graph(name=name, nodes=self._nodes.copy(), edges=self._edges.copy())

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

    def get_spmd_class(self, node_name: str) -> Optional[Type]:
        """
        Get the SPMD class associated with a node.

        Args:
            node_name: Name of the node

        Returns:
            The SPMD class or None if not found
        """
        return self._spmd_classes.get(node_name)
