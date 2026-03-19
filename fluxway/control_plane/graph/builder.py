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

from fluxway import SPMDConfig
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
from fluxway.common.types.sharding import (
    ShardingSpec,
    ShardingConfig,
    ChunkPartition,
    AlgebraicRouting,
    ReferentialRouting,
)
from fluxway.common.types.configs import ArgumentType


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
        self._uuid_counter = 0
        # Unified Registry for all TUUID slots: {(node, op, direction, name): tuuid}
        # Includes both definitions (outputs) and references (inputs)
        self._tuuid_slots: Dict[Tuple[str, str, str, str], int] = {}

    def _next_uuid(self) -> int:
        self._uuid_counter += 1
        return self._uuid_counter

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
                        elif isinstance(v, ShardingSpec) and isinstance(
                            v.routing, AlgebraicRouting
                        ):
                            # It's a concrete ShardingSpec
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

            # Re-bind the mesh at the config and spec level
            spmd_config.mesh = mesh

            for meta in spmd_config.operator_metas.values():
                for sharding_dict in [meta.input_sharding, meta.output_sharding]:
                    if not sharding_dict:
                        continue
                    for k, v in sharding_dict.items():
                        if isinstance(v, ShardingSpec):
                            # In the new architecture, we don't 'with_mesh' on logical routing.
                            # The mesh binding happens at runtime via PhysicalLayout.
                            # However, we can validate or store it if needed.
                            pass

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

        # Pre-assign UUIDs for all operator outputs to ensure stable ordered routing
        for op_name, meta in spmd_config.operator_metas.items():
            out_args = meta.get_arguments(is_output=True, arg_type=ArgumentType.DATA)
            tuuids = []
            for out_name in out_args:
                slot = (name, op_name, "out", out_name)
                if slot not in self._tuuid_slots:
                    self._tuuid_slots[slot] = self._next_uuid()
                tuuids.append(self._tuuid_slots[slot])
            meta.output_uuids = tuuids

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
        tensor_map: Optional[Dict[str, str]] = None,
        symmetric_to: Optional[Tuple[str, str, str, str]] = None,
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
            symmetric_to=symmetric_to,
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
            tensor_map=tensor_map,
            symmetric_to=symmetric_to,
        )

        return self

    def add_symmetric_path(
        self,
        node1: str,
        op1_fwd: str,
        op1_bwd: str,
        node2: str,
        op2_fwd: str,
        op2_bwd: str,
        tensor_map: Optional[Dict[str, str]] = None,
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
        tensor_map: Optional[Dict[str, str]] = None,
        symmetric_to: Optional[Tuple[str, str, str, str]] = None,
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
            tensor_map=tensor_map,
        )

        # 2. Linking Phase (Update topology and sharding info in SPMDConfigs)
        # Compute sharding maps for topology before linking
        src_out_specs = effective_sharding["src_out_specs"]
        dst_in_specs = effective_sharding["dst_in_specs"]

        # TUUID Generation and Mapping
        # We assign an integer TUUID to cada tensor produced by an operator
        tuuid_map: Dict[str, int] = {}  # {target_name: tuuid}
        tuuid_src_out: Dict[int, Any] = {}  # {tuuid: src_spec}
        tuuid_dst_in: Dict[int, Any] = {}  # {tuuid: dst_spec}

        if tensor_map:
            for src_name, target_name in tensor_map.items():
                u = self._tuuid_slots.get((source_node, source_op, "out", src_name))
                if u is None:
                    raise RuntimeError(
                        f"TUUID Definition not found for {source_node}.{source_op}.out.{src_name}"
                    )

                # Input-side slot registration with consistency check
                target_slot = (target_node, target_op, "in", target_name)
                if (
                    target_slot in self._tuuid_slots
                    and self._tuuid_slots[target_slot] != u
                ):
                    raise ValueError(
                        f"TUUID Conflict at {target_node}.{target_op}.in.{target_name}: "
                        f"already bound to TUUID {self._tuuid_slots[target_slot]}, "
                        f"attempted to bind to {u} via {source_node}.{source_op}.out.{src_name}"
                    )
                self._tuuid_slots[target_slot] = u

                tuuid_map[target_name] = u
                tuuid_src_out[u] = src_out_specs[src_name]
                tuuid_dst_in[u] = dst_in_specs[target_name]
        else:
            ds_names = effective_sharding["dst_names"]
            for n in ds_names:
                u = self._tuuid_slots.get((source_node, source_op, "out", n))
                if u is None:
                    raise RuntimeError(
                        f"TUUID Definition not found for {source_node}.{source_op}.out.{n}"
                    )

                target_slot = (target_node, target_op, "in", n)
                if (
                    target_slot in self._tuuid_slots
                    and self._tuuid_slots[target_slot] != u
                ):
                    raise ValueError(
                        f"TUUID Conflict at {target_node}.{target_op}.in.{n}"
                    )
                self._tuuid_slots[target_slot] = u

                tuuid_map[n] = u
                tuuid_src_out[u] = src_out_specs[n]
                tuuid_dst_in[u] = dst_in_specs[n]

        self._link_topology(
            source_node,
            source_op,
            target_node,
            target_op,
            condition_fn,
            tensor_map,
            src_meta,
            tuuid_map=tuuid_map,
            tuuid_src_out=tuuid_src_out,
            tuuid_dst_in=tuuid_dst_in,
            symmetric_to=symmetric_to,
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
            tuuid_map=tuuid_map,
            tuuid_src_out=tuuid_src_out,
            tuuid_dst_in=tuuid_dst_in,
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
        tensor_map: Optional[Dict[str, str]] = None,
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

        tensor_map = tensor_map if tensor_map is not None else {}

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

        if not tensor_map and (effective_src_names and effective_dst_names):
            raise ValueError(
                f"\n[Fluxway Topology Error] In {ctx}:\n"
                f"  Could not automatically map outputs from '{source_node}.{source_op}' to inputs of '{target_node}.{target_op}'.\n"
                f"  Both operators have data fields, but no names matched.\n"
                f"  Available outputs: {list(src_args)}\n"
                f"  Available inputs: {list(dst_args)}\n"
                f"  Fix: Explicitly specify target arguments using kwargs handling or ensure argument names intersect."
            )

        # 4. Validate and Extract Sharding Specs
        def to_routing(raw):
            if isinstance(raw, dict):
                try:
                    return ShardingConfig.model_validate(raw).to_routing()
                except Exception as e:
                    raise ValueError(
                        f"\n[Fluxway Sharding Error] In {ctx}:\n"
                        f"  Invalid sharding configuration: {raw}\n"
                        f"  Validation Error: {str(e)}\n"
                        f"  Fix: Ensure the dictionary contains 'type' (e.g. 'static') and 'axis'. Refer to ShardingConfig documentation."
                    ) from e
            return raw

        src_meta.output_sharding = {
            k: to_routing(v) for k, v in (src_meta.output_sharding or {}).items()
        }
        src_out_specs = src_meta.output_sharding

        dst_meta.input_sharding = {
            k: to_routing(v) for k, v in (dst_meta.input_sharding or {}).items()
        }
        dst_in_specs = dst_meta.input_sharding

        src_names = list(tensor_map.keys())
        dst_names = list(tensor_map.values())

        for src_name, dst_name in tensor_map.items():
            src_spec = src_out_specs[src_name]
            dst_spec = dst_in_specs[dst_name]

            src_is_mirror = isinstance(src_spec.routing, ReferentialRouting)
            dst_is_mirror = isinstance(dst_spec.routing, ReferentialRouting)

            if src_is_mirror != dst_is_mirror:
                mismatch_type = "Source" if src_is_mirror else "Destination"
                other_type = "Mirror (Referential)"
                current_type = "Algebraic (Static/Dynamic)"

                raise ValueError(
                    f"\n[Fluxway Sharding Protocol Error] In {ctx}:\n"
                    f"  Sharding protocol mismatch for tensor '{src_name}' -> '{dst_name}'.\n"
                    f"  {mismatch_type} uses {other_type} routing, but the other side uses {current_type}.\n"
                    f"  Fluxway requires both sides of a communication edge to agree on the routing protocol.\n"
                    f"  Fix: If you are using mirror/follow routing, ensure both the producer's output_sharding "
                    f"and the consumer's input_sharding are set to type='mirror'."
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
        src_meta: OperatorMeta,
        tuuid_map: Dict[str, int],
        tuuid_src_out: Dict[int, Any],
        tuuid_dst_in: Dict[int, Any],
        symmetric_to: Optional[Tuple[str, str, str, str]] = None,
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
                spmd_id=target_node,
                world_size=dst_cfg.num_workers,
                mesh=dst_cfg.mesh,
                condition_fn=condition_fn,
                tuuid_map=tuuid_map,
                tuuid_sharding=tuuid_dst_in,
                symmetric_to=symmetric_to,
                is_interleaved=getattr(src_meta, "is_interleaved", False),
                interleave_unit_size=getattr(src_meta, "interleave_unit_size", 1),
            )
        else:
            topo = src_cfg.topology[source_op].successors[target_node]
            topo.world_size = dst_cfg.num_workers
            topo.mesh = dst_cfg.mesh
            topo.tuuid_map.update(tuuid_map)
            topo.tuuid_sharding.update(tuuid_dst_in)
            if symmetric_to:
                topo.symmetric_to = symmetric_to

        # Link Predecessor
        if source_node not in dst_cfg.topology[target_op].predecessors:
            dst_cfg.topology[target_op].predecessors[source_node] = SPMDTopologyInfo(
                spmd_id=source_node,
                world_size=src_cfg.num_workers,
                mesh=src_cfg.mesh,
                tuuid_map=tuuid_map,
                tuuid_sharding=tuuid_src_out,
                symmetric_to=symmetric_to,
                is_interleaved=getattr(src_meta, "is_interleaved", False),
                interleave_unit_size=getattr(src_meta, "interleave_unit_size", 1),
            )
        else:
            topo = dst_cfg.topology[target_op].predecessors[source_node]
            topo.world_size = src_cfg.num_workers
            topo.mesh = src_cfg.mesh
            topo.tuuid_map.update(tuuid_map)
            topo.tuuid_sharding.update(tuuid_src_out)
            if symmetric_to:
                topo.symmetric_to = symmetric_to

        # Eagerly resolve topological and async strategy references
        dst_meta = dst_cfg.operator_metas[target_op]

        def resolver_src(path: str) -> Optional[int]:
            return self._resolve_path_to_tuuid(source_node, source_op, "out", path)

        def resolver_dst(path: str) -> Optional[int]:
            return self._resolve_path_to_tuuid(target_node, target_op, "in", path)

        for meta, res in [(src_meta, resolver_src), (dst_meta, resolver_dst)]:
            meta.resolve_sharding_topology(res)
            meta.resolve_async_strategy(res)

    def _infer_interleave_params(
        self, spmd_config: SPMDConfig, meta: OperatorMeta
    ) -> Tuple[bool, int]:
        """
        Infer interleave parameters based on mesh and output sharding.

        Logic:
        1. If it's not a producer (source), it doesn't interleave by default.
        2. If it's a source, we look for 'interleavable' axes.
           An axis is interleavable if it is used for sharding.
           The workers NOT part of the sharded dimension form a 'unit'.
        """
        if spmd_config.mesh is None or not isinstance(spmd_config.mesh, DeviceMesh):
            raise ValueError(
                f"\n[Fluxway Topology Error] In {spmd_config.spmd_id}:\n"
                f"  Mesh is not defined or not a DeviceMesh.\n"
                f"  Fix: Add mesh_axis_names and mesh_axis_shapes to the @operator decorator."
            )
        mesh: DeviceMesh = spmd_config.mesh
        if not mesh or mesh.size <= 1:
            return False, 1

        # Identify axes used in output sharding
        sharded_axes = set()
        for v in (meta.output_sharding or {}).values():
            if isinstance(v, dict):
                axis = v.get("axis", [])
                if isinstance(axis, str):
                    sharded_axes.add(axis)
                else:
                    sharded_axes.update(axis)
            elif isinstance(v, ShardingSpec):
                for sub in v.partitions.values():
                    sharded_axes.update(sub)

        # interleave_unit_size is the product of sizes of axes NOT used for sharding
        # These represent "Replicated" or "Model Parallel" groups that must act together.
        unit_size = 1
        for ax_name, ax_size in zip(mesh.axis_names, mesh.shape, strict=True):  # type: ignore
            if ax_name not in sharded_axes:
                unit_size *= ax_size

        # If we have more than one unit, we can interleave them
        is_interleaved = mesh.size > unit_size
        return is_interleaved, unit_size

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
        tuuid_map: Dict[str, int],
        tuuid_src_out: Dict[int, Any],
        tuuid_dst_in: Dict[int, Any],
    ) -> None:
        """Inject edge sharding information into OperatorMeta objects."""
        src_cfg = self._nodes[source_node].spmd_config
        dst_cfg = self._nodes[target_node].spmd_config
        src_out_specs = effective_sharding["src_out_specs"]
        dst_in_specs = effective_sharding["dst_in_specs"]

        # Use explicitly defined properties from OperatorMeta
        is_interleaved = getattr(src_meta, "is_interleaved", False)
        interleave_unit_size = getattr(src_meta, "interleave_unit_size", 1)

        # Legacy automated inference (if not explicitly set via @spmd_source)
        is_source = src_meta.is_source
        if not is_source:
            # Graph-based fallback: if this node:op has no incoming edges, it's a source
            has_preds = any(
                e.target_node == source_node and e.target_op == source_op
                for e in self._edges
            )
            if not has_preds:
                is_source = True

        if is_source and not is_interleaved:
            # Inference based on Sharding and Mesh
            auto_interleaved, auto_unit_size = self._infer_interleave_params(
                src_cfg, src_meta
            )
            if auto_interleaved:
                is_interleaved = auto_interleaved
                interleave_unit_size = auto_unit_size

        if is_interleaved:
            logger.debug(
                f"[Builder] Edge {source_node}.{source_op} -> {target_node}.{target_op} inferred as INTERLEAVED "
                f"(unit_size={interleave_unit_size})"
            )

        dst_meta.input_pre_sharding.setdefault(source_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tuuid_sharding=tuuid_src_out,
                tuuid_map=tuuid_map,
                target_op=target_op,
                tensor_map=tensor_map,
                input_param_map=dst_meta.input_param_map,
                output_param_map=src_meta.output_param_map,
                is_interleaved=is_interleaved,
                interleave_unit_size=interleave_unit_size,
            )
        )
        src_meta.output_post_sharding.setdefault(target_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tuuid_sharding=tuuid_dst_in,
                tuuid_map=tuuid_map,
                target_op=target_op,
                tensor_map=tensor_map,
                input_param_map=dst_meta.input_param_map,
                output_param_map=src_meta.output_param_map,
                is_interleaved=is_interleaved,
                interleave_unit_size=interleave_unit_size,
            )
        )

        logger.debug(
            "Builder linked sharding: %s.%s <-> %s.%s, tuuids=%s",
            source_node,
            source_op,
            target_node,
            target_op,
            list(tuuid_map.values()),
        )

        # When a Static tensor flows from a smaller mesh to a larger one:
        # - Source ops (interleaved virtual SPMD): round-robin scheduling covers all downstream
        #   workers regardless of mesh size mismatch. No starving risk; emit INFO only.
        # - Non-source ops: the tensor is split by mesh size, so the target workers may not
        #   receive data if the batch is too small. Emit WARNING.
        for src_name, dst_name in tensor_map.items():
            s_spec = src_out_specs.get(src_name)
            t_spec = dst_in_specs.get(dst_name)
            if (
                isinstance(s_spec, ShardingSpec)
                and isinstance(t_spec, ShardingSpec)
                and isinstance(s_spec.routing, AlgebraicRouting)
                and isinstance(t_spec.routing, AlgebraicRouting)
                and isinstance(s_spec.routing.strategy, ChunkPartition)
                and isinstance(t_spec.routing.strategy, ChunkPartition)
                and src_cfg.mesh is not None
                and dst_cfg.mesh is not None
                and src_cfg.mesh.size < dst_cfg.mesh.size
            ):
                if is_source:
                    logger.info(
                        "Static tensor '%s' (edge %s.%s -> %s.%s) flows from "
                        "mesh(size=%d) to mesh(size=%d). Source op uses interleaved "
                        "virtual SPMD scheduling; all %d downstream workers will "
                        "receive data in round-robin order.",
                        src_name,
                        source_node,
                        source_op,
                        target_node,
                        target_op,
                        src_cfg.mesh.size,
                        dst_cfg.mesh.size,
                        dst_cfg.mesh.size,
                    )
                else:
                    logger.warning(
                        "\n[Fluxway Config Warning] Static tensor '%s' "
                        "(edge %s.%s -> %s.%s) flows from mesh(size=%d) to mesh(size=%d). "
                        "Ensure the data size along every partitioned dimension is >= %d at "
                        "runtime, otherwise some workers will starve silently.",
                        src_name,
                        source_node,
                        source_op,
                        target_node,
                        target_op,
                        src_cfg.mesh.size,
                        dst_cfg.mesh.size,
                        dst_cfg.mesh.size,
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

    def _resolve_path_to_tuuid(
        self, node_name: str, op_name: str, primary_direction: str, path: str
    ) -> Optional[int]:
        """
        Find the physical TUUID for a logical path like 'forward.y'.
        Searches the unified slot registry (covers both definitions and references).
        If not found in the primary direction, falls back to the other direction.
        """
        if "." in path:
            target_op, field = path.split(".", 1)
        else:
            target_op, field = op_name, path

        tuuid = self._tuuid_slots.get((node_name, target_op, primary_direction, field))
        if tuuid is not None:
            return tuuid

        other_direction = "out" if primary_direction == "in" else "in"
        return self._tuuid_slots.get((node_name, target_op, other_direction, field))

    def reset(self) -> "Builder":
        """
        Reset the builder to empty state.

        Returns:
            Self for method chaining
        """
        self._nodes.clear()
        self._spmd_classes.clear()
        self._edges.clear()
        self._tuuid_slots.clear()
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
