"""
Sharding specification types for Fluxway framework.

Provides comprehensive sharding primitives supporting:
- Static/Dense sharding (block partitioning)
- Dynamic/Sparse sharding (hash, cyclic, lookup)
- Dependency-based sharding (shadow, direct indices)
"""

from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

import logging

_logger = logging.getLogger(__name__)


# Type alias for dynamic routing function
# fn(tensor, tensor_meta, post_sharding_spec) -> routing result (indices/destinations)
# Note: tensor_meta is TensorMeta from data_types.py (using Any to avoid circular import)
DynamicRoutingFn = Callable[[Any, Any, "ShardingSpec"], Any]


# =============================================================================
# Layout Operations
# =============================================================================

class LayoutOp(IntEnum):
    """Layout operation for data distribution."""
    
    # Static / Dense
    BLOCK = auto()       # Block partitioning: rank = index // size
    
    # Dynamic / Sparse
    CYCLIC = auto()      # Round-robin: rank = index % size
    HASH = auto()        # Hash-based: rank = hash(val) % size
    LOOKUP = auto()      # Table lookup: rank = table[val]
    
    # Special
    REPLICATE = auto()   # Broadcast to all ranks


class LayoutType(Enum):
    """Type of data layout."""
    DENSE = 1   # Contiguous geometric blocks (Replicated, Sharded, Partial)
    SPARSE = 2  # Discrete sets (Hash/Mod/Lookup)


# =============================================================================
# Mesh Definition
# =============================================================================

class LayoutMesh(BaseModel):
    """
    N-dimensional device mesh with user-defined axis names.
    
    The mesh defines the topology of devices and supports:
    - User-defined axis names (e.g., "data", "model", "pipeline")
    - Ordered axes where position matters
    - Coordinate lookups for ranks
    
    Example:
        mesh = LayoutMesh(
            axis_names=("data", "model"),
            axis_shapes=(4, 2)
        )
        # Creates a 4x2 mesh with 8 devices total
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    axis_names: Tuple[str, ...]
    axis_shapes: Tuple[int, ...]
    
    # Computed fields (set in model_post_init)
    device_mesh: Optional[np.ndarray] = Field(default=None, exclude=True)
    axis_map: Dict[str, int] = Field(default_factory=dict, exclude=True)
    axis_name_to_idx: Dict[str, int] = Field(default_factory=dict, exclude=True)
    
    @model_validator(mode='after')
    def validate_and_build_mesh(self) -> 'LayoutMesh':
        """Validate and build internal mesh structures."""
        if len(self.axis_names) != len(self.axis_shapes):
            raise ValueError(
                f"axis_names and axis_shapes must have same length: "
                f"{len(self.axis_names)} != {len(self.axis_shapes)}"
            )
        
        # Build device mesh array
        device_num = int(np.prod(self.axis_shapes)) if self.axis_shapes else 1
        self.device_mesh = np.reshape(np.arange(0, device_num), self.axis_shapes) if self.axis_shapes else np.array([0])
        
        # Build axis name -> shape mapping
        self.axis_map = dict(zip(self.axis_names, self.axis_shapes))
        
        # Build axis name -> index mapping
        self.axis_name_to_idx = {name: i for i, name in enumerate(self.axis_names)}
        
        return self
    
    @property
    def num_devices(self) -> int:
        """Total number of devices in the mesh."""
        return int(np.prod(self.axis_shapes)) if self.axis_shapes else 1
    
    @property
    def ndim(self) -> int:
        """Number of mesh dimensions."""
        return len(self.axis_names)
    
    def get_axis_size(self, axis_name: str) -> int:
        """Get size of a named axis."""
        return self.axis_map.get(axis_name, 1)
    
    def get_axis_index(self, axis_name: str) -> int:
        """Get index of a named axis."""
        if axis_name not in self.axis_name_to_idx:
            raise ValueError(f"Unknown axis name: {axis_name}")
        return self.axis_name_to_idx[axis_name]
    
    def get_coordinate(self, rank_id: int) -> Tuple[int, ...]:
        """Get rank's multi-dimensional coordinate in the mesh."""
        if not self.axis_shapes or self.device_mesh is None:
            return ()
        coords = np.argwhere(self.device_mesh == rank_id)
        if len(coords) == 0:
            raise ValueError(f"Rank {rank_id} not found in mesh")
        return tuple(int(c) for c in coords[0])
    
    def get_rank(self, coordinate: Tuple[int, ...]) -> int:
        """Get rank ID from coordinate."""
        if self.device_mesh is None:
            return 0
        return int(self.device_mesh[coordinate])
    
    def to_engine_mesh(self, engine: Any) -> Any:
        """Convert to framework-specific mesh (JAX, PyTorch, etc.)."""
        # TODO: Implement framework-specific conversion
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"LayoutMesh(axis_names={self.axis_names}, axis_shapes={self.axis_shapes})"


# =============================================================================
# Axis Specification
# =============================================================================

class Axis(BaseModel):
    """
    Axis identifier that can be specified by index or name.
    
    Supports three formats:
    1. Name with None: ("batch", None, "model") - for static sharding
    2. Name compact: ("batch", "model") - only sharded dims
    3. Index: (0, 2) - direct index specification
    """
    model_config = ConfigDict(frozen=False)
    
    axis: Tuple[Union[int, str, None], ...] = Field(default=(None,))
    
    def __init__(self, *axes: Union[int, str, None], **data):
        """Initialize with one or more axis identifiers."""
        if axes:
            axis_val = axes if len(axes) > 1 else (axes[0],) if axes else (None,)
            super().__init__(axis=axis_val, **data)
        else:
            super().__init__(**data)
    
    def validate_against_mesh(self, mesh: LayoutMesh) -> None:
        """Validate axis values against mesh, raise ValueError if invalid."""
        for a in self.axis:
            if a is None:
                continue
            elif isinstance(a, int):
                if a < 0 or a >= mesh.ndim:
                    raise ValueError(
                        f"Axis index {a} out of range [0, {mesh.ndim})"
                    )
            elif isinstance(a, str):
                if a not in mesh.axis_names:
                    raise ValueError(
                        f"Axis name '{a}' not found in mesh. "
                        f"Available: {list(mesh.axis_names)}"
                    )
    
    def resolve(self, mesh: LayoutMesh) -> Tuple[int, ...]:
        """
        Resolve axis specification to a sharding mask.
        
        Returns:
            Tuple of size mesh.ndim where:
            - 1 means the axis is sharded
            - -1 means the axis is replicated (not sharded)
        """
        self.validate_against_mesh(mesh)

        # Initialize all axes as unsharded (-1)
        result = [-1] * mesh.ndim
        
        # Mark sharded axes as 1
        for a in self.axis:
            if a is None:
                continue
            elif isinstance(a, int):
                result[a] = 1
            elif isinstance(a, str):
                result[mesh.get_axis_index(a)] = 1
        
        return tuple(result)


# =============================================================================
# Dense Layout Types
# =============================================================================

class DenseLayoutSlice(BaseModel):
    """Slice of a dimension for dense sharding."""
    start: int
    end: int

    axis_index: int
    
    @property
    def size(self) -> int:
        return self.end - self.start


# =============================================================================
# Dependency Types
# =============================================================================

class DependencyMode(Enum):
    """Mode for sharding dependency between tensors."""
    
    # Mode A: Shadow following (e.g., KV Cache)
    # This tensor follows the sharding of the dependency tensor.
    # If Key goes to Rank 1, Value also goes to Rank 1.
    SHADOW = auto()
    
    # Mode B: Direct indices (e.g., MoE)
    # The dependency tensor's values are the destination ranks.
    # If TopK indices say token i goes to Rank 5, this tensor goes to Rank 5.
    DIRECT_INDICES = auto()


class ShardingDependency(BaseModel):
    """Describes sharding dependency between tensors."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Name of the tensor this depends on
    target_tensor_name: str
    
    # Dependency mode
    mode: DependencyMode
    
    # Which axis of the target tensor provides routing guidance
    target_axis: Union[Axis, int] = Field(default=0)
    
    @field_validator('target_axis', mode='before')
    @classmethod
    def convert_target_axis(cls, v):
        if isinstance(v, int):
            return Axis(v)
        return v


# =============================================================================
# Sparse Layout Rule
# =============================================================================

class SparseLayoutRule(BaseModel):
    """
    Describes sparse/dynamic distribution rule for a tensor.
    Does not include physical destinations, only logical routing algorithm.
    
    The routing function `fn` determines how data is distributed:
        fn(tensor, tensor_meta, post_sharding_spec) -> routing result
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Which dimensions are dynamically distributed
    axis: Axis
    
    # User-provided routing function
    # fn(tensor, tensor_meta, post_sharding_spec) -> routing result
    fn: DynamicRoutingFn


# =============================================================================
# Internal Layout
# =============================================================================

class InternalLayout(BaseModel):
    """
    Internal representation of tensor layout.
    Result of lowering a ShardingSpec.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: LayoutType
    mesh: LayoutMesh
    global_shape: Tuple[int, ...]
    
    # Dense: slices per dimension
    dense_slices: Optional[Tuple[DenseLayoutSlice, ...]] = None
    
    # Sparse: routing rule
    sparse_rule: Optional[SparseLayoutRule] = None
    
    def get_local_shape(self) -> Tuple[int, ...]:
        """Get local tensor shape for this rank."""
        if self.type == LayoutType.DENSE and self.dense_slices:
            return tuple(s.size for s in self.dense_slices)
        # Sparse shape is dynamic, unknown at compile time
        return (-1,)


# =============================================================================
# Sharding Spec Base
# =============================================================================
class ShardingSpec(BaseModel):
    """
    Base class for user-visible sharding specifications.
    
    The `axis` field specifies which tensor dimensions are sharded.
    
    Supports three input formats:
    1. Name with None: ("batch", None, "model") - None means no sharding on that dim
    2. Name compact: ("batch", "model") - only specify sharded dims
    3. Index: (0, 2) - specify sharded dims by index
    
    After initialization, axis is normalized to index format.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    layout_mesh: LayoutMesh
    axis: Axis  # User-provided axis specification
    
    # Sharding mask: tuple of size mesh.ndim where 1=sharded, -1=replicated
    normalized_axes: Tuple[int, ...] = Field(default=(), exclude=True)
    
    @model_validator(mode='after')
    def validate_and_normalize_axis(self) -> 'ShardingSpec':
        """Validate axis against mesh and normalize to index format."""
        mesh = self.layout_mesh
        
        # Use Axis.resolve_with_mapping for full normalization
        normalized = self.axis.resolve(mesh)
        
        object.__setattr__(self, 'normalized_axes', normalized)
        
        _logger.debug(f"ShardingSpec normalized: {self.axis.axis} -> {normalized}")
        
        return self
    
    def lower(self, global_shape: Tuple[int, ...], rank_id: int) -> InternalLayout:
        """Lower high-level spec to internal layout."""
        raise NotImplementedError
    
    def to_engine_spec(self, engine: Any) -> Any:
        """Convert to framework-specific sharding spec."""
        raise NotImplementedError


# =============================================================================
# Static/Dense Sharding Spec
# =============================================================================

class StaticShardingSpec(ShardingSpec):
    """
    Static sharding spec similar to JAX PartitionSpec.
    
    Uses inherited `axis` field with format: ("batch", None, "model")
    where each position maps a tensor dimension to a mesh axis.
    None means that dimension is replicated.
    
    After validation, uses `dim_to_mesh_axis: Tuple[int, ...]`
    where -1 indicates replicated dimension.
    
    Example:
        mesh = LayoutMesh(axis_names=("data", "model"), axis_shapes=(4, 2))
        spec = StaticShardingSpec(
            layout_mesh=mesh,
            axis=Axis("data", None, "model")  # -> dim_to_mesh_axis=(0, -1, 1)
        )
    """
    
    def lower(self, global_shape: Tuple[int, ...], rank_id: int) -> InternalLayout:
        """
        Lower high-level spec to internal layout.
        
        Uses normalized_axes mask where:
        - 1 means the axis is sharded (apply block partitioning)
        - -1 means the axis is replicated (full range)
        """
        normalized_axes = self.normalized_axes
        
        if len(normalized_axes) != self.layout_mesh.ndim:
            raise ValueError(
                f"normalized_axes length {len(normalized_axes)} does not match "
                f"mesh.ndim {self.layout_mesh.ndim}"
            )
        
        if len(global_shape) != self.layout_mesh.ndim:
            raise ValueError(
                "Global shape must have same number of dimensions as mesh. "
                f"Got {len(global_shape)} dimensions but mesh has {self.layout_mesh.ndim} dimensions."
            )
        
        # Get rank's coordinate in mesh
        mesh_coords = self.layout_mesh.get_coordinate(rank_id)
        
        slices = []
        for mesh_axis_idx, dim_len in enumerate(global_shape):
            if normalized_axes[mesh_axis_idx] == -1:
                # Replicated: full range
                slices.append(DenseLayoutSlice(start=0, end=dim_len, axis_index=mesh_axis_idx))
            else:
                # Sharded: block partitioning
                num_devices = self.layout_mesh.axis_shapes[mesh_axis_idx]
                device_pos = mesh_coords[mesh_axis_idx]
                
                chunk_size = math.ceil(dim_len / num_devices)
                start = device_pos * chunk_size
                end = min(start + chunk_size, dim_len)
                
                slices.append(DenseLayoutSlice(start=start, end=end, axis_index=mesh_axis_idx))
        
        return InternalLayout(
            type=LayoutType.DENSE,
            mesh=self.layout_mesh,
            global_shape=global_shape,
            dense_slices=tuple(slices)
        )
    
    def infer_global_shape(self, local_shape: Tuple[int, ...], rank_id: int) -> Tuple[int, ...]:
        """
        Infer global shape from local shape and rank.
        
        This is the inverse of the lower() operation. Given a local tensor shape
        and the rank that holds it, compute the original global shape.
        
        Uses normalized_axes mask where:
        - 1 means the axis is sharded (multiply by num_devices)
        - -1 means the axis is replicated (keep local size)
        
        Args:
            local_shape: Shape of the local tensor shard
            rank_id: Rank ID of the worker holding this shard
            
        Returns:
            Inferred global shape
        """
        normalized_axes = self.normalized_axes
        
        if len(normalized_axes) != self.layout_mesh.ndim:
            raise ValueError(
                f"normalized_axes length {len(normalized_axes)} does not match "
                f"mesh.ndim {self.layout_mesh.ndim}"
            )
        
        if len(local_shape) != self.layout_mesh.ndim:
            raise ValueError(
                "Local shape must have same number of dimensions as mesh. "
                f"Got {len(local_shape)} dimensions but mesh has {self.layout_mesh.ndim} dimensions."
            )
        
        global_shape = []
        for mesh_axis_idx, local_dim in enumerate(local_shape):
            if normalized_axes[mesh_axis_idx] == -1:
                # Replicated: global size equals local size
                global_shape.append(local_dim)
            else:
                # Sharded: global size = local size * num_devices
                num_devices = self.layout_mesh.axis_shapes[mesh_axis_idx]
                global_shape.append(local_dim * num_devices)
        
        return tuple(global_shape)


# =============================================================================
# Dynamic/Sparse Sharding Spec
# =============================================================================

class DynamicShardingSpec(ShardingSpec):
    """
    Dynamic sharding spec for sparse/runtime-determined distribution.
    
    Uses inherited `axis` field to specify which dimensions are dynamically routed.
    Used for MoE routing, embedding lookups, etc.
    
    The routing function `fn` determines how data is distributed:
        fn(tensor, tensor_meta, post_sharding_spec) -> routing result
    
    Example:
        def my_routing_fn(tensor, tensor_meta, post_spec):
            # Custom routing logic based on tensor content
            return compute_destinations(tensor, post_spec)
        
        mesh = LayoutMesh(axis_names=("expert",), axis_shapes=(8,))
        spec = DynamicShardingSpec(
            layout_mesh=mesh,
            axis=Axis(0),  # Batch dimension is dynamic
            fn=my_routing_fn,
        )
    """
    # User-provided routing function
    # fn(tensor, tensor_meta, post_sharding_spec) -> routing result
    fn: DynamicRoutingFn
    
    def lower(self, global_shape: Tuple[int, ...], rank_id: int) -> InternalLayout:
        rule = SparseLayoutRule(
            axis=self.axis,
            fn=self.fn,
        )
        
        return InternalLayout(
            type=LayoutType.SPARSE,
            mesh=self.layout_mesh,
            global_shape=global_shape,
            sparse_rule=rule
        )


# =============================================================================
# Raw Sharding Specification (for @operator decorator)
# =============================================================================

class RawShardingDependency(BaseModel):
    """Raw dependency specification (all basic types)."""
    target_tensor_name: str
    mode: str  # "shadow" or "direct_indices"
    target_axis: Union[int, str, None] = 0


class RawShardingSpec(BaseModel):
    """
    Raw sharding specification for @operator decorator.
    
    This is converted to ShardingSpec (StaticShardingSpec/DynamicShardingSpec)
    in @spmd decorator when mesh is available.
    
    For sparse/dynamic sharding, user provides a routing function `fn`:
        fn(tensor, tensor_meta, post_sharding_spec) -> routing result
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: str  # "dense" or "sparse"
    axis: List[Union[int, str, None]] = Field(default_factory=list)
    
    # Sparse-specific field: user-provided routing function
    # fn(tensor, tensor_meta, post_sharding_spec) -> routing result
    fn: Optional[DynamicRoutingFn] = None
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v not in ('dense', 'sparse'):
            raise ValueError(f"type must be 'dense' or 'sparse', got '{v}'")
        return v
    
    @model_validator(mode='after')
    def validate_sparse_has_fn(self) -> 'RawShardingSpec':
        """Validate that sparse sharding has fn."""
        if self.type == 'dense':
            if self.fn is not None:
                raise ValueError(
                    "dense sharding cannot have 'fn' field"
                )
        # elif self.type == 'sparse':
        #     if self.fn is None:
        #         raise ValueError(
        #             "sparse sharding must have 'fn' field"
        #         )
        return self
    
    def to_sharding_spec(self, mesh: LayoutMesh) -> 'ShardingSpec':
        """
        Convert raw spec to ShardingSpec using mesh.
        
        Args:
            mesh: LayoutMesh to resolve axis names
            
        Returns:
            StaticShardingSpec or DynamicShardingSpec
        """
        # 校验 self.axis 是否全部包含在 mesh.axis_names 中
        missing_axes = [a for a in self.axis if a is not None and a not in mesh.axis_names]
        if missing_axes:
            raise ValueError(
                f"ShardingSpec axis {missing_axes} not found in mesh.axis_names {mesh.axis_names}"
            )
        axis_obj = Axis(*self.axis)
        
        if self.type == 'dense':
            return StaticShardingSpec(
                layout_mesh=mesh,
                axis=axis_obj
            )
        else:  # sparse
            return DynamicShardingSpec(
                layout_mesh=mesh,
                axis=axis_obj,
                fn=self.fn,
            )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mesh(
    axis_names: Tuple[str, ...], 
    axis_shapes: Tuple[int, ...]
) -> LayoutMesh:
    """
    Create a device mesh with named axes.
    
    Args:
        axis_names: Ordered names for each axis (e.g., ("data", "model"))
        axis_shapes: Size of each axis (e.g., (4, 2))
        
    Returns:
        LayoutMesh instance
    """
    return LayoutMesh(axis_names=axis_names, axis_shapes=axis_shapes)


# =============================================================================
# Sharding Strategy for Pre-computed Routing
# =============================================================================

class DimShardingInfo(BaseModel):
    """
    单个维度的切分信息。
    
    捕获源端和目标端的分区关系：
    - src_num_shards: 源端 mesh 轴的切片数量（复制时为1）
    - dst_num_shards: 目标端 mesh 轴的切片数量（复制时为1）
    """
    model_config = ConfigDict(frozen=True)
    
    src_num_shards: int  # 源端切片数，复制时为1
    dst_num_shards: int  # 目标端切片数，复制时为1
    is_src_sharded: bool
    is_dst_sharded: bool


def _compute_intersection_range(
    src_coord: int,
    src_num_shards: int,
    dst_coord: int,
    dst_num_shards: int,
) -> bool:
    """
    判断源端和目标端在某个维度上是否有交集。
    
    基于坐标比率来判断：将源端和目标端的坐标归一化到 [0, 1) 区间，
    然后判断区间是否重叠。
    
    Args:
        src_coord: 源端在该轴的坐标
        src_num_shards: 源端该轴的切片数
        dst_coord: 目标端在该轴的坐标
        dst_num_shards: 目标端该轴的切片数
        
    Returns:
        True 如果有交集，否则 False
    """
    # 源端区间: [src_coord/src_num_shards, (src_coord+1)/src_num_shards)
    # 目标端区间: [dst_coord/dst_num_shards, (dst_coord+1)/dst_num_shards)
    
    # 使用分数比较避免浮点误差
    # src_start = src_coord / src_num_shards
    # src_end = (src_coord + 1) / src_num_shards
    # dst_start = dst_coord / dst_num_shards
    # dst_end = (dst_coord + 1) / dst_num_shards
    
    # 交集存在条件: src_start < dst_end AND dst_start < src_end
    # 即: src_coord / src_num_shards < (dst_coord + 1) / dst_num_shards
    #     AND dst_coord / dst_num_shards < (src_coord + 1) / src_num_shards
    
    # 转换为整数比较（交叉相乘）
    # src_coord * dst_num_shards < (dst_coord + 1) * src_num_shards
    # dst_coord * src_num_shards < (src_coord + 1) * dst_num_shards
    
    cond1 = src_coord * dst_num_shards < (dst_coord + 1) * src_num_shards
    cond2 = dst_coord * src_num_shards < (src_coord + 1) * dst_num_shards
    
    return cond1 and cond2


class DenseShardingStrategy(BaseModel):
    """
    Dense-to-Dense 路由的预计算策略。
    
    将 rank 作为构造参数，在初始化时预计算所有不依赖 tensor 形状的信息。
    
    设计原则：
    1. 构造时确定：rank、mesh 几何关系、依赖的目标 rank 列表
    2. 运行时计算：给定 tensor 形状后的具体 slice mask
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 构造参数
    src_rank: int                         # 当前源端 rank（固定）
    src_mesh: "LayoutMesh"
    dst_mesh: "LayoutMesh"
    dim_infos: List[DimShardingInfo]
    src_normalized_axes: Tuple[int, ...]  # 源端 normalized_axes
    dst_normalized_axes: Tuple[int, ...]  # 目标端 normalized_axes
    
    # 预计算结果（在构造时计算）
    src_coords: Tuple[int, ...] = Field(default_factory=tuple)
    dependent_ranks: List[int] = Field(default_factory=list)
    
    @classmethod
    def from_specs(
        cls,
        src_rank: int,
        src_spec: "StaticShardingSpec",
        dst_spec: "StaticShardingSpec",
    ) -> "DenseShardingStrategy":
        """
        从源端和目标端 spec 构建策略。
        
        Args:
            src_rank: 当前源端 SPMD 的 rank
            src_spec: 源端 sharding spec
            dst_spec: 目标端 sharding spec
            
        Returns:
            预计算的路由策略
        """
        src_mesh = src_spec.layout_mesh
        dst_mesh = dst_spec.layout_mesh
        src_normalized = src_spec.normalized_axes
        dst_normalized = dst_spec.normalized_axes
        
        # 1. 提取每个维度的切分信息
        dim_infos = cls._extract_dim_infos(src_mesh, dst_mesh, src_normalized, dst_normalized)
        
        # 2. 计算 src_rank 的坐标
        src_coords = src_mesh.get_coordinate(src_rank)
        
        # 3. 预计算依赖的目标 rank
        dependent_ranks = cls._compute_dependent_ranks(
            src_coords, dim_infos, dst_mesh
        )
        
        return cls(
            src_rank=src_rank,
            src_mesh=src_mesh,
            dst_mesh=dst_mesh,
            dim_infos=dim_infos,
            src_normalized_axes=src_normalized,
            dst_normalized_axes=dst_normalized,
            src_coords=src_coords,
            dependent_ranks=dependent_ranks,
        )
    
    @staticmethod
    def _extract_dim_infos(
        src_mesh: "LayoutMesh",
        dst_mesh: "LayoutMesh",
        src_normalized: Tuple[int, ...],
        dst_normalized: Tuple[int, ...],
    ) -> List[DimShardingInfo]:
        """
        提取每个维度的切分信息。
        
        normalized_axes 中：
        - 1 表示该维度被切分
        - -1 表示该维度被复制
        """
        dim_infos = []
        num_dims = len(src_normalized)
        
        for dim_idx in range(num_dims):
            src_is_sharded = src_normalized[dim_idx] == 1
            dst_is_sharded = dst_normalized[dim_idx] == 1
            
            src_num_shards = src_mesh.axis_shapes[dim_idx] if src_is_sharded else 1
            dst_num_shards = dst_mesh.axis_shapes[dim_idx] if dst_is_sharded else 1
            
            dim_infos.append(DimShardingInfo(
                src_num_shards=src_num_shards,
                dst_num_shards=dst_num_shards,
                is_src_sharded=src_is_sharded,
                is_dst_sharded=dst_is_sharded,
            ))
        
        return dim_infos
    
    @staticmethod
    def _compute_dependent_ranks(
        src_coords: Tuple[int, ...],
        dim_infos: List[DimShardingInfo],
        dst_mesh: "LayoutMesh",
    ) -> List[int]:
        """
        计算依赖的目标 rank 列表。
        
        对于每个目标 rank，检查所有维度是否都有交集。
        如果所有维度都有交集，则该目标 rank 是依赖的。
        """
        dependent = []
        
        for dst_rank in range(dst_mesh.num_devices):
            dst_coords = dst_mesh.get_coordinate(dst_rank)
            
            has_intersection = True
            for dim_idx, dim_info in enumerate(dim_infos):
                # 获取源端和目标端在该维度的坐标
                # 对于未切分的维度，坐标视为0（整个维度）
                src_coord = src_coords[dim_idx] if dim_info.is_src_sharded else 0
                dst_coord = dst_coords[dim_idx] if dim_info.is_dst_sharded else 0
                
                if not _compute_intersection_range(
                    src_coord, dim_info.src_num_shards,
                    dst_coord, dim_info.dst_num_shards
                ):
                    has_intersection = False
                    break
            
            if has_intersection:
                dependent.append(dst_rank)
        
        return dependent
    
    def compute_route(
        self,
        local_shape: Tuple[int, ...],
    ) -> List[Tuple[Tuple[slice, ...], int]]:
        """
        根据给定的 tensor 形状计算路由 mask。
        
        该方法复用 dependent_ranks，只对已知有交集的目标 rank 计算具体 slice。
        
        Args:
            local_shape: 本地 tensor 形状
            
        Returns:
            List[Tuple[本地切片元组, 目标 rank]]
        """
        num_dims = len(local_shape)
        
        # 逆向推导 Global Shape
        global_shape = self._infer_global_shape(local_shape)
        
        # 计算当前 rank 的全局偏移区间
        src_slices = self._compute_rank_slices(
            global_shape, self.src_coords, self.src_normalized_axes, self.src_mesh
        )
        
        results: List[Tuple[Tuple[slice, ...], int]] = []
        
        # 只遍历已知有交集的目标 rank
        for dst_rank in self.dependent_ranks:
            dst_coords = self.dst_mesh.get_coordinate(dst_rank)
            dst_slices = self._compute_rank_slices(
                global_shape, dst_coords, self.dst_normalized_axes, self.dst_mesh
            )
            
            local_slices: List[slice] = []
            has_intersection = True
            
            for dim_idx in range(num_dims):
                src_start, src_end = src_slices[dim_idx]
                dst_start, dst_end = dst_slices[dim_idx]
                
                # 在全局坐标系下计算交集
                global_start = max(src_start, dst_start)
                global_end = min(src_end, dst_end)
                
                if global_start >= global_end:
                    has_intersection = False
                    break
                
                # 转换为本地坐标（相对于 src_start）
                local_start = global_start - src_start
                local_end = global_end - src_start
                
                local_slices.append(slice(local_start, local_end))
            
            if has_intersection:
                results.append((tuple(local_slices), dst_rank))
        
        return results
    
    def _infer_global_shape(self, local_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """从本地形状推导全局形状。"""
        global_shape = []
        for dim_idx, local_dim in enumerate(local_shape):
            if self.src_normalized_axes[dim_idx] == -1:
                # 复制维度：全局大小等于本地大小
                global_shape.append(local_dim)
            else:
                # 切分维度：全局大小 = 本地大小 * 切片数
                global_shape.append(local_dim * self.dim_infos[dim_idx].src_num_shards)
        return tuple(global_shape)
    
    def _compute_rank_slices(
        self,
        global_shape: Tuple[int, ...],
        coords: Tuple[int, ...],
        normalized_axes: Tuple[int, ...],
        mesh: "LayoutMesh",
    ) -> List[Tuple[int, int]]:
        """计算某个 rank 在全局坐标系下的切片范围。"""
        slices = []
        for dim_idx, dim_size in enumerate(global_shape):
            if normalized_axes[dim_idx] == -1:
                # 复制维度：整个范围
                slices.append((0, dim_size))
            else:
                # 切分维度：块分区
                num_devices = mesh.axis_shapes[dim_idx]
                device_pos = coords[dim_idx]
                
                chunk_size = math.ceil(dim_size / num_devices)
                start = device_pos * chunk_size
                end = min(start + chunk_size, dim_size)
                
                slices.append((start, end))
        
        return slices