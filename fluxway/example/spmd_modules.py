"""
SPMD modules for the example.

All SPMDs:
- Use CPU only (no GPU)
- Simulate data transmission only (no actual computation)
- Have total_rank <= 6
- Cover all four transmission combinations:
  - dense -> dense
  - dense -> sparse
  - sparse -> dense
  - sparse -> sparse

Note: DataReaderSPMD has been replaced by the new add_reader() API
which creates a dedicated Reader node that reads data from files.
"""

import logging
import time
from typing import Any, List, Tuple

import numpy as np  # noqa: F401 - used indirectly via tensor operations

from fluxway.common.decorators import spmd, operator
from fluxway.common.types import ShardingSpec
from .data_types import (
    DataState1, DataState2, DataState3,
    Gradient1, Gradient2
)


# =============================================================================
# Dynamic Routing Functions
# =============================================================================

def hash_routing_fn(
    tensor: Any,
    tensor_meta: Any,
    post_spec: ShardingSpec
) -> List[Tuple[Tuple[slice, ...], int]]:
    """
    Hash-based routing function for dynamic sharding.
    
    Routes data based on hash of batch indices to target ranks.
    
    Args:
        tensor: The tensor data to route
        tensor_meta: TensorMeta with global_indices info
        post_spec: Destination sharding spec (provides mesh info)
    
    Returns:
        List of (slice, target_rank) tuples
    """
    num_ranks = post_spec.layout_mesh.num_devices
    results = []
    
    # Get batch size from tensor
    if hasattr(tensor, 'shape'):
        batch_size = tensor.shape[0]
    else:
        batch_size = len(tensor) if hasattr(tensor, '__len__') else 1
    
    # Route each batch element by hash
    for i in range(batch_size):
        target_rank = hash(i) % num_ranks
        results.append((slice(i, i + 1),), target_rank)
    
    return results


def modulo_routing_fn(
    tensor: Any,
    tensor_meta: Any,
    post_spec: ShardingSpec
) -> List[Tuple[Tuple[slice, ...], int]]:
    """
    Modulo-based (cyclic) routing function for dynamic sharding.
    
    Routes data using round-robin distribution.
    
    Args:
        tensor: The tensor data to route
        tensor_meta: TensorMeta with global_indices info
        post_spec: Destination sharding spec (provides mesh info)
    
    Returns:
        List of (slice, target_rank) tuples
    """
    num_ranks = post_spec.layout_mesh.num_devices
    results = []
    
    # Get batch size from tensor
    if hasattr(tensor, 'shape'):
        batch_size = tensor.shape[0]
    else:
        batch_size = len(tensor) if hasattr(tensor, '__len__') else 1
    
    # Route each batch element by modulo (round-robin)
    for i in range(batch_size):
        target_rank = i % num_ranks
        results.append(((slice(i, i + 1),), target_rank))
    
    return results


def expert_routing_fn(
    tensor: Any,
    tensor_meta: Any,
    post_spec: ShardingSpec
) -> List[Tuple[Tuple[slice, ...], int]]:
    """
    Expert-based routing function for MoE (Mixture of Experts).
    
    Routes data based on expert assignment derived from tensor content.
    
    Args:
        tensor: The tensor data to route
        tensor_meta: TensorMeta with global_indices info  
        post_spec: Destination sharding spec (provides mesh info)
    
    Returns:
        List of (slice, target_rank) tuples
    """
    num_experts = post_spec.layout_mesh.num_devices
    results = []
    
    # Get batch size from tensor
    if hasattr(tensor, 'shape'):
        batch_size = tensor.shape[0]
    else:
        batch_size = len(tensor) if hasattr(tensor, '__len__') else 1
    
    # Simulate expert assignment (in real MoE, this would be from gating network)
    for i in range(batch_size):
        # Use sum of first few elements to determine expert
        if hasattr(tensor, '__getitem__') and hasattr(tensor[i], 'sum'):
            expert_id = int(abs(tensor[i].sum())) % num_experts
        else:
            expert_id = i % num_experts
        results.append(((slice(i, i + 1),), expert_id))
    
    return results

logger = logging.getLogger(__name__)


# =============================================================================
# Note: Data reading is now handled by the Reader node using add_reader() API
# The DataReaderSPMD class has been removed as it's replaced by the new
# Reader functionality that actively reads data from files with backpressure
# support.
# =============================================================================


# =============================================================================
# Dense -> Dense SPMD
# =============================================================================

@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("data", "tensor"),
    mesh_axis_shapes=(2, 2),
    max_async_steps=2,
    params_update_steps=1,
)
class DenseDenseSPMD:
    """
    SPMD with dense -> dense transmission.
    
    Receives dense data, outputs dense data.
    """
    
    @operator(
        name="forward",
        input_sharding={
            "tensor": {"type": "dense", "axis": ["data"]}
        },
        output_sharding={
            "tensor": {"type": "dense", "axis": ["data", "tensor"]}
        },
        state_argument=["batch_id", "metadata"],
        tensor_argument=["tensor"]
    )
    def forward(self, state: DataState1) -> DataState2:
        """Process dense input and output dense data."""
        start_time = time.time()
        logger.info(
            f"[DenseDenseSPMD.forward] batch_id={state.batch_id}, "
            f"tensor_shape={state.tensor.shape}, time={start_time:.6f}"
        )
        
        # Simulate transmission: just pass through with shape preserved
        output_tensor = state.tensor.copy()
        
        # Simulate processing time
        time.sleep(1)
        
        end_time = time.time()
        logger.info(
            f"[DenseDenseSPMD.forward] completed batch_id={state.batch_id}, "
            f"output_shape={output_tensor.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return DataState2(
            batch_id=state.batch_id,
            tensor=output_tensor,
            processed=True
        )
    
    @operator(
        name="backward",
        input_sharding={
            "tensor": {"type": "dense", "axis": ["data", "tensor"]}
        },
        output_sharding={
            "tensor": {"type": "dense", "axis": ["data"]}
        },
        state_argument=["batch_id", "grad_step"],
        tensor_argument=["tensor"]
    )
    def backward(self, grad: Gradient2) -> Gradient1:
        """Backward pass: dense -> dense."""
        start_time = time.time()
        logger.info(
            f"[DenseDenseSPMD.backward] batch_id={grad.batch_id}, "
            f"grad_shape={grad.tensor.shape}, time={start_time:.6f}"
        )
        
        # Simulate gradient transmission
        output_grad = grad.tensor.copy()
        
        # Simulate processing time
        time.sleep(1)
        
        end_time = time.time()
        logger.info(
            f"[DenseDenseSPMD.backward] completed batch_id={grad.batch_id}, "
            f"output_shape={output_grad.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return Gradient1(
            batch_id=grad.batch_id,
            tensor=output_grad,
            grad_step=grad.grad_step + 1
        )


# =============================================================================
# Dense -> Sparse SPMD
# =============================================================================

@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("expert",),
    mesh_axis_shapes=(4,),
    max_async_steps=2,
    params_update_steps=1,
)
class DenseSparseSPMD:
    """
    SPMD with dense -> sparse transmission.
    
    Receives dense data, outputs sparse data using dynamic routing.
    Uses expert_routing_fn for MoE-style routing.
    """
    
    @operator(
        name="forward",
        input_sharding={
            "tensor": {"type": "dense", "axis": ["expert"]}
        },
        output_sharding={
            # Sparse output: uses fn for dynamic routing
            "tensor": {"type": "sparse", "axis": ["expert"], "fn": expert_routing_fn}
        },
        state_argument=["batch_id", "metadata"],
        tensor_argument=["tensor"]
    )
    def forward(self, state: DataState1) -> DataState2:
        """Process dense input and output sparse data (MoE routing)."""
        start_time = time.time()
        logger.info(
            f"[DenseSparseSPMD.forward] batch_id={state.batch_id}, "
            f"tensor_shape={state.tensor.shape}, time={start_time:.6f}"
        )
        
        # Simulate MoE computation: route different parts to different experts
        output_tensor = state.tensor.copy()
        
        # Simulate processing time
        time.sleep(0.5)
        
        end_time = time.time()
        logger.info(
            f"[DenseSparseSPMD.forward] completed batch_id={state.batch_id}, "
            f"output_shape={output_tensor.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return DataState2(
            batch_id=state.batch_id,
            tensor=output_tensor,
            processed=True
        )


# =============================================================================
# Sparse -> Dense SPMD  
# =============================================================================

@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("data",),
    mesh_axis_shapes=(4,),
    max_async_steps=2,
    params_update_steps=1,
)
class SparseDenseSPMD:
    """
    SPMD with sparse -> dense transmission.
    
    Receives sparse data (from dynamic routing), outputs dense data.
    Uses modulo_routing_fn for input routing.
    """
    
    @operator(
        name="forward",
        input_sharding={
            # Sparse input: uses fn for dynamic routing
            "tensor": {"type": "sparse", "axis": ["data"], "fn": modulo_routing_fn}
        },
        output_sharding={
            # Dense output: regular block sharding
            "tensor": {"type": "dense", "axis": ["data"]}
        },
        state_argument=["batch_id", "processed"],
        tensor_argument=["tensor"]
    )
    def forward(self, state: DataState2) -> DataState3:
        """Process sparse input and output dense data."""
        start_time = time.time()
        logger.info(
            f"[SparseDenseSPMD.forward] batch_id={state.batch_id}, "
            f"tensor_shape={state.tensor.shape}, time={start_time:.6f}"
        )
        
        # Aggregate sparse inputs to dense output
        output_tensor = state.tensor.copy()
        
        # Simulate processing time
        time.sleep(0.5)
        
        end_time = time.time()
        logger.info(
            f"[SparseDenseSPMD.forward] completed batch_id={state.batch_id}, "
            f"output_shape={output_tensor.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return DataState3(
            batch_id=state.batch_id,
            tensor=output_tensor,
            stage=1
        )


# =============================================================================
# Sparse -> Sparse SPMD
# =============================================================================

@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("expert",),
    mesh_axis_shapes=(4,),
    max_async_steps=2,
    params_update_steps=1,
)
class SparseSparseSPMD:
    """
    SPMD with sparse -> sparse transmission.
    
    Both input and output use dynamic routing.
    Uses hash_routing_fn for both input and output routing.
    """
    
    @operator(
        name="forward",
        input_sharding={
            # Sparse input
            "tensor": {"type": "sparse", "axis": ["expert"], "fn": hash_routing_fn}
        },
        output_sharding={
            # Sparse output
            "tensor": {"type": "sparse", "axis": ["expert"], "fn": expert_routing_fn}
        },
        state_argument=["batch_id", "processed"],
        tensor_argument=["tensor"]
    )
    def forward(self, state: DataState2) -> DataState2:
        """Process sparse input and output sparse data."""
        start_time = time.time()
        logger.info(
            f"[SparseSparseSPMD.forward] batch_id={state.batch_id}, "
            f"tensor_shape={state.tensor.shape}, time={start_time:.6f}"
        )
        
        # Transform sparse data
        output_tensor = state.tensor.copy() * 2.0  # Simple transformation
        
        # Simulate processing time
        time.sleep(0.5)
        
        end_time = time.time()
        logger.info(
            f"[SparseSparseSPMD.forward] completed batch_id={state.batch_id}, "
            f"output_shape={output_tensor.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return DataState2(
            batch_id=state.batch_id,
            tensor=output_tensor,
            processed=True
        )
    
    @operator(
        name="backward",
        input_sharding={
            "tensor": {"type": "sparse", "axis": ["expert"], "fn": expert_routing_fn}
        },
        output_sharding={
            "tensor": {"type": "sparse", "axis": ["expert"], "fn": hash_routing_fn}
        },
        state_argument=["batch_id", "grad_step"],
        tensor_argument=["tensor"]
    )
    def backward(self, grad: Gradient2) -> Gradient2:
        """Backward pass: sparse -> sparse."""
        start_time = time.time()
        logger.info(
            f"[SparseSparseSPMD.backward] batch_id={grad.batch_id}, "
            f"grad_shape={grad.tensor.shape}, time={start_time:.6f}"
        )
        
        output_grad = grad.tensor.copy() * 0.5  # Gradient scaling
        
        # Simulate processing time
        time.sleep(0.5)
        
        end_time = time.time()
        logger.info(
            f"[SparseSparseSPMD.backward] completed batch_id={grad.batch_id}, "
            f"output_shape={output_grad.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return Gradient2(
            batch_id=grad.batch_id,
            tensor=output_grad,
            grad_step=grad.grad_step + 1
        )


# =============================================================================
# Dense -> Dense SPMD (Additional)
# =============================================================================

@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("data", "tensor"),
    mesh_axis_shapes=(4, 2),
    max_async_steps=2,
    params_update_steps=1,
)
class DenseDenseSPMD1:
    """
    SPMD with dense -> dense transmission (additional example).
    """
    
    @operator(
        name="forward",
        input_sharding={
            "tensor": {"type": "dense", "axis": ["data", "tensor"]}
        },
        output_sharding={
            "tensor": {"type": "dense", "axis": ["data", "tensor"]}
        },
        state_argument=["batch_id", "processed"],
        tensor_argument=["tensor"]
    )
    def forward(self, state: DataState2) -> Gradient2:
        """Forward pass: dense -> dense."""
        start_time = time.time()
        logger.info(
            f"[DenseDenseSPMD1.forward] batch_id={state.batch_id}, "
            f"time={start_time:.6f}"
        )
        
        output_tensor = state.tensor.copy()
        
        # Simulate processing time
        time.sleep(0.5)
        
        end_time = time.time()
        logger.info(
            f"[DenseDenseSPMD1.forward] completed batch_id={state.batch_id}, "
            f"output_shape={output_tensor.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return Gradient2(
            batch_id=state.batch_id,
            tensor=output_tensor,
            grad_step=1
        )

    @operator(
        name="backward",
        input_sharding={
            "tensor": {"type": "dense", "axis": ["data", "tensor"]}
        },
        output_sharding={
            "tensor": {"type": "dense", "axis": ["data", "tensor"]}
        },
        state_argument=["batch_id", "grad_step"],
        tensor_argument=["tensor"]
    )
    def backward(self, grad: Gradient2) -> Gradient2:
        """Backward pass: dense -> dense."""
        start_time = time.time()
        logger.info(
            f"[DenseDenseSPMD1.backward] batch_id={grad.batch_id}, "
            f"time={start_time:.6f}"
        )

        time.sleep(0.5)
        output_grad = grad.tensor.copy()
        
        end_time = time.time()
        logger.info(
            f"[DenseDenseSPMD1.backward] completed batch_id={grad.batch_id}, "
            f"output_shape={output_grad.shape}, duration={end_time-start_time:.6f}s"
        )
        
        return Gradient2(
            batch_id=grad.batch_id,
            tensor=output_grad,
            grad_step=grad.grad_step + 1
        )
