"""
Fluxway SPMD decorator.

Provides the @spmd decorator to mark user-defined classes
with SPMD execution configuration.
"""

from typing import Dict, Type, TypeVar, Optional, Any

from ..types import SPMDConfig, ResourceSpec, PartitionType

T = TypeVar('T')


def spmd(
    resource: Optional[Dict[str, Any]] = None,
    mesh: Optional[Dict[PartitionType, int]] = None,
    input_sharding: Optional[Dict[PartitionType, int]] = None,
    output_sharding: Optional[Dict[PartitionType, int]] = None,
    max_async_steps: int = 1,
    params_update_steps: int = 1
):
    """
    Decorator to mark a class with SPMD execution configuration.
    
    Usage:
        @spmd(
            resource={"nums_cpu": 4, "nums_gpu": 2},
            mesh={PartitionType.DP: 4, PartitionType.TP: 2},
            input_sharding={PartitionType.DP: 4},
            max_async_steps=10
        )
        class MySPMD:
            def forward(self, state):
                return process(state)
    
    Args:
        resource: Resource requirements dict
        mesh: Mesh topology spec Dict[PartitionType, int]
        input_sharding: Input sharding spec Dict[PartitionType, int]
        output_sharding: Output sharding spec Dict[PartitionType, int]
        max_async_steps: Maximum async steps between workers
        params_update_steps: Sync barrier every N steps
        
    Returns:
        Decorated class with SPMD configuration
    """
    # Parse resource specification
    resource_spec = None
    if resource:
        resource_spec = ResourceSpec(
            num_cpus=resource.get('nums_cpu', resource.get('num_cpus', 1)),
            num_gpus=resource.get('nums_gpu', resource.get('num_gpus', 0)),
            memory_gb=resource.get('memory', resource.get('memory_gb', 0.0))
        )
    
    def decorator(cls: Type[T]) -> Type[T]:
        # Add attributes directly to the class
        cls._resource = resource_spec
        cls._mesh = mesh
        cls._input_sharding = input_sharding
        cls._output_sharding = output_sharding
        cls._max_async_steps = max_async_steps
        cls._params_update_steps = params_update_steps
        
        # Add methods directly to the class
        cls.get_resource = classmethod(lambda c: c._resource)
        cls.get_mesh = classmethod(lambda c: c._mesh)
        cls.get_input_sharding = classmethod(lambda c: c._input_sharding)
        cls.get_output_sharding = classmethod(lambda c: c._output_sharding)
        cls.get_max_async_steps = classmethod(lambda c: c._max_async_steps)
        cls.get_params_update_steps = classmethod(lambda c: c._params_update_steps)
        cls.is_fluxway_spmd = classmethod(lambda c: True)
        
        def get_spmd_config(c) -> SPMDConfig:
            forward_fn = getattr(c, 'forward', None)
            backward_fn = getattr(c, 'backward', None)
            update_fn = getattr(c, 'update', None)
            return SPMDConfig(
                mesh=c._mesh,
                input_sharding=c._input_sharding,
                output_sharding=c._output_sharding,
                user_forward_fn=forward_fn,
                user_backward_fn=backward_fn,
                user_update_fn=update_fn,
                max_async_steps=c._max_async_steps,
                params_update_steps=c._params_update_steps
            )
        cls.get_spmd_config = classmethod(get_spmd_config)
        
        return cls
    
    return decorator
