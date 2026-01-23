"""
Unit tests for DenseShardingStrategy.

Tests the pre-computed sharding strategy implementation for dense-to-dense routing.
"""

import pytest
from typing import Tuple, List

from fluxway.common.types.sharding import (
    LayoutMesh,
    StaticShardingSpec,
    Axis,
    DenseShardingStrategy,
    DimShardingInfo,
    _compute_intersection_range,
)


class TestComputeIntersectionRange:
    """Test the intersection range computation helper."""
    
    def test_same_sharding(self):
        """Same sharding: only same coord has intersection."""
        # src has 4 shards, dst has 4 shards
        # coord 0 only intersects with coord 0
        assert _compute_intersection_range(0, 4, 0, 4) is True
        assert _compute_intersection_range(0, 4, 1, 4) is False
        assert _compute_intersection_range(1, 4, 1, 4) is True
        
    def test_different_sharding(self):
        """Different sharding: compute proper intersection."""
        # src has 2 shards, dst has 4 shards
        # src coord 0 owns [0, 0.5), should intersect with dst coords 0,1
        assert _compute_intersection_range(0, 2, 0, 4) is True   # [0,0.5) ∩ [0,0.25)
        assert _compute_intersection_range(0, 2, 1, 4) is True   # [0,0.5) ∩ [0.25,0.5)
        assert _compute_intersection_range(0, 2, 2, 4) is False  # [0,0.5) ∩ [0.5,0.75)
        assert _compute_intersection_range(0, 2, 3, 4) is False
        
        # src coord 1 owns [0.5, 1), should intersect with dst coords 2,3
        assert _compute_intersection_range(1, 2, 0, 4) is False
        assert _compute_intersection_range(1, 2, 1, 4) is False
        assert _compute_intersection_range(1, 2, 2, 4) is True
        assert _compute_intersection_range(1, 2, 3, 4) is True
    
    def test_replicated_src(self):
        """Replicated source: should intersect with all dst."""
        # src has 1 shard (replicated), dst has 4 shards
        assert _compute_intersection_range(0, 1, 0, 4) is True
        assert _compute_intersection_range(0, 1, 1, 4) is True
        assert _compute_intersection_range(0, 1, 2, 4) is True
        assert _compute_intersection_range(0, 1, 3, 4) is True
    
    def test_replicated_dst(self):
        """Replicated destination: should intersect with all src."""
        # src has 4 shards, dst has 1 shard (replicated)
        assert _compute_intersection_range(0, 4, 0, 1) is True
        assert _compute_intersection_range(1, 4, 0, 1) is True
        assert _compute_intersection_range(2, 4, 0, 1) is True
        assert _compute_intersection_range(3, 4, 0, 1) is True


class TestDenseShardingStrategy:
    """Test DenseShardingStrategy class."""
    
    @pytest.fixture
    def mesh_1d_4(self) -> LayoutMesh:
        """1D mesh with 4 devices."""
        return LayoutMesh(axis_names=("x",), axis_shapes=(4,))
    
    @pytest.fixture
    def mesh_1d_2(self) -> LayoutMesh:
        """1D mesh with 2 devices."""
        return LayoutMesh(axis_names=("x",), axis_shapes=(2,))
    
    @pytest.fixture
    def mesh_2d_2x2(self) -> LayoutMesh:
        """2D mesh with 2x2 devices."""
        return LayoutMesh(axis_names=("x", "y"), axis_shapes=(2, 2))
    
    def test_same_mesh_same_sharding(self, mesh_1d_4):
        """Same mesh and sharding: diagonal pattern."""
        src_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        
        for rank in range(4):
            strategy = DenseShardingStrategy.from_specs(rank, src_spec, dst_spec)
            # Each rank should only depend on itself
            assert strategy.dependent_ranks == [rank]
            assert strategy.src_rank == rank
    
    def test_different_sharding_counts(self, mesh_1d_2, mesh_1d_4):
        """Different shard counts: 2 -> 4."""
        src_spec = StaticShardingSpec(layout_mesh=mesh_1d_2, axis=Axis("x"))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        
        # src rank 0 should depend on dst ranks 0, 1
        strategy_0 = DenseShardingStrategy.from_specs(0, src_spec, dst_spec)
        assert strategy_0.dependent_ranks == [0, 1]
        
        # src rank 1 should depend on dst ranks 2, 3
        strategy_1 = DenseShardingStrategy.from_specs(1, src_spec, dst_spec)
        assert strategy_1.dependent_ranks == [2, 3]
    
    def test_replicated_to_sharded(self, mesh_1d_4):
        """Replicated source to sharded destination."""
        # src is replicated (None axis)
        src_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis(None))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        
        # Every src rank should depend on all dst ranks
        for rank in range(4):
            strategy = DenseShardingStrategy.from_specs(rank, src_spec, dst_spec)
            assert strategy.dependent_ranks == [0, 1, 2, 3]
    
    def test_sharded_to_replicated(self, mesh_1d_4):
        """Sharded source to replicated destination."""
        src_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis(None))
        
        # Every src rank should depend on all dst ranks (broadcast)
        for rank in range(4):
            strategy = DenseShardingStrategy.from_specs(rank, src_spec, dst_spec)
            assert strategy.dependent_ranks == [0, 1, 2, 3]
    
    def test_compute_route_same_sharding(self, mesh_1d_4):
        """Test compute_route with same sharding."""
        src_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        
        strategy = DenseShardingStrategy.from_specs(0, src_spec, dst_spec)
        
        # Local shape (100,), meaning global shape (400,)
        # Rank 0 owns [0, 100), dst rank 0 also owns [0, 100)
        routes = strategy.compute_route((100,))
        
        assert len(routes) == 1
        local_slice, dst_rank = routes[0]
        assert dst_rank == 0
        assert local_slice == (slice(0, 100),)
    
    def test_compute_route_different_sharding(self, mesh_1d_2, mesh_1d_4):
        """Test compute_route with different sharding."""
        src_spec = StaticShardingSpec(layout_mesh=mesh_1d_2, axis=Axis("x"))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_1d_4, axis=Axis("x"))
        
        strategy = DenseShardingStrategy.from_specs(0, src_spec, dst_spec)
        
        # Local shape (100,), meaning global shape (200,)
        # Rank 0 owns [0, 100)
        # dst rank 0 owns [0, 50), dst rank 1 owns [50, 100)
        routes = strategy.compute_route((100,))
        
        assert len(routes) == 2
        
        # Sort by dst_rank for stable comparison
        routes_sorted = sorted(routes, key=lambda x: x[1])
        
        # dst rank 0: local slice [0, 50)
        assert routes_sorted[0] == ((slice(0, 50),), 0)
        # dst rank 1: local slice [50, 100)
        assert routes_sorted[1] == ((slice(50, 100),), 1)


class TestDenseShardingStrategy2D:
    """Test DenseShardingStrategy with 2D mesh."""
    
    @pytest.fixture
    def mesh_2d_2x2(self) -> LayoutMesh:
        """2D mesh with 2x2 devices."""
        return LayoutMesh(axis_names=("x", "y"), axis_shapes=(2, 2))
    
    def test_2d_same_sharding(self, mesh_2d_2x2):
        """2D mesh with same sharding: only self dependency."""
        src_spec = StaticShardingSpec(layout_mesh=mesh_2d_2x2, axis=Axis("x", "y"))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_2d_2x2, axis=Axis("x", "y"))
        
        for rank in range(4):
            strategy = DenseShardingStrategy.from_specs(rank, src_spec, dst_spec)
            assert strategy.dependent_ranks == [rank]
    
    def test_2d_partial_sharding(self, mesh_2d_2x2):
        """2D mesh with partial sharding."""
        # src: shard on x only, dst: shard on both
        src_spec = StaticShardingSpec(layout_mesh=mesh_2d_2x2, axis=Axis("x", None))
        dst_spec = StaticShardingSpec(layout_mesh=mesh_2d_2x2, axis=Axis("x", "y"))
        
        # src rank 0 is coord (0, 0), owns x=[0,1), y=[0,2) (full y)
        # dst ranks with x=0: ranks 0 (0,0) and 1 (0,1)
        strategy_0 = DenseShardingStrategy.from_specs(0, src_spec, dst_spec)
        assert set(strategy_0.dependent_ranks) == {0, 1}
        
        # src rank 2 is coord (1, 0), owns x=[1,2), y=[0,2)
        # dst ranks with x=1: ranks 2 (1,0) and 3 (1,1)
        strategy_2 = DenseShardingStrategy.from_specs(2, src_spec, dst_spec)
        assert set(strategy_2.dependent_ranks) == {2, 3}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
