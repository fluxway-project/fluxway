"""
Tests for Scheduler and PGManager components.

These tests require Ray to be installed and running.
"""

import pytest
import ray
import asyncio

from fluxway.common.types import (
    Graph, Node, Edge, ResourceSpec, PlacementPlan, 
    PlacementStrategy, ResourcePriority, SPMDConfig
)
from fluxway.control_plane.scheduler import Scheduler, PGManager
from fluxway.control_plane.spmd_manager import SPMDManager


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    """Initialize Ray for tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    # Don't shutdown - let subsequent tests reuse


class TestPGManager:
    """Tests for PGManager class."""
    
    def test_init(self):
        """Test PGManager initialization."""
        pg_manager = PGManager()
        assert pg_manager._spmd_to_pg == {}
        assert pg_manager._spmd_to_resource == {}
        assert pg_manager._spmd_to_priority == {}
    
    def test_create_pg(self):
        """Test create_pg creates a placement group."""
        pg_manager = PGManager()
        resource = ResourceSpec(num_cpus=1)  # Only 1 CPU, no GPU
        
        pg = asyncio.run(
            pg_manager.create_pg(
                spmd_id="test_spmd",
                resource=resource,
                priority=ResourcePriority.HIGH,
            )
        )
        
        assert pg is not None
        assert pg_manager.get_resource("test_spmd") == resource
        assert pg_manager.get_priority("test_spmd") == ResourcePriority.HIGH
        
        # Cleanup
        pg_manager.release_pg("test_spmd")
    
    def test_get_pg_not_exists(self):
        """Test get_pg returns None for non-existent SPMD."""
        pg_manager = PGManager()
        assert pg_manager.get_pg("nonexistent") is None
    
    def test_get_priority_default(self):
        """Test get_priority returns MID as default."""
        pg_manager = PGManager()
        assert pg_manager.get_priority("unknown") == ResourcePriority.MID
    
    def test_release_pg(self):
        """Test release_pg removes mappings."""
        pg_manager = PGManager()
        resource = ResourceSpec(num_cpus=1)
        
        asyncio.run(pg_manager.create_pg("spmd1", resource, ResourcePriority.LOW))
        assert pg_manager.get_resource("spmd1") is not None
        
        result = pg_manager.release_pg("spmd1")
        assert result is True
        assert pg_manager.get_resource("spmd1") is None
        assert pg_manager.get_priority("spmd1") == ResourcePriority.MID  # default
    
    def test_release_pg_not_exists(self):
        """Test release_pg returns False for non-existent SPMD."""
        pg_manager = PGManager()
        assert pg_manager.release_pg("nonexistent") is False
    
    def test_get_all_spmd_ids(self):
        """Test get_all_spmd_ids returns all SPMD IDs with resources."""
        pg_manager = PGManager()
        resource = ResourceSpec(num_cpus=1)
        
        asyncio.run(pg_manager.create_pg("spmd1", resource))
        asyncio.run(pg_manager.create_pg("spmd2", resource))
        asyncio.run(pg_manager.create_pg("spmd3", resource))
        
        spmd_ids = pg_manager.get_all_spmd_ids()
        assert set(spmd_ids) == {"spmd1", "spmd2", "spmd3"}
        
        # Cleanup
        for spmd_id in spmd_ids:
            pg_manager.release_pg(spmd_id)
    
    def test_get_preemptable_spmds_sorted_by_priority(self):
        """Test get_preemptable_spmds returns SPMDs sorted by priority."""
        pg_manager = PGManager()
        resource = ResourceSpec(num_cpus=1)
        
        asyncio.run(pg_manager.create_pg("high", resource, ResourcePriority.HIGH))
        asyncio.run(pg_manager.create_pg("low", resource, ResourcePriority.LOW))
        asyncio.run(pg_manager.create_pg("mid", resource, ResourcePriority.MID))
        
        preemptable = pg_manager.get_preemptable_spmds(ResourceSpec(num_cpus=1))
        
        # Should be sorted: LOW, MID, HIGH
        assert preemptable[0] == "low"
        assert preemptable[-1] == "high"
        
        # Cleanup
        for spmd_id in ["high", "low", "mid"]:
            pg_manager.release_pg(spmd_id)
    
    def test_preempt_not_implemented(self):
        """Test preempt returns False (not implemented)."""
        pg_manager = PGManager()
        assert pg_manager.preempt("spmd1", "test reason") is False
    
    def test_evict_not_implemented(self):
        """Test evict returns False (not implemented)."""
        pg_manager = PGManager()
        assert pg_manager.evict("spmd1") is False


class TestScheduler:
    """Tests for Scheduler class."""
    
    def test_init_without_spmd_manager(self):
        """Test Scheduler initialization without spmd_manager."""
        scheduler = Scheduler()
        assert scheduler.spmd_manager is None
        assert scheduler.pg_manager is not None
    
    def test_init_with_spmd_manager(self):
        """Test Scheduler initialization with spmd_manager."""
        spmd_manager = SPMDManager()
        scheduler = Scheduler(spmd_manager=spmd_manager)
        assert scheduler.spmd_manager is spmd_manager
    
    def test_schedule_simple_graph(self):
        """Test schedule with a simple graph."""
        scheduler = Scheduler()
        
        node = Node(name="test_node", resource=ResourceSpec(num_cpus=1))
        graph = Graph(name="test_graph", nodes={"test_node": node})
        
        success = asyncio.run(scheduler.schedule(graph))
        assert success is True
        
        # Cleanup
        scheduler.release_resources("test_node")
    
    def test_schedule_with_node_priority(self):
        """Test schedule respects node priority."""
        scheduler = Scheduler()
        
        node = Node(
            name="high_priority_node",
            resource=ResourceSpec(num_cpus=1),
            priority=ResourcePriority.HIGH
        )
        graph = Graph(name="test", nodes={"high_priority_node": node})
        
        success = asyncio.run(scheduler.schedule(graph))
        assert success is True
        
        # Check priority was applied
        priority = scheduler.get_spmd_priority("high_priority_node")
        assert priority == ResourcePriority.HIGH
        
        # Cleanup
        scheduler.release_resources("high_priority_node")
    
    def test_schedule_default_resource(self):
        """Test schedule assigns default resource when node has none."""
        scheduler = Scheduler()
        
        node = Node(name="no_resource_node")  # No resource specified
        graph = Graph(name="test", nodes={"no_resource_node": node})
        
        success = asyncio.run(scheduler.schedule(graph))
        assert success is True
        
        # Should have default resource
        resource = scheduler.get_allocated_resources("no_resource_node")
        assert resource is not None
        assert resource.num_cpus == 1
        
        # Cleanup
        scheduler.release_resources("no_resource_node")
    
    def test_schedule_multi_node_graph(self):
        """Test schedule with multiple nodes."""
        scheduler = Scheduler()
        
        nodes = {
            "node1": Node(name="node1", resource=ResourceSpec(num_cpus=1)),
            "node2": Node(name="node2", resource=ResourceSpec(num_cpus=1)),
            "node3": Node(name="node3", resource=ResourceSpec(num_cpus=1)),
        }
        graph = Graph(name="multi", nodes=nodes)
        
        success = asyncio.run(scheduler.schedule(graph))
        assert success is True
        
        # Cleanup
        for node_name in nodes:
            scheduler.release_resources(node_name)
    
    def test_generate_plan(self):
        """Test _generate_plan returns correct PlacementPlan."""
        scheduler = Scheduler()
        
        node = Node(
            name="test", 
            resource=ResourceSpec(num_cpus=1),
            priority=ResourcePriority.LOW
        )
        graph = Graph(name="test", nodes={"test": node})
        
        plan = scheduler._generate_plan(graph)
        
        assert isinstance(plan, PlacementPlan)
        assert "test" in plan.spmd_placements
        assert plan.spmd_placements["test"].num_cpus == 1
        assert plan.spmd_priorities["test"] == ResourcePriority.LOW
        assert plan.strategy == PlacementStrategy.PACK
    
    def test_release_resources(self):
        """Test release_resources cleans up allocations."""
        scheduler = Scheduler()
        
        node = Node(name="test", resource=ResourceSpec(num_cpus=1))
        graph = Graph(name="test", nodes={"test": node})
        
        asyncio.run(scheduler.schedule(graph))
        assert scheduler.get_allocated_resources("test") is not None
        
        scheduler.release_resources("test")
        assert scheduler.get_allocated_resources("test") is None
    
    def test_preempt_spmd(self):
        """Test preempt_spmd delegates to pg_manager."""
        scheduler = Scheduler()
        result = scheduler.preempt_spmd("test", "reason")
        assert result is False  # Not implemented
    
    def test_evict_spmd(self):
        """Test evict_spmd delegates to pg_manager."""
        scheduler = Scheduler()
        result = scheduler.evict_spmd("test")
        assert result is False  # Not implemented
    
    def test_get_preemptable_spmds(self):
        """Test get_preemptable_spmds returns sorted list."""
        scheduler = Scheduler()
        
        nodes = {
            "high": Node(name="high", priority=ResourcePriority.HIGH, resource=ResourceSpec(num_cpus=1)),
            "low": Node(name="low", priority=ResourcePriority.LOW, resource=ResourceSpec(num_cpus=1)),
        }
        graph = Graph(name="test", nodes=nodes)
        asyncio.run(scheduler.schedule(graph))
        
        preemptable = scheduler.get_preemptable_spmds(ResourceSpec(num_cpus=1))
        assert preemptable[0] == "low"
        
        # Cleanup
        for node_name in nodes:
            scheduler.release_resources(node_name)


class TestSchedulerWithSPMDManager:
    """Tests for Scheduler with SPMDManager integration."""
    
    def test_schedule_establishes_topology(self):
        """Test schedule establishes topology via spmd_manager."""
        spmd_manager = SPMDManager()
        scheduler = Scheduler(spmd_manager=spmd_manager)
        
        nodes = {
            "a": Node(name="a", resource=ResourceSpec(num_cpus=1)),
            "b": Node(name="b", resource=ResourceSpec(num_cpus=1)),
        }
        edges = [Edge(source_node="a", target_node="b")]
        graph = Graph(name="test", nodes=nodes, edges=edges)
        
        success = asyncio.run(scheduler.schedule(graph))
        assert success is True
        
        # Cleanup
        for node_name in nodes:
            scheduler.release_resources(node_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
