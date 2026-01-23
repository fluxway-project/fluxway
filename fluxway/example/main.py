"""
Main entry point for the Fluxway example.

This example demonstrates:
1. All SPMDs use CPU only (no GPU)
2. All operations simulate data transmission only
3. Single SPMD total_rank <= 6
4. Four transmission combinations: dense->dense, dense->sparse, sparse->dense, sparse->sparse
5. Bidirectional edges and forward/backward edge differences
6. Complete logging with timestamps
7. Both tensor and state data transmission
8. Reader node using add_reader() API for data reading
9. Backpressure handling between Reader and SPMD nodes
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to Python path
# This allows the script to be run from any directory
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray is not available. Please install ray: pip install ray")
    sys.exit(1)

# Import fluxway modules after path setup
# Note: These imports must come after sys.path modification above
from fluxway import Builder  # noqa: E402
from fluxway.control_plane import ControllerActor  # noqa: E402
from fluxway.common.types import Graph, HeartbeatConfig, AsyncStepConfig  # noqa: E402
from fluxway.common.types.data_types import OpType  # noqa: E402

# Use absolute imports to support both module and direct execution
from fluxway.example.spmd_modules import (  # noqa: E402
    DenseDenseSPMD, DenseDenseSPMD1,
)

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'
)
logger = logging.getLogger(__name__)




def build_example_graph() -> Graph:
    """
    Build the example graph with:
    - DataReader (Reader node) -> DenseDenseSPMD -> DenseDenseSPMD1
    - Bidirectional edges
    - Forward/backward edge differences
    - New add_reader() API for data reading
    """
    builder = Builder()
    
    # Add Reader node using the new add_reader() API
    logger.info("Building graph nodes...")
    
    # Add DataReader using new add_reader() API
    # This creates a Reader node that actively reads data from files
    builder.add_reader(
        name="DataReader",
        src_dir="/tmp/fluxway_example_data",  # Source data directory
        data_type="json",                       # Data format
        batch_size=100,                         # Batch size for reading
        file_pattern="*.json",                  # File pattern to match
        worker_count=2,                         # Number of reader workers
        resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
        description="JSON data reader node using add_reader() API",
    )
    
    # Add SPMD nodes
    builder.add_node("DenseDense", DenseDenseSPMD, "Dense -> Dense transmission SPMD")
    builder.add_node("DenseDense1", DenseDenseSPMD1, "Dense -> Dense transmission SPMD1")
    
    # Forward edges
    logger.info("Building forward edges...")
    # DataReader -> DenseDense
    builder.add_forward_edge("DataReader", "DenseDense")
    
    # DenseDense -> DenseDense1
    builder.add_forward_edge("DenseDense", "DenseDense1")

    builder.add_func_edge("DenseDense1", OpType.FORWARD, "DenseDense1", OpType.BACKWARD)
    
    # Backward edges (different from forward to demonstrate forward/backward differences)
    logger.info("Building backward edges...")
    # DenseDense1 -> DenseDense
    builder.add_backward_edge("DenseDense1", "DenseDense")
    
    # Note: No backward edge from DenseDense to DataReader (Reader is a source)

    # Build graph
    graph = builder.build(name="example_graph")
    logger.info(f"Graph built successfully with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    return graph


def create_example_data():
    """Create example JSON data files for testing."""
    import json
    import os
    
    data_dir = "/tmp/fluxway_example_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a few example JSON files
    for i in range(4):
        file_path = os.path.join(data_dir, f"data_{i}.json")
        data = [
            {
                "batch_id": i * 100 + j,
                "tensor": [[float(x) for x in range(128)] for _ in range(64)],
                "metadata": f"sample_data_{i}_{j}"
            }
            for j in range(25)  # 25 records per file
        ]
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    logger.info(f"Created example data files in {data_dir}")


async def run_example():
    """Run the example with Ray and Controller."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Starting Fluxway Example")
    logger.info("=" * 80)
    
    # Initialize Ray
    if not ray.is_initialized():
        logger.info("Initializing Ray...")
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.INFO,
            log_to_driver=True
        )
        logger.info("Ray initialized successfully")
    
    try:
        # Create example data files for the Reader
        logger.info("Creating example data files...")
        create_example_data()
        
        # Build graph
        graph = build_example_graph()
        
        # Create controller
        logger.info("Creating ControllerActor...")
        heartbeat_config = HeartbeatConfig(
            heartbeat_interval=5.0,
            heartbeat_timeout=15.0,
            max_missed_heartbeats=3
        )
        async_step_config = AsyncStepConfig(
            max_forward_step_diff=2,
            max_backward_step_diff=2,
            min_lock_duration=1.0
        )
        
        controller = ControllerActor.remote(
            graph=graph,
            heartbeat_config=heartbeat_config,
            async_step_config=async_step_config
        )
        logger.info("ControllerActor created")
        
        # Initialize controller
        logger.info("Initializing controller...")
        init_start = time.time()
        init_success = await controller.initialize.remote()
        init_duration = time.time() - init_start
        
        if not init_success:
            logger.error("Controller initialization failed")
            return
        
        logger.info(f"Controller initialized in {init_duration:.2f}s")
        
        # Wait for all drivers to be ready
        logger.info("Waiting for all SPMD drivers to be ready...")
        wait_start = time.time()
        
        # Poll controller state until RUNNING
        max_wait_time = 60.0  # 60 seconds timeout
        poll_interval = 1.0
        
        while time.time() - wait_start < max_wait_time:
            state = await controller.get_state.remote()
            logger.info(f"Controller state: {state}")
            
            if state == "running":
                logger.info("All drivers are ready, controller is running")
                break
            
            await asyncio.sleep(poll_interval)
        else:
            logger.warning("Timeout waiting for controller to reach RUNNING state")
        
        wait_duration = time.time() - wait_start
        logger.info(f"Waited {wait_duration:.2f}s for drivers to be ready")
        
        # Start execution by sending data to source nodes
        logger.info("Starting execution by sending data to source nodes...")
        exec_start = time.time()
        
        # Get source node handles
        source_spmd_handle = await controller.get_spmd_handle.remote("DataReader")
        if source_spmd_handle is None:
            logger.error("Failed to get source SPMD handle")
            return
        
        # Reader node will automatically read data from files
        # The Reader uses the new add_reader() API with backpressure support
        logger.info("Reader node will read data from files automatically...")
        logger.info("Backpressure handling is enabled between Reader and SPMD nodes...")
        logger.info("Waiting for data processing to complete...")
        
        # Wait a bit for data generation to start
        await asyncio.sleep(2.0)
        
        exec_duration = time.time() - exec_start
        logger.info(f"Execution phase completed in {exec_duration:.2f}s")
        
        # Wait for completion
        logger.info("Waiting for all tasks to complete...")
        completion_start = time.time()
        
        # Poll until all tasks are completed
        max_completion_time = 120.0  # 2 minutes timeout
        while time.time() - completion_start < max_completion_time:
            state = await controller.get_state.remote()
            logger.info(f"Controller state: {state}")
            
            if state == "completed":
                logger.info("All tasks completed successfully")
                break
            
            await asyncio.sleep(poll_interval)
        else:
            logger.warning("Timeout waiting for tasks to complete")
        
        completion_duration = time.time() - completion_start
        logger.info(f"Waited {completion_duration:.2f}s for completion")
        
        # Summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("Example Execution Summary")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {total_duration:.2f}s")
        logger.info(f"Initialization time: {init_duration:.2f}s")
        logger.info(f"Driver ready wait time: {wait_duration:.2f}s")
        logger.info(f"Execution time: {exec_duration:.2f}s")
        logger.info(f"Completion wait time: {completion_duration:.2f}s")
        logger.info("=" * 80)
        logger.info("Example completed successfully!")
        logger.info("Check Ray Dashboard for detailed metrics and performance data")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        # Note: In production, you might want to gracefully shutdown
        # For this example, we'll just log
        logger.info("Cleanup completed")


def main():
    """Main entry point."""
    try:
        asyncio.run(run_example())
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
