"""
Main entry point for the Fluxway example.

This example demonstrates a machine learning pipeline with:
1. Reader: Reads data from JSON files (outputs sparse)
2. Preprocessing: Preprocesses raw data (periodic (streaming) mode, sparse -> sparse)
3. Embedding: Performs embedding operations (sparse -> sparse)
4. Model: Performs model computation (sparse -> dense)

Features:
- All SPMDs use CPU only (no GPU)
- All operations simulate data transmission only
- Single SPMD total_rank <= 6
- Bidirectional edges and forward/backward edge differences
- Complete logging with timestamps
- Both tensor and state data transmission
- Reader node using add_reader() API for data reading
- Backpressure handling between Reader and SPMD nodes
- Eager mode execution for immediate data processing
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

# Use absolute imports to support both module and direct execution
from fluxway.example.spmd_modules import PreprocessingSPMD, EmbeddingSPMD, ModelSPMD  # noqa: E402
from fluxway.example.data_types import TensorOnly  # noqa: E402

# Use the global Fluxway logging configuration (respects FLUXWAY_LOG_LEVEL / FLUXWAY_LOG_FORCE)
logger = logging.getLogger(__name__)


DEFAULT_SRC_DIR = "/data/chiave-workspace/tmp/recsys/src_dir"


def build_example_graph(
    *, batch_size: int = 1, src_dir: str = DEFAULT_SRC_DIR
) -> Graph:
    """
    Build the example graph with:
    - Reader -> Preprocessing -> Embedding -> Model
    - Forward: reader (sparse) -> preprocessing (sparse->sparse) -> embedding (sparse->sparse) -> model (sparse->dense)
    - Backward: model forward -> model backward (self-loop), model -> embedding
    - Preprocessing is periodic (streaming) mode
    """
    builder = Builder()

    # Add Reader node using the new add_reader() API
    logger.info("Building graph nodes...")

    # Add Reader node
    # This creates a Reader node that actively reads data from files
    builder.add_reader(
        name="Reader",
        src_dir=src_dir,  # Source data directory
        data_type="json",  # Data format
        batch_size=batch_size,  # Batch size for reading
        file_pattern="*.json",  # File pattern to match
        worker_count=2,  # Number of reader workers
        resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
        description="Data reader node for reading JSON files",
        output_data_type=TensorOnly,  # Output type: TensorOnly (only tensor field)
    )

    # Add SPMD nodes
    builder.add_node(
        "Preprocessing", PreprocessingSPMD, "Preprocessing SPMD (Eager mode)"
    )
    builder.add_node("Embedding", EmbeddingSPMD, "Embedding SPMD")
    builder.add_node("Model", ModelSPMD, "Model SPMD")

    # Forward edges
    logger.info("Building forward edges...")
    # Reader -> Preprocessing (periodic)
    builder.add_forward_edge("Reader", "Preprocessing")

    # Preprocessing -> Embedding
    builder.add_forward_edge("Preprocessing", "Embedding")

    # Embedding -> Model
    builder.add_forward_edge("Embedding", "Model")

    # Backward edges
    logger.info("Building backward edges...")
    # Model forward -> Model backward (self-loop)
    builder.add_func_edge("Model", "forward", "Model", "backward")

    # Model -> Embedding
    builder.add_backward_edge("Model", "Embedding")

    # Note: No backward edge from Embedding to Preprocessing (periodic node doesn't need backward)
    # Note: No backward edge from Reader (Reader is a source)

    # Build graph
    graph = builder.build(name="example_graph")
    logger.info(
        f"Graph built successfully with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
    )

    return graph


def create_example_data(
    *, num_files: int = 4, records_per_file: int = 25, src_dir: str = DEFAULT_SRC_DIR
) -> None:
    """Create example JSON data files for testing.

    Args:
        num_files: Number of JSON input files to create.
        records_per_file: Number of records in each JSON file.
        src_dir: Directory to write example data files into.
    """
    import json
    import os

    data_dir = src_dir
    os.makedirs(data_dir, exist_ok=True)

    # Create a few example JSON files
    for i in range(num_files):
        file_path = os.path.join(data_dir, f"data_{i}.json")
        data = [
            {
                "tensor": [
                    [float(j + i * records_per_file) for _ in range(8)]
                    for _ in range(4)
                ],
                "metadata": f"sample_data_{i}_{j}",
            }
            for j in range(records_per_file)
        ]
        with open(file_path, "w") as f:
            json.dump(data, f)

    logger.info(
        f"Created example data files in {data_dir} "
        f"(num_files={num_files}, records_per_file={records_per_file})"
    )


async def run_example(
    *,
    num_files: int = 4,
    records_per_file: int = 25,
    batch_size: int = 1,
    src_dir: str = DEFAULT_SRC_DIR,
):
    """Run the example with Ray and Controller."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Starting Fluxway Example")
    logger.info("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        logger.info("Initializing Ray...")
        # Explicitly propagate Fluxway logging env vars into all Ray workers so that
        # Controller/Driver/Worker actors can also honour FLUXWAY_LOG_LEVEL /
        # FLUXWAY_LOG_FORCE and emit DEBUG logs when requested.
        import os

        # NOTE: Ray expects env vars under the "env_vars" key in runtime_env.
        # Using "env" (what we had before) does NOT propagate variables to workers,
        # which is why Controller/Driver/Worker saw env_level=None in logs.
        runtime_env = {
            "env_vars": {
                "FLUXWAY_LOG_LEVEL": os.getenv("FLUXWAY_LOG_LEVEL", "DEBUG"),
                "FLUXWAY_LOG_FORCE": os.getenv("FLUXWAY_LOG_FORCE", "1"),
            }
        }
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.DEBUG,
            log_to_driver=True,
            runtime_env=runtime_env,
        )
        logger.info("Ray initialized successfully")

    try:
        # Create example data files for the Reader
        logger.info("Creating example data files...")
        create_example_data(
            num_files=num_files, records_per_file=records_per_file, src_dir=src_dir
        )

        # Build graph
        graph = build_example_graph(batch_size=batch_size, src_dir=src_dir)

        # Create controller
        logger.info("Creating ControllerActor...")
        heartbeat_config = HeartbeatConfig(
            heartbeat_interval=5.0, heartbeat_timeout=15.0, max_missed_heartbeats=3
        )
        async_step_config = AsyncStepConfig(
            max_forward_step_diff=2, max_backward_step_diff=2, min_lock_duration=1.0
        )

        controller = ControllerActor.remote(
            graph=graph,
            heartbeat_config=heartbeat_config,
            async_step_config=async_step_config,
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
        source_spmd_handle = await controller.get_spmd_handle.remote("Reader")
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
    import argparse

    parser = argparse.ArgumentParser(description="Run Fluxway example.")
    parser.add_argument(
        "--num-files",
        type=int,
        default=4,
        help="Number of JSON input files to generate (default: 4).",
    )
    parser.add_argument(
        "--records-per-file",
        type=int,
        default=25,
        help="Number of records per JSON file (default: 25).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the Reader node (default: 1).",
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        default=DEFAULT_SRC_DIR,
        help=f"Source data directory for Reader (default: {DEFAULT_SRC_DIR}).",
    )
    args = parser.parse_args()
    if args.num_files <= 0:
        raise ValueError("--num-files must be > 0")
    if args.records_per_file <= 0:
        raise ValueError("--records-per-file must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    try:
        asyncio.run(
            run_example(
                num_files=args.num_files,
                records_per_file=args.records_per_file,
                batch_size=args.batch_size,
                src_dir=args.src_dir,
            )
        )
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
