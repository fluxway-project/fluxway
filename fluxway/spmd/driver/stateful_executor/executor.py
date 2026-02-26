"""
Stateful Executor for SPMD worker state management.

Based on ARCHITECTURE.md Section 3.1.2

Responsibilities:
- Execute save, load, sync Worker state functions
- Execute SPMD Worker stateless functions (e.g., computation)
- Collect Worker business metrics (loss, throughput, auc, etc.)
- Report state to Main Orchestrator's State Manager
"""

from typing import Any, Dict, List

from ....common.types import SPMDState


class StatefulExecutor:
    """
    Stateful Executor for SPMD.

    Manages execution of stateful and stateless worker functions.

    State information:
    - SPMD internal training Metric Dict (step, loss, auc, etc.)
    - Profiler-level resource usage
    - Large-scale state like model parameter checkpoints
    """

    def __init__(self, spmd_id: str):
        self.spmd_id = spmd_id
        self._metrics: Dict[str, Any] = {}

    def execute_on_all_workers(self, method: str, *args, **kwargs) -> List[Any]:
        """
        Execute a method on all workers.

        Args:
            method: Method name to call
            *args, **kwargs: Arguments to pass

        Returns:
            List of results from all workers

        TODO: Implement parallel execution
        TODO: Handle failures
        """
        pass

    def save_worker_states(self) -> Dict[str, Any]:
        """
        Save all worker states.

        Returns:
            Mapping of worker_id to saved state

        TODO: Call save_states on all workers
        TODO: Aggregate saved states
        """
        pass

    def load_worker_states(self, states: Dict[str, Any]) -> None:
        """
        Load states to all workers.

        Args:
            states: Mapping of worker_id to state to load

        TODO: Distribute states to workers
        TODO: Call load_states on all workers
        """
        pass

    def sync_worker_states(self) -> None:
        """
        Synchronize states across workers.

        TODO: Implement state synchronization
        TODO: Handle AllReduce for shared state
        """
        pass

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics from all workers.

        Returns:
            Aggregated metrics

        TODO: Query each worker for metrics
        TODO: Aggregate (mean, sum, etc.)
        """
        pass

    def report_to_state_manager(self, state: SPMDState) -> None:
        """
        Report SPMD state to Main Orchestrator.

        TODO: Serialize state
        TODO: Send to StateManager
        """
        pass

    def execute_op(self, op_name: str, step: int) -> Any:
        """
        Trigger execution of an operator on all workers.

        This is the generic execution trigger. Instead of hardcoded
        named methods, users call execute_op("rollout", step),
        execute_op("compute_reward", step), etc.

        The actual execution is driven by the Worker's mailbox:
        workers begin processing when all expected shards for that (op_name, step)
        have arrived via receive_shards / receive_states.

        Args:
            op_name: Name of the operator to execute (e.g., "train_step", "rollout")
            step: Execution step number

        Returns:
            List of results from workers (currently placeholder)

        TODO: Coordinate operator execution across workers
        TODO: Support result aggregation
        """
        return self.execute_on_all_workers(op_name, step)
