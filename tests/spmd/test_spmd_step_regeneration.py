import ast
import types
from pathlib import Path


def _load_compute_output_step():
    """
    Load compute_output_step without importing `fluxway` (importing it triggers runtime deps like Ray).

    We parse `worker.py` and exec only the function definition.
    """
    repo_root = Path(__file__).resolve().parents[2]
    worker_path = repo_root / "fluxway" / "spmd" / "worker" / "worker.py"
    source = worker_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(worker_path))

    method_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SPMDWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "compute_output_step":
                    method_node = item
                    break
    assert method_node is not None, "SPMDWorker.compute_output_step not found in worker.py"

    module = types.ModuleType("fluxway_worker_step_extract")
    # Compile the method as a standalone function (first arg is `self`).
    compiled = compile(ast.Module(body=[method_node], type_ignores=[]), str(worker_path), "exec")
    exec(compiled, module.__dict__)
    return module.compute_output_step


compute_output_step = _load_compute_output_step()


class _DummyWorker:
    def __init__(self, *, enable_output_step_remap: bool, total_rank: int, rank: int):
        self.config = types.SimpleNamespace(enable_output_step_remap=enable_output_step_remap)
        self._total_rank = total_rank
        self.rank = rank


def test_compute_output_step_regenerate_off_keeps_input_step() -> None:
    assert (
        compute_output_step(
            _DummyWorker(enable_output_step_remap=False, total_rank=8, rank=3),
            is_eager=False,
            current_step=100,
            input_step=7,
        )
        == 7
    )
    assert (
        compute_output_step(
            _DummyWorker(enable_output_step_remap=False, total_rank=8, rank=3),
            is_eager=True,
            current_step=100,
            input_step=7,
        )
        == 7
    )


def test_compute_output_step_regenerate_on_non_eager_uses_current_step() -> None:
    assert (
        compute_output_step(
            _DummyWorker(enable_output_step_remap=True, total_rank=8, rank=3),
            is_eager=False,
            current_step=12,
            input_step=7,
        )
        == 12
    )


def test_compute_output_step_regenerate_on_eager_uses_global_step_formula() -> None:
    # output_step = total_rank * current_step + local_rank
    assert (
        compute_output_step(
            _DummyWorker(enable_output_step_remap=True, total_rank=8, rank=3),
            is_eager=True,
            current_step=12,
            input_step=7,
        )
        == 8 * 12 + 3
    )

