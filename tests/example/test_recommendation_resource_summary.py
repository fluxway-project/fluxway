import ast
from pathlib import Path

TRAIN_FILE = (
    Path(__file__).resolve().parents[2]
    / "fluxway"
    / "example"
    / "recommendation"
    / "train.py"
)


def _load_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _is_controller_resource_summary_remote_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "remote":
        return False
    owner = func.value
    if not isinstance(owner, ast.Attribute):
        return False
    if owner.attr != "get_resource_allocation_summary":
        return False
    return isinstance(owner.value, ast.Name) and owner.value.id == "controller"


def test_recommendation_train_prints_controller_resource_summary():
    module = _load_module(TRAIN_FILE)

    assert any(
        _is_controller_resource_summary_remote_call(node) for node in ast.walk(module)
    ), "train.py should query controller.get_resource_allocation_summary.remote()"
