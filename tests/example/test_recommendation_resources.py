import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "fluxway" / "example" / "recommendation"


def _load_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _decorator_has_resource_kw(decorator: ast.AST, decorator_names: set[str]) -> bool:
    if not isinstance(decorator, ast.Call):
        return False
    if not isinstance(decorator.func, ast.Name):
        return False
    if decorator.func.id not in decorator_names:
        return False
    return any(keyword.arg == "resource" for keyword in decorator.keywords)


def test_recommendation_spmd_classes_declare_resources():
    files = [
        EXAMPLE_DIR / "input_spmd.py",
        EXAMPLE_DIR / "sparse_spmd.py",
        EXAMPLE_DIR / "dense_spmd.py",
        EXAMPLE_DIR / "reader.py",
    ]

    for path in files:
        module = _load_module(path)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        assert classes, f"expected at least one class in {path}"

        for cls in classes:
            assert any(
                _decorator_has_resource_kw(decorator, {"spmd", "spmd_source"})
                for decorator in cls.decorator_list
            ), f"{path.name}:{cls.name} should declare explicit resource requirements"


def test_recommendation_train_uses_fluxway_launch():
    module = _load_module(EXAMPLE_DIR / "train.py")
    launch_calls = []

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Accept both `launch(...)` and `fluxway.launch(...)` call styles
        is_bare_launch = isinstance(func, ast.Name) and func.id == "launch"
        is_attr_launch = (
            isinstance(func, ast.Attribute)
            and func.attr == "launch"
            and isinstance(func.value, ast.Name)
            and func.value.id == "fluxway"
        )
        if is_bare_launch or is_attr_launch:
            launch_calls.append(node)

    assert launch_calls, (
        "expected train.py to call fluxway.launch() for zero-config Ray initialization"
    )
