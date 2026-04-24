import ast
from pathlib import Path

REPO_ROOT = Path("/Users/admin/Project/fluxway-antigravity")
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


def test_recommendation_train_configures_ray_logical_cpus():
    module = _load_module(EXAMPLE_DIR / "train.py")
    ray_init_calls = []

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id == "ray" and node.func.attr == "init":
            ray_init_calls.append(node)

    assert ray_init_calls, "expected train.py to call ray.init"
    assert any(
        any(keyword.arg == "num_cpus" for keyword in call.keywords)
        for call in ray_init_calls
    ), "train.py should set ray.init(num_cpus=...) for the local demo"
