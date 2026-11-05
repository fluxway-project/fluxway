import pytest

from fluxway.common.types.checkpoint import CheckpointComponentSpec
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponent,
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
)


class _DeclaredOutput(CheckpointComponent):
    def __init__(self, component_id: str, output_path: str | None) -> None:
        self._descriptor = CheckpointComponentSpec(
            component_id, output_path=output_path
        )

    @property
    def descriptor(self) -> CheckpointComponentSpec:
        return self._descriptor


def _registry(*outputs: tuple[str, str | None]) -> CheckpointComponentRegistry:
    return CheckpointComponentRegistry(
        tuple(
            CheckpointComponentRegistration(_DeclaredOutput(component_id, path))
            for component_id, path in outputs
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("left_path", "right_path", "same_spmd"),
    [
        ("sparse/shared", "sparse/shared", False),
        ("sparse/shared", "sparse/shared/child", False),
        ("sparse/shared/child", "sparse/shared", False),
        ("sparse/shared", "sparse/shared", True),
        ("sparse/shared", "sparse/shared/child", True),
    ],
)
async def test_checkpointer_rejects_duplicate_output_owners(
    checkpointer_factory, tmp_path, left_path, right_path, same_spmd
):
    registries = (
        {"UserTable": _registry(("first", left_path), ("second", right_path))}
        if same_spmd
        else {
            "UserTable": _registry(("first", left_path)),
            "ItemTable": _registry(("second", right_path)),
        }
    )
    with pytest.raises(
        ValueError, match="Conflicting checkpoint output paths"
    ) as error:
        await checkpointer_factory(
            object(),
            str(tmp_path / "repository"),
            interval_steps=1,
            component_registries=registries,
        )
    message = str(error.value)
    for expected in (
        "UserTable",
        "first",
        "second",
        left_path,
        right_path,
        "output_path",
        "overlap",
        "do not overlap",
    ):
        assert expected in message
    if not same_spmd:
        assert "ItemTable" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("left_path", "right_path"),
    [
        ("sparse/UserTable", "sparse/ItemTable"),
        ("sparse/table1", "sparse/table10"),
        ("model", "sparse/UserTable"),
        (None, None),
    ],
)
async def test_checkpointer_accepts_disjoint_outputs(
    checkpointer_factory, tmp_path, left_path, right_path
):
    await checkpointer_factory(
        object(),
        str(tmp_path / "repository"),
        interval_steps=1,
        component_registries={
            "UserTable": _registry(("first", left_path)),
            "ItemTable": _registry(("second", right_path)),
        },
    )


@pytest.mark.parametrize(
    "output_path",
    ["", "/model", "sparse//table", "sparse/../table", "sparse\\table", "a\x00b"],
)
def test_output_path_errors_explain_the_required_syntax(output_path):
    with pytest.raises(ValueError) as error:
        CheckpointComponentSpec("embedding", output_path=output_path)
    message = str(error.value)
    assert "embedding" in message
    assert repr(output_path) in message
    assert "relative POSIX" in message
    assert "sparse/UserTable" in message
