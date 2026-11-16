import copy

import pytest

from fluxway.common.decorators import spmd, operator
from fluxway.common.bind import bind_spmd, _apply_operator_configs
from fluxway import ResourceProfile


class PlainBusinessClass:
    def __init__(self, embed_dim=3):
        self.embed_dim = embed_dim

    def step(self, **kwargs):
        return {"grad": 1.0}

    def update(self, **kwargs):
        pass


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(2,),
    max_rank_step_skew=3,
)
class DecoratedSPMD:
    @operator(
        name="step",
        input_sharding={"x": {"type": "static", "axis": ["dp"]}},
    )
    def step(self, **kwargs):
        return kwargs


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(2,),
)
class ConfigurableSPMD:
    @operator(
        name="fwd",
        input_sharding={"a": {"type": "static", "axis": ["dp"]}},
    )
    def fwd(self, **kwargs):
        return kwargs

    @operator(name="existing")
    def existing(self):
        return None

    def new_method(self, **kwargs):
        return kwargs


def test_bind_spmd_creates_spmd_from_plain_class():
    BoundCls = bind_spmd(
        PlainBusinessClass,
        resource_profile=ResourceProfile(base={"CPU": 2, "GPU": 1}),
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(4,),
        max_rank_step_skew=2,
        step={
            "input_sharding": {"emb": {"type": "static", "axis": ["dp"]}},
            "output_sharding": {"grad": {"type": "static", "axis": ["dp"]}},
        },
    )

    assert BoundCls.is_fluxway_spmd()
    assert BoundCls.__name__ == "PlainBusinessClass"

    config = BoundCls.get_spmd_config()
    assert "step" in config.operator_specs
    assert "emb" in config.operator_specs["step"].input_sharding
    assert "grad" in config.operator_specs["step"].output_sharding
    assert config.mesh.size == 4
    assert config.max_rank_step_skew == 2

    profile = BoundCls.get_resource_profile()
    assert profile.base["GPU"] == 1
    assert profile.base["CPU"] == 2


def test_bind_spmd_preserves_business_class_identity() -> None:
    left = type(
        "SharedName",
        (),
        {"__module__": "business.left", "step": lambda self: None},
    )
    right = type(
        "SharedName",
        (),
        {"__module__": "business.right", "step": lambda self: None},
    )

    bound_left = bind_spmd(
        left,
        resource_profile=ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("worker",),
        mesh_axis_shapes=(1,),
    )
    bound_right = bind_spmd(
        right,
        resource_profile=ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("worker",),
        mesh_axis_shapes=(1,),
    )

    assert (bound_left.__module__, bound_left.__qualname__) == (
        "business.left",
        "SharedName",
    )
    assert (bound_right.__module__, bound_right.__qualname__) == (
        "business.right",
        "SharedName",
    )


def test_spmd_decorator_sets_per_spmd_rank_step_skew():
    assert DecoratedSPMD.get_spmd_config().max_rank_step_skew == 3


def test_bind_spmd_multiple_operators():
    BoundCls = bind_spmd(
        PlainBusinessClass,
        resource_profile=ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(2,),
        step={
            "input_sharding": {"x": {"type": "static", "axis": ["dp"]}},
        },
        update={
            "input_sharding": {"grad": {"type": "static", "axis": ["dp"]}},
        },
    )

    config = BoundCls.get_spmd_config()
    assert "step" in config.operator_specs
    assert "update" in config.operator_specs
    assert "x" in config.operator_specs["step"].input_sharding
    assert "grad" in config.operator_specs["update"].input_sharding


def test_bind_spmd_original_class_unmodified():
    bind_spmd(
        PlainBusinessClass,
        resource_profile=ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(2,),
        step={"input_sharding": {"x": {"type": "static", "axis": ["dp"]}}},
    )

    assert not hasattr(PlainBusinessClass, "is_fluxway_spmd")


def test_bind_spmd_overrides_existing_decorated_operator():
    Overridden = bind_spmd(
        DecoratedSPMD,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(8,),
        step={
            "input_sharding": {
                "x": {"type": "static", "axis": ["dp"]},
                "y": {"type": "static", "axis": ["dp"]},
            },
        },
    )

    config = Overridden.get_spmd_config()
    assert config.mesh.size == 8
    meta = config.operator_specs["step"]
    assert "x" in meta.input_sharding
    assert "y" in meta.input_sharding


def test_bind_spmd_as_source():
    BoundCls = bind_spmd(
        PlainBusinessClass,
        resource_profile=ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("worker",),
        mesh_axis_shapes=(1,),
        max_rank_step_skew=1,
        is_source=True,
        interleaved=True,
        interleave_unit_size=1,
        step={"output_sharding": {"data": {"type": "static", "axis": ["worker"]}}},
    )

    assert BoundCls.is_fluxway_spmd()
    config = BoundCls.get_spmd_config()
    meta = config.operator_specs["step"]
    assert meta.is_source is True
    assert config.max_rank_step_skew == 1


def test_spmd_config_rejects_negative_rank_step_skew():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        bind_spmd(
            PlainBusinessClass,
            mesh_axis_names=("worker",),
            mesh_axis_shapes=(1,),
            max_rank_step_skew=-1,
            step={},
        ).get_spmd_config()


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_spmd_config_rejects_non_integer_rank_step_skew(value):
    with pytest.raises(ValueError, match="valid integer"):
        bind_spmd(
            PlainBusinessClass,
            mesh_axis_names=("worker",),
            mesh_axis_shapes=(1,),
            max_rank_step_skew=value,
            step={},
        ).get_spmd_config()


def test_bind_spmd_source_descriptor_is_validated_with_source_trait():
    class Descriptor:
        requires_source_control_runtime = False

        def build_component(self, context):
            raise AssertionError("descriptor construction is a Driver responsibility")

    BoundCls = bind_spmd(
        PlainBusinessClass,
        resource_profile=ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("worker",),
        mesh_axis_shapes=(2,),
        is_source=True,
        interleaved=True,
        interleave_unit_size=1,
        step={"source": Descriptor()},
    )

    spec = BoundCls.get_spmd_config().operator_specs["step"]
    assert spec.is_source is True
    assert spec.is_interleaved is True
    assert spec.interleave_unit_size == 1
    assert isinstance(spec.source, Descriptor)


def test_bind_spmd_rejects_source_trait_conflict():
    with pytest.raises(ValueError, match="cannot set is_source=False"):
        bind_spmd(
            PlainBusinessClass,
            resource_profile=ResourceProfile(base={"CPU": 1}),
            mesh_axis_names=("worker",),
            mesh_axis_shapes=(1,),
            is_source=True,
            step={"is_source": False},
        )


def test_bind_spmd_raises_on_missing_method():
    with pytest.raises(ValueError, match="method not found"):
        bind_spmd(
            PlainBusinessClass,
            mesh_axis_names=("dp",),
            mesh_axis_shapes=(2,),
            nonexistent_method={
                "input_sharding": {"x": {"type": "static", "axis": ["dp"]}}
            },
        )


def test_apply_operator_configs_override_existing():
    config = copy.deepcopy(ConfigurableSPMD.get_spmd_config())  # type: ignore[attr-defined]
    _apply_operator_configs(
        config,
        {
            "fwd": {
                "input_sharding": {
                    "a": {"type": "static", "axis": ["dp"]},
                    "b": {"type": "static", "axis": ["dp"]},
                }
            }
        },
        ConfigurableSPMD,
    )

    meta = config.operator_specs["fwd"]
    assert "a" in meta.input_sharding
    assert "b" in meta.input_sharding


def test_apply_operator_configs_register_new():
    config = copy.deepcopy(ConfigurableSPMD.get_spmd_config())  # type: ignore[attr-defined]
    _apply_operator_configs(
        config,
        {"new_method": {"input_sharding": {"x": {"type": "static", "axis": ["dp"]}}}},
        ConfigurableSPMD,
    )

    assert "new_method" in config.operator_specs
    assert "x" in config.operator_specs["new_method"].input_sharding


def test_apply_operator_configs_raises_on_missing_method():
    config = copy.deepcopy(ConfigurableSPMD.get_spmd_config())  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="method not found"):
        _apply_operator_configs(
            config,
            {
                "ghost_method": {
                    "input_sharding": {"x": {"type": "static", "axis": ["dp"]}}
                }
            },
            ConfigurableSPMD,
        )
