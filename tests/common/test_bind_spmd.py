import copy

import pytest

from fluxway.common.decorators import spmd, operator
from fluxway.common.bind import bind_spmd, _apply_operator_configs


class PlainBusinessClass:
    def __init__(self, embed_dim=3):
        self.embed_dim = embed_dim

    def step(self, **kwargs):
        return {"grad": 1.0}

    def update(self, **kwargs):
        pass


def test_bind_spmd_creates_spmd_from_plain_class():
    BoundCls = bind_spmd(
        PlainBusinessClass,
        resource={"num_cpus": 2, "num_gpus": 1},
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(4,),
        step={
            "input_sharding": {"emb": {"type": "static", "axis": ["dp"]}},
            "output_sharding": {"grad": {"type": "static", "axis": ["dp"]}},
        },
    )

    assert BoundCls.is_fluxway_spmd()
    assert BoundCls.__name__ == "PlainBusinessClass"

    config = BoundCls.get_spmd_config()
    assert "step" in config.operator_metas
    assert "emb" in config.operator_metas["step"].input_sharding
    assert "grad" in config.operator_metas["step"].output_sharding
    assert config.mesh.size == 4

    res = BoundCls.get_resource()
    assert res.num_gpus == 1
    assert res.num_cpus == 2


def test_bind_spmd_multiple_operators():
    BoundCls = bind_spmd(
        PlainBusinessClass,
        resource={"num_cpus": 1},
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
    assert "step" in config.operator_metas
    assert "update" in config.operator_metas
    assert "x" in config.operator_metas["step"].input_sharding
    assert "grad" in config.operator_metas["update"].input_sharding


def test_bind_spmd_original_class_unmodified():
    bind_spmd(
        PlainBusinessClass,
        resource={"num_cpus": 1},
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(2,),
        step={"input_sharding": {"x": {"type": "static", "axis": ["dp"]}}},
    )

    assert not hasattr(PlainBusinessClass, "is_fluxway_spmd")


def test_bind_spmd_overrides_existing_decorated_operator():
    @spmd(
        resource={"num_cpus": 1},
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(2,),
    )
    class DecoratedSPMD:
        @operator(
            name="step",
            input_sharding={"x": {"type": "static", "axis": ["dp"]}},
        )
        def step(self, **kwargs):
            return kwargs

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
    meta = config.operator_metas["step"]
    assert "x" in meta.input_sharding
    assert "y" in meta.input_sharding


def test_bind_spmd_as_source():
    BoundCls = bind_spmd(
        PlainBusinessClass,
        resource={"num_cpus": 1},
        mesh_axis_names=("worker",),
        mesh_axis_shapes=(1,),
        is_source=True,
        interleaved=True,
        interleave_unit_size=1,
        step={"output_sharding": {"data": {"type": "static", "axis": ["worker"]}}},
    )

    assert BoundCls.is_fluxway_spmd()
    config = BoundCls.get_spmd_config()
    meta = config.operator_metas["step"]
    assert meta.is_source is True


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
    @spmd(
        resource={"num_cpus": 1},
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(2,),
    )
    class TestCls:
        @operator(
            name="fwd",
            input_sharding={"a": {"type": "static", "axis": ["dp"]}},
        )
        def fwd(self, **kwargs):
            pass

    config = copy.deepcopy(TestCls.get_spmd_config())
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
        TestCls,
    )

    meta = config.operator_metas["fwd"]
    assert "a" in meta.input_sharding
    assert "b" in meta.input_sharding


def test_apply_operator_configs_register_new():
    @spmd(
        resource={"num_cpus": 1},
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(2,),
    )
    class TestCls:
        @operator(name="existing")
        def existing(self):
            pass

        def new_method(self, **kwargs):
            pass

    config = copy.deepcopy(TestCls.get_spmd_config())
    _apply_operator_configs(
        config,
        {"new_method": {"input_sharding": {"x": {"type": "static", "axis": ["dp"]}}}},
        TestCls,
    )

    assert "new_method" in config.operator_metas
    assert "x" in config.operator_metas["new_method"].input_sharding


def test_apply_operator_configs_raises_on_missing_method():
    @spmd(
        resource={"num_cpus": 1},
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(2,),
    )
    class TestCls:
        @operator(name="existing")
        def existing(self):
            pass

    config = copy.deepcopy(TestCls.get_spmd_config())
    with pytest.raises(ValueError, match="method not found"):
        _apply_operator_configs(
            config,
            {
                "ghost_method": {
                    "input_sharding": {"x": {"type": "static", "axis": ["dp"]}}
                }
            },
            TestCls,
        )
