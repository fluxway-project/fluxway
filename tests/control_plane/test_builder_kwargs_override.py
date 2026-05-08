from fluxway.common.decorators import spmd, operator
from fluxway.control_plane.graph.builder import Builder


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(2,),
)
class _TestSPMD:
    @operator(name="step")
    def step(self):
        pass


@spmd(
    resource={"num_cpus": 1},
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(2,),
)
class _TestSPMDWithSharding:
    @operator(
        name="step",
        input_sharding={"a": {"type": "static", "axis": ["dp"]}},
    )
    def step(self, **kwargs):
        pass

    def undecorated_method(self, **kwargs):
        pass


def test_builder_resource_override():
    builder = Builder()
    builder.add_node(
        "n1",
        _TestSPMD,
        resource={"num_gpus": 2, "num_cpus": 4, "memory_gb": 8.0},
    )
    node = builder._nodes["n1"]
    assert node.resource.num_gpus == 2
    assert node.resource.num_cpus == 4
    assert node.resource.memory_gb == 8.0


def test_builder_resource_partial_override():
    builder = Builder()
    builder.add_node("n1", _TestSPMD, resource={"num_gpus": 1})
    node = builder._nodes["n1"]
    assert node.resource.num_gpus == 1
    assert node.resource.num_cpus == 1
    assert node.resource.memory_gb == 1


def test_builder_runtime_env_override():
    builder = Builder()
    env = {"env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}}
    builder.add_node("n1", _TestSPMD, runtime_env=env)
    node = builder._nodes["n1"]
    assert node.spmd_config.runtime_env == env


def test_builder_reserved_kwargs_not_in_static_args():
    builder = Builder()
    builder.add_node(
        "n1",
        _TestSPMD,
        resource={"num_gpus": 1},
        runtime_env={"env_vars": {}},
        my_custom_arg="value",
    )
    node = builder._nodes["n1"]
    assert "resource" not in node.spmd_config.static_args
    assert "runtime_env" not in node.spmd_config.static_args
    assert "operator_configs" not in node.spmd_config.static_args
    assert node.spmd_config.static_args["my_custom_arg"] == "value"


def test_builder_operator_configs_override_sharding():
    builder = Builder()
    builder.add_node(
        "n1",
        _TestSPMDWithSharding,
        operator_configs={
            "step": {
                "input_sharding": {
                    "a": {"type": "static", "axis": ["dp"]},
                    "b": {"type": "static", "axis": ["dp"]},
                },
            },
        },
    )
    node = builder._nodes["n1"]
    meta = node.spmd_config.operator_metas["step"]
    assert "a" in meta.input_sharding
    assert "b" in meta.input_sharding


def test_builder_operator_configs_register_new():
    builder = Builder()
    builder.add_node(
        "n1",
        _TestSPMDWithSharding,
        operator_configs={
            "undecorated_method": {
                "input_sharding": {"x": {"type": "static", "axis": ["dp"]}},
            },
        },
    )
    node = builder._nodes["n1"]
    assert "undecorated_method" in node.spmd_config.operator_metas
    assert "x" in node.spmd_config.operator_metas["undecorated_method"].input_sharding


def test_builder_no_resource_override_uses_default():
    builder = Builder()
    builder.add_node("n1", _TestSPMD)
    node = builder._nodes["n1"]
    assert node.resource.num_cpus == 1
    assert node.resource.memory_gb == 1
    assert node.resource.num_gpus == 0
