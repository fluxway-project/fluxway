import unittest

from fluxway import OperatorSpec
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.common.operator_spec import OperatorExecutionTrait
from fluxway.common.types.tensor import TransportTensor
from fluxway.spmd.worker.operator_manager import OperatorManager
from tests.compiled_graph_fixtures import single_spmd_config


class DummyNode:
    def step(self, arg1: TransportTensor, **kwargs):
        pass

    async def async_step(self, arg1: TransportTensor, **kwargs):
        pass


def _operator_config(name: str):
    operator_spec = OperatorSpec(
        name=name,
        owner=DummyNode,
        output_sharding={"out1": {"type": "static", "axis": []}},
        execution_trait=(
            OperatorExecutionTrait.ASYNC_COROUTINE
            if name == "async_step"
            else OperatorExecutionTrait.SYNC_COMPUTE
        ),
    )
    return single_spmd_config(
        name="operator-injection",
        spmd_id="test_spmd",
        mesh=DeviceMesh(axis_names=("dp",), device_shape=(1,)),
        operators={name: operator_spec},
    )


class TestOperatorInjection(unittest.TestCase):
    def test_operator_injection_without_decorator(self):
        """Test that OperatorManager dynamically wraps methods without @operator decorator"""
        # Create a config with OperatorSpec pointing to DummyNode.step
        config = _operator_config("step")

        # Instantiate OperatorManager
        manager = OperatorManager(
            config=config,
            spmd_id="test_spmd",
            rank=0,
            static_args={},
            instance_container=OperatorInstanceContainer(),
        )

        # OperatorManager should find DummyNode.step and wrap it with _OperatorWrapper
        op_func = manager._operators["step"]

        # Check if the function is wrapped and has _fluxway_operator_spec
        self.assertTrue(hasattr(op_func, "_fluxway_operator_spec"))
        self.assertFalse(manager.operator_is_coroutine_function("step"))

        # The meta should match the one we provided in the config
        meta = op_func._fluxway_operator_spec
        self.assertEqual(meta.name, "step")
        self.assertIn("out1", meta.output_sharding)

    def test_operator_manager_classifies_coroutine_function_once(self):
        config = _operator_config("async_step")

        manager = OperatorManager(
            config=config,
            spmd_id="test_spmd",
            rank=0,
            static_args={},
            instance_container=OperatorInstanceContainer(),
        )

        self.assertTrue(manager.operator_is_coroutine_function("async_step"))


if __name__ == "__main__":
    unittest.main()
