import unittest

from fluxway.common.types.tensor import TransportTensor
from fluxway.common.types.configs import SPMDConfig, OperatorMeta
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.common.mesh import DeviceMesh


class DummyNode:
    def step(self, arg1: TransportTensor, **kwargs):
        pass


class TestOperatorInjection(unittest.TestCase):
    def test_operator_injection_without_decorator(self):
        """Test that OperatorManager dynamically wraps methods without @operator decorator"""
        # Create a config with OperatorMeta pointing to DummyNode.step
        config = SPMDConfig(
            num_workers=1,
            mesh=DeviceMesh(axis_names=("dp",), device_shape=(1,)),
            operators={"step": "step"},
            operator_metas={
                "step": OperatorMeta(
                    name="step",
                    owner_class_module=DummyNode.__module__,
                    owner_class_qualname=DummyNode.__qualname__,
                    output_uuids=[1],
                    output_sharding={"out1": {"type": "static", "axis": []}},
                )
            },
        )

        # Instantiate OperatorManager
        manager = OperatorManager(config=config, spmd_id="test_spmd", rank=0)

        # OperatorManager should find DummyNode.step and wrap it with _OperatorWrapper
        op_func = manager._operators["step"]

        # Check if the function is wrapped and has _fluxway_operator_meta
        self.assertTrue(hasattr(op_func, "_fluxway_operator_meta"))

        # The meta should match the one we provided in the config
        meta = op_func._fluxway_operator_meta
        self.assertEqual(meta.name, "step")
        self.assertIn("out1", meta.output_sharding)


if __name__ == "__main__":
    unittest.main()
