import unittest
from dataclasses import dataclass
from fluxway.common.decorators import data_type, operator, _get_fluxway_streams
from fluxway.common.types.tensor import TransportTensor

# --- Test Data Types ---


@data_type()
@dataclass
class DeepLeaf:
    val: TransportTensor


@data_type()
@dataclass
class MidLevel:
    leaf: DeepLeaf
    direct: TransportTensor


@data_type()
@dataclass
class RootLevel:
    mid: MidLevel
    top_val: TransportTensor


# --- Test Operators ---

spec = {"type": "static", "axis": ["dp"]}


@operator(
    name="test_op",
    input_sharding={
        "inp": {
            "mid": {"leaf": spec, "direct": {"type": "dynamic", "axis": ["shard"]}},
            "top_val": spec,
        }
    },
)
def my_op(inp: RootLevel):
    pass


@operator(name="uniform_op", input_sharding={"inp": spec})
def uniform_op(inp: RootLevel):
    pass


@operator(name="output_op", output_sharding=spec)
def output_op() -> RootLevel:
    return None  # type: ignore


# --- Test Class ---


class TestRecursiveDataType(unittest.TestCase):
    def test_recursive_stream_extraction(self):
        """Verify that _get_fluxway_streams finds nested paths with correct dot notation."""
        data_f, control_f = _get_fluxway_streams(RootLevel)

        expected_data = ["mid.leaf.val", "mid.direct", "top_val"]
        self.assertCountEqual(data_f, expected_data)

    def test_recursive_sharding_resolution(self):
        """Test how @operator resolves nested input_sharding specs using parameter prefixes."""
        # my_op is now a regular function in our test (since it's top-level)
        meta = my_op._fluxway_operator_meta
        sharding = meta.input_sharding

        # Keys should be: param_name.field_path
        self.assertEqual(sharding["mid.leaf.val"], spec)
        self.assertEqual(sharding["mid.direct"]["type"], "dynamic")
        self.assertEqual(sharding["top_val"], spec)

    def test_uniform_sharding_resolution(self):
        """Test applying one spec to an entire nested @data_type parameter branch."""
        sharding = uniform_op._fluxway_operator_meta.input_sharding
        for field in ["mid.leaf.val", "mid.direct", "top_val"]:
            self.assertEqual(sharding[field], spec)

    def test_output_sharding_resolution(self):
        """Test applying uniform spec to @data_type return (no parameter prefix for returns)."""
        sharding = output_op._fluxway_operator_meta.output_sharding
        for field in ["mid.leaf.val", "mid.direct", "top_val"]:
            self.assertEqual(sharding[field], spec)


if __name__ == "__main__":
    unittest.main()
