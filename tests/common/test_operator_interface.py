from dataclasses import dataclass
from fluxway.common.decorators import operator, data_type
from fluxway.common.types.tensor import TransportTensor


@data_type()
@dataclass
class RaggedInput:
    feature1: TransportTensor
    feature2: TransportTensor


@data_type()
@dataclass
class OtherInput:
    other_tensor: TransportTensor


class MockSPMD:
    @operator(
        name="test_uniform", input_sharding={"data": {"type": "static", "axis": ["dp"]}}
    )
    def test_uniform(self, data: RaggedInput):
        pass

    @operator(
        name="test_nested",
        input_sharding={
            "data": {
                "feature1": {"type": "static", "axis": ["dp"]},
                "feature2": {"type": "dynamic", "axis": ["tp"]},
            }
        },
    )
    def test_nested(self, data: RaggedInput):
        pass

    @operator(
        name="test_mixed",
        input_sharding={
            "data": {"type": "static", "axis": ["dp"]},
            "other": {"type": "static", "axis": ["mp"]},
        },
    )
    def test_mixed(self, data: RaggedInput, other: OtherInput):
        pass

    @operator(
        name="test_output",
        output_sharding={"placeholder": {"type": "static", "axis": ["dp"]}},
    )
    def test_output(self) -> RaggedInput:  # type: ignore
        pass


def test_sharding_resolution():
    meta_uniform = getattr(MockSPMD.test_uniform, "_flux_operator_meta", None)
    assert meta_uniform is not None
    assert meta_uniform.input_sharding["feature1"]["axis"] == ["dp"]
    assert meta_uniform.input_sharding["feature2"]["axis"] == ["dp"]
    assert "data" in meta_uniform.input_param_map
    assert "feature1" in meta_uniform.input_param_map["data"]

    meta_nested = getattr(MockSPMD.test_nested, "_flux_operator_meta", None)
    assert meta_nested is not None
    assert meta_nested.input_sharding["feature1"]["axis"] == ["dp"]
    assert meta_nested.input_sharding["feature2"]["axis"] == ["tp"]

    meta_mixed = getattr(MockSPMD.test_mixed, "_flux_operator_meta", None)
    assert meta_mixed is not None
    assert meta_mixed.input_sharding["feature1"]["axis"] == ["dp"]
    assert meta_mixed.input_sharding["other_tensor"]["axis"] == ["mp"]

    # Test Output Sharding & Placeholder
    meta_output = getattr(MockSPMD.test_output, "_flux_operator_meta", None)
    assert meta_output is not None
    # "placeholder" should have been expanded to individual fields
    assert "feature1" in meta_output.output_sharding
    assert "feature2" in meta_output.output_sharding
    assert meta_output.output_sharding["feature1"]["axis"] == ["dp"]
    assert "return" in meta_output.output_param_map
    assert "feature1" in meta_output.output_param_map["return"]


if __name__ == "__main__":
    test_sharding_resolution()
    print("All operator interface tests passed!")
