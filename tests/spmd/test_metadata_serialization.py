"""
Tests for OperatorMeta serialization behavior.

These tests verify that OperatorMeta modifications survive cross-process
serialization (pickle), which is critical for Ray-based worker communication.

Background:
    Python pickles module-level functions by reference (module + qualified_name).
    When unpickled in a different process, Python imports the module fresh,
    getting the original function with original attributes.

    This means modifications to function attributes (like _flux_operator_meta)
    made after class definition are lost when the function is sent to workers.

    The solution is to store OperatorMeta explicitly in SPMDConfig fields
    rather than relying on function attributes.
"""

import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile

import pytest

sys.path.append(os.getcwd())

from fluxway.common.types.configs import (
    SPMDConfig,
    OperatorMeta,
    EdgeShardingInfo,
    ArgumentType,
)
from fluxway.common.decorators import (
    operator,
    spmd,
    _instance_storage,
    _resolve_owner_class,
)
from fluxway.spmd.worker.worker import SPMDWorker


def dummy_my_function():
    pass


class TestPickleBehavior:
    """Tests demonstrating Python pickle behavior with function attributes."""

    def test_same_process_pickle_preserves_function_attributes(self):
        """
        Within the same process, pickle preserves function attributes.

        This works because pickle returns the same object reference.
        """
        my_function = dummy_my_function

        my_function.meta = {"key": "original"}
        my_function.meta["key"] = "modified"
        my_function.meta["new_key"] = "new_value"

        pickled = pickle.dumps(my_function)
        restored = pickle.loads(pickled)

        # Same process: restored IS my_function (same object reference)
        assert restored is my_function
        assert restored.meta == {"key": "modified", "new_key": "new_value"}

    def test_cross_process_pickle_loses_function_attribute_modifications(self):
        """
        Cross-process pickle loses function attribute modifications.

        This is the root cause of the sharding metadata issue:
        - Control plane modifies OperatorMeta on functions
        - Ray sends SPMDConfig (with functions) to workers via pickle
        - Workers import module fresh, getting original OperatorMeta
        """
        # Create a temporary module file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write("""
def test_function():
    pass
test_function.meta = {"key": "default_value"}
""")
            module_path = f.name

        try:
            spec = importlib.util.spec_from_file_location("test_module", module_path)
            test_module = importlib.util.module_from_spec(spec)
            sys.modules["test_module"] = test_module
            spec.loader.exec_module(test_module)

            fn = test_module.test_function
            assert fn.meta == {"key": "default_value"}

            # Modify the metadata (simulating Builder modification)
            fn.meta["key"] = "modified_in_main_process"
            fn.meta["new_key"] = "added_in_main"
            assert fn.meta == {
                "key": "modified_in_main_process",
                "new_key": "added_in_main",
            }

            # Pickle to bytes
            pickled = pickle.dumps(fn)

            # Save pickle for child process
            pickle_path = module_path + ".pkl"
            with open(pickle_path, "wb") as pf:
                pf.write(pickled)

            # Child process script
            child_script = f"""
import pickle
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("test_module", "{module_path}")
test_module = importlib.util.module_from_spec(spec)
sys.modules['test_module'] = test_module
spec.loader.exec_module(test_module)

with open("{pickle_path}", 'rb') as f:
    restored_fn = pickle.loads(f.read())

# Print the meta value for assertion
print(repr(restored_fn.meta))
"""

            # Run child process
            result = subprocess.run(
                [sys.executable, "-c", child_script],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Child process should see ORIGINAL meta, not modified
            assert result.returncode == 0, f"Child process failed: {result.stderr}"
            child_output = result.stdout.strip()

            # The key assertion: modifications are LOST in cross-process pickle
            assert child_output == "{'key': 'default_value'}", (
                f"Expected original meta, but got: {child_output}"
            )

        finally:
            # Cleanup
            if "test_module" in sys.modules:
                del sys.modules["test_module"]
            os.unlink(module_path)
            if os.path.exists(pickle_path):
                os.unlink(pickle_path)


def dummy_op(x):
    return x


@operator(name="free_step", static_argument=["x"])
def _free_step_operator(self, x):
    return x


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class _ResolvableModel:
    @operator(name="forward", static_argument=["x"])
    def forward(self, x):
        return x


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(1,))
class _InstantiableModel:
    @operator(name="run", static_argument=["x"])
    def run(self, x):
        return x


class TestOperatorMetaSerialization:
    """Tests for OperatorMeta serialization in SPMDConfig using actual types."""

    def test_real_pydantic_model_preserves_modifications_in_pickle(self):
        """
        OperatorMeta (Pydantic model) preserves modifications when pickled.
        """
        # Create and modify meta
        meta = OperatorMeta(name="test_op", input_arguments=None, output_arguments=None)
        edge_info = EdgeShardingInfo(tuuid_sharding={123: "some_spec"})
        meta.input_pre_sharding["Preprocessing"] = [edge_info]

        # Pickle and unpickle
        pickled = pickle.dumps(meta)
        restored = pickle.loads(pickled)

        # Modifications ARE preserved for Pydantic models
        assert list(restored.input_pre_sharding.keys()) == ["Preprocessing"]
        assert restored.input_pre_sharding["Preprocessing"][0].tuuid_sharding == {
            123: "some_spec"
        }

    def test_real_spmd_config_preserves_modifications(self):
        """
        SPMDConfig with explicit meta fields preserves OperatorMeta modifications.
        """
        my_op = dummy_op

        meta = OperatorMeta(
            name="test_op",
            input_arguments={ArgumentType.DATA.value: ["input"]},
            output_arguments={ArgumentType.DATA.value: ["output"]},
        )
        my_op._flux_operator_meta = meta

        # Create config with explicit meta
        config = SPMDConfig(
            operators={"test_op": my_op}, operator_metas={"test_op": meta}
        )

        # Modify meta after config creation (simulating Builder linking)
        edge_info = EdgeShardingInfo(tuuid_sharding={123: "sharded_spec"})
        meta.input_pre_sharding["SourceSPMD"] = [edge_info]

        # Pickle and unpickle (simulating Ray serialization)
        pickled = pickle.dumps(config)
        restored = pickle.loads(pickled)

        # op_metas preserves modifications
        assert "test_op" in restored.operator_metas
        assert list(restored.operator_metas["test_op"].input_pre_sharding.keys()) == [
            "SourceSPMD"
        ]
        assert restored.operator_metas["test_op"].input_pre_sharding["SourceSPMD"][
            0
        ].tuuid_sharding == {123: "sharded_spec"}


class TestOwnerClassResolution:
    """Tests verifying that owner class info survives pickle and resolves correctly."""

    def test_free_function_operator_meta_has_empty_owner_fields(self):
        meta = _free_step_operator._fluxway_operator_meta
        assert meta.owner_class_qualname == ""
        assert meta.owner_class_module == ""

    def test_class_method_operator_meta_records_owner_info(self):
        @spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
        class MyModel:
            @operator(name="train_step", static_argument=["x"])
            def train_step(self, x):
                return x

        config = MyModel.get_spmd_config()
        meta = config.operator_metas["train_step"]
        assert "MyModel" in meta.owner_class_qualname
        assert meta.owner_class_module == __name__

    def test_resolve_owner_class_returns_same_object(self):
        config = _ResolvableModel.get_spmd_config()
        meta = config.operator_metas["forward"]
        resolved = _resolve_owner_class(
            meta.owner_class_module, meta.owner_class_qualname
        )
        assert resolved is _ResolvableModel

    def test_resolve_owner_class_is_stable_across_calls(self):
        config = _ResolvableModel.get_spmd_config()
        meta = config.operator_metas["forward"]
        first = _resolve_owner_class(meta.owner_class_module, meta.owner_class_qualname)
        second = _resolve_owner_class(
            meta.owner_class_module, meta.owner_class_qualname
        )
        assert first is second

    def test_owner_info_survives_pickle(self):
        config = _ResolvableModel.get_spmd_config()
        restored_config = pickle.loads(pickle.dumps(config))

        meta = restored_config.operator_metas["forward"]
        assert meta.owner_class_qualname != ""
        assert meta.owner_class_module != ""

    def test_pre_instantiate_operators_uses_owner_meta(self):
        config = _InstantiableModel.get_spmd_config()

        _instance_storage.clear()
        SPMDWorker._pre_instantiate_operators(config, rank=0)

        cls = _resolve_owner_class(
            config.operator_metas["run"].owner_class_module,
            config.operator_metas["run"].owner_class_qualname,
        )
        assert id(cls) in _instance_storage

    def test_pre_instantiate_operators_logs_init_boundaries(self, caplog):
        config = _InstantiableModel.get_spmd_config()

        _instance_storage.clear()
        caplog.set_level("DEBUG", logger="fluxway.spmd.worker.worker")

        SPMDWorker._pre_instantiate_operators(config, rank=0)

        begin_records = [
            record
            for record in caplog.records
            if "operator_init_begin" in record.getMessage()
        ]
        complete_records = [
            record
            for record in caplog.records
            if "operator_init_complete" in record.getMessage()
        ]
        assert begin_records
        assert complete_records
        assert all(record.levelname == "DEBUG" for record in begin_records)
        assert all(record.levelname == "DEBUG" for record in complete_records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
