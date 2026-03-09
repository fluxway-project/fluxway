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

import pickle
import subprocess
import sys
import os
import tempfile

import pytest

sys.path.append(os.getcwd())

from fluxway.common.types.configs import (
    SPMDConfig,
    OperatorMeta,
    EdgeShardingInfo,
    ArgumentType,
)


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
            # Main process: load module and modify attribute
            import importlib.util

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
            input_arguments={ArgumentType.TENSOR.value: ["input"]},
            output_arguments={ArgumentType.TENSOR.value: ["output"]},
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
