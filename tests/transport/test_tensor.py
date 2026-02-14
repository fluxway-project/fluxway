import pytest
import pickle
import jax.numpy as jnp
from fluxway.transport.tensor import TransportTensor
from fluxway.transport.backend import set_backend, get_backend


# Force JAX backend
@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.fixture(autouse=False)
def use_torch_backend():
    set_backend("torch")
    yield


def test_jax_zero_copy_construction():
    """
    Verify that constructing TransportTensor from a JAX array
    does not copy the data.
    """
    original_flat = jnp.arange(1000, dtype=jnp.float32)
    splits = [jnp.array([0, 500, 1000], dtype=jnp.int32)]

    tensor = TransportTensor(original_flat, splits)

    # Verify Identity
    assert tensor.flat_values is original_flat
    assert jnp.array_equal(tensor.flat_values, original_flat)


def test_flat_ragged_conversion():
    data = [[1.0, 2.0], [3.0, 4.0, 5.0]]
    tensor = TransportTensor.from_lists(data)

    backend = get_backend()
    # Use backend.tensor to allow conversion from lists if needed, but jap.array works too
    expected_flat = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert backend.equal(tensor.flat_values, expected_flat)
    assert len(tensor.nested_splits) == 1

    expected_splits = jnp.array([0, 2, 5], dtype=jnp.int32)
    assert backend.equal(tensor.nested_splits[0], expected_splits)

    assert tensor.to_lists() == data


def test_deep_nesting():
    data = [[[1.0], [2.0]], [[3.0]]]
    tensor = TransportTensor.from_lists(data)

    backend = get_backend()
    expected_flat = jnp.array([1.0, 2.0, 3.0])
    assert backend.equal(tensor.flat_values, expected_flat)

    assert len(tensor.nested_splits) == 2
    assert backend.equal(tensor.nested_splits[0], jnp.array([0, 2, 3], dtype=jnp.int32))
    assert backend.equal(
        tensor.nested_splits[1], jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    )

    assert tensor.to_lists() == data


def test_slicing_simple():
    data = [[1, 2], [3], [4, 5, 6]]
    tensor = TransportTensor.from_lists(data)
    backend = get_backend()

    sliced = tensor.slice(0, 2)

    assert sliced.to_lists() == [[1.0, 2.0], [3.0]]
    assert backend.equal(sliced.nested_splits[0], jnp.array([0, 2, 3], dtype=jnp.int32))


def test_slicing_empty():
    data = [[1, 2], [3, 4]]
    tensor = TransportTensor.from_lists(data)
    backend = get_backend()

    sliced = tensor.slice(0, 0)
    assert sliced.to_lists() == []

    # Check size using backend-agnostic way (or JAX property)
    # JAX arrays have .size property, Torch has .numel() or shape
    # backend.shape returns tuple
    shape = backend.shape(sliced.flat_values)
    assert shape[0] == 0


def test_serialization():
    data = [[1, 2], [3, 4]]
    tensor = TransportTensor.from_lists(data)
    backend = get_backend()

    serialized = pickle.dumps(tensor)
    deserialized = pickle.loads(serialized)

    assert deserialized.to_lists() == data
    assert backend.equal(deserialized.flat_values, tensor.flat_values)


def test_gather_reordering():
    # [[1, 2], [3], [4, 5, 6]]
    data = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]
    tensor = TransportTensor.from_lists(data)
    backend = get_backend()

    indices = backend.tensor([2, 0, 1], dtype=backend.int32())
    gathered = tensor.gather(indices)

    expected_list = [[4.0, 5.0, 6.0], [1.0, 2.0], [3.0]]
    assert gathered.to_lists() == expected_list

    expected_splits = backend.tensor([0, 3, 5, 6], dtype=backend.int32())
    assert backend.equal(gathered.nested_splits[0], expected_splits)


def test_dense_optimization():
    """
    Verify that dense input is optimized to use multidimensional flat_values
    and empty splits.
    """
    data = [[1.0, 2.0], [3.0, 4.0]]
    tensor = TransportTensor.from_lists(data)
    backend = get_backend()

    # 1. Check that splits are empty (meaning dense representation used)
    # The optimization we added converts uniform nested lists into a single dense tensor
    assert len(tensor.nested_splits) == 0

    # 2. Check flat_values shape is (2, 2)
    # Standard backend.shape returns a tuple/list
    shape = backend.shape(tensor.flat_values)
    assert tuple(shape) == (2, 2)

    # 3. Check operations still work
    # to_lists
    assert tensor.to_lists() == data

    # Slice
    sliced = tensor.slice(0, 1)
    # sliced flat_values should be (1, 2)
    sliced_shape = backend.shape(sliced.flat_values)
    assert tuple(sliced_shape) == (1, 2)
    assert sliced.to_lists() == [[1.0, 2.0]]

    # Gather
    indices = backend.tensor([1, 0], dtype=backend.int32())
    gathered = tensor.gather(indices)
    gathered_shape = backend.shape(gathered.flat_values)
    assert tuple(gathered_shape) == (2, 2)
    assert gathered.to_lists() == [[3.0, 4.0], [1.0, 2.0]]


def test_dense_optimization_3d():
    """
    Verify 3D dense optimization.
    """
    data = [[[1], [2]], [[3], [4]]]  # 2x2x1
    tensor = TransportTensor.from_lists(data)
    backend = get_backend()

    assert len(tensor.nested_splits) == 0

    shape = backend.shape(tensor.flat_values)
    assert tuple(shape) == (2, 2, 1)

    assert tensor.to_lists() == data


def test_ragged_bypass():
    """
    Ragged input should NOT be optimized and should retain splits.
    """
    data = [[1, 2], [3]]
    tensor = TransportTensor.from_lists(data)

    assert len(tensor.nested_splits) > 0  # Should have splits


def test_dense_init_optimization():
    """
    Verify initializing with a dense array directly (no splits provided) works efficiently.
    This simulates user pattern of passing a JAX array directly.
    """
    backend = get_backend()
    # Create (3, 2) dense array using backend
    dense_data = backend.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Init with just values, rely on default empty splits
    tensor = TransportTensor(dense_data)

    # 1. Check splits are empty
    assert len(tensor.nested_splits) == 0

    # 2. Check shape is preserved
    shape = backend.shape(tensor.flat_values)
    assert tuple(shape) == (3, 2)

    # 3. Check operations (slice/gather) work on this "dense wrapper"

    # Slice first 2
    sliced = tensor.slice(0, 2)
    assert backend.shape(sliced.flat_values) == (2, 2)
    assert len(sliced.nested_splits) == 0
    assert backend.equal(sliced.flat_values, dense_data[:2])

    # Gather [2, 0]
    indices = backend.tensor([2, 0], dtype=backend.int32())
    gathered = tensor.gather(indices)
    assert backend.shape(gathered.flat_values) == (2, 2)
    assert len(gathered.nested_splits) == 0

    expected_gathered = backend.tensor([[5.0, 6.0], [1.0, 2.0]])
    assert backend.equal(gathered.flat_values, expected_gathered)


def test_dense_init_bypass_splits():
    """
    Verify that even if splits are None, we treat it as dense.
    """
    backend = get_backend()
    dense_data = backend.tensor([1, 2, 3])
    tensor = TransportTensor(dense_data, nested_splits=[])
    assert len(tensor.nested_splits) == 0


def test_ragged_slice_offset():
    """Test slicing of a ragged tensor with non-zero start offset."""
    # Rows: [ [1, 2], [3], [4, 5, 6], [] ]
    # Splits: [0, 2, 3, 6, 6]
    splits = [jnp.array([0, 2, 3, 6, 6], dtype=jnp.int32)]
    values = jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.int32)

    tensor = TransportTensor(values, splits)

    # Slice [1:3] -> Rows [3], [4, 5, 6]
    # Corresponding splits in original: indices 1, 2.
    # Row 1: 3
    # Row 2: 4, 5, 6
    # New Splits should be: [0, 1, 4] (derived from local lengths 1, 3)

    sliced = tensor.slice(1, 3)

    assert sliced.is_ragged

    lists = sliced.to_lists()
    assert lists == [[3], [4, 5, 6]]

    # Check internals if possible (backend specific)
    backend = get_backend()
    assert backend.equal(sliced.nested_splits[0], jnp.array([0, 1, 4], dtype=jnp.int32))
    assert backend.equal(sliced.flat_values, jnp.array([3, 4, 5, 6], dtype=jnp.int32))


def test_ragged_slice_empty_result():
    """Test slicing resulting in empty tensor from ragged input."""
    splits = [jnp.array([0, 2, 3], dtype=jnp.int32)]
    values = jnp.array([1, 2, 3], dtype=jnp.int32)
    tensor = TransportTensor(values, splits)

    sliced = tensor.slice(0, 0)
    assert len(sliced.to_lists()) == 0
    assert sliced.flat_values.shape[0] == 0


def test_ragged_slice_out_of_bounds():
    """Test slicing with out of bounds indices."""
    splits = [jnp.array([0, 1], dtype=jnp.int32)]
    values = jnp.array([1], dtype=jnp.int32)
    tensor = TransportTensor(values, splits)

    sliced = tensor.slice(0, 10)  # Should behave like slice(0, 1)
    lists = sliced.to_lists()
    assert lists == [[1]]


def test_ragged_slice_multilevel():
    """Test slicing nested ragged tensor (3D equivalent)."""
    # Level 0 splits: defines rows.
    # Level 1 splits: defines sub-rows.
    # Example:
    # Row 0: [ [1, 2], [3] ]  (2 sub-rows)
    # Row 1: [ [4] ]          (1 sub-row)

    l0_splits = jnp.array([0, 2, 3], dtype=jnp.int32)
    l1_splits = jnp.array([0, 2, 3, 4], dtype=jnp.int32)
    values = jnp.array([1, 2, 3, 4], dtype=jnp.int32)

    tensor = TransportTensor(values, [l0_splits, l1_splits])

    # Slice [0:1] -> Should keep Row 0 only.
    sliced = tensor.slice(0, 1)

    lists = sliced.to_lists()
    assert lists == [[[1, 2], [3]]]

    # Slice [1:2] -> Keep Row 1
    sliced_2 = tensor.slice(1, 2)
    assert sliced_2.to_lists() == [[[4]]]


def test_ragged_dynamic_slice_interface():
    """Test usage of dynamic_slice on ragged tensor."""
    splits = [jnp.array([0, 2, 3, 6, 6], dtype=jnp.int32)]
    values = jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.int32)
    tensor = TransportTensor(values, splits)

    # Dynamic slice [1:3] (start=1, size=2)
    backend = get_backend()
    start = backend.tensor(1, dtype=backend.int32())
    size = backend.tensor(2, dtype=backend.int32())

    sliced = tensor.dynamic_slice(start, size)
    assert sliced.to_lists() == [[3], [4, 5, 6]]
