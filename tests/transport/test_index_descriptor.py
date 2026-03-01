import pytest
import jax.numpy as jnp
from fluxway.common.index_descriptor import (
    IndexDescriptor,
    ContinuousIndex,
    IndexPattern,
    ExplicitIndex,
)
from fluxway.common.backend import set_backend, get_backend


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


def test_continuous_index_merge_contiguous():
    """Verify O(1) merging of adjacent contiguous chunks."""
    c1 = ContinuousIndex.chunk(start=0, count=10)
    c2 = ContinuousIndex.chunk(start=10, count=20)

    merged = IndexDescriptor.concatenate([c1, c2])

    assert isinstance(merged, ContinuousIndex)
    assert merged.pattern == IndexPattern.CONTIGUOUS
    assert merged.start == 0
    assert merged.count == 30

    # Verify values match materialization
    backend = get_backend()
    expected = jnp.arange(30, dtype=jnp.int32)
    assert backend.equal(merged.materialize(), expected)


def test_continuous_index_merge_strided():
    """Verify O(1) merging of adjacent strided patterns."""
    s1 = ContinuousIndex.cyclic(start=0, step=2, count=5)  # [0, 2, 4, 6, 8]
    s2 = ContinuousIndex.cyclic(start=10, step=2, count=5)  # [10, 12, 14, 16, 18]

    merged = IndexDescriptor.concatenate([s1, s2])

    assert isinstance(merged, ContinuousIndex)
    assert merged.pattern == IndexPattern.STRIDED
    assert merged.start == 0
    assert merged.step == 2
    assert merged.count == 10

    # Verify values
    backend = get_backend()
    expected = jnp.arange(0, 20, 2, dtype=jnp.int32)
    assert backend.equal(merged.materialize(), expected)


def test_continuous_index_merge_fail_non_adjacent():
    """Verify fallback to ExplicitIndex when not adjacent."""
    c1 = ContinuousIndex.chunk(start=0, count=10)
    c2 = ContinuousIndex.chunk(start=15, count=10)  # Gap!

    merged = IndexDescriptor.concatenate([c1, c2])

    assert isinstance(merged, ExplicitIndex)
    backend = get_backend()
    expected = jnp.concatenate([jnp.arange(10), jnp.arange(15, 25)])
    assert backend.equal(merged.indices, expected)


def test_continuous_index_merge_fail_mismatched_pattern():
    """Verify fallback to ExplicitIndex when patterns differ."""
    c1 = ContinuousIndex.chunk(start=0, count=10)
    s1 = ContinuousIndex.cyclic(start=10, step=2, count=5)

    merged = IndexDescriptor.concatenate([c1, s1])

    assert isinstance(merged, ExplicitIndex)
    backend = get_backend()
    expected = jnp.concatenate([jnp.arange(10), jnp.arange(10, 20, 2)])
    assert backend.equal(merged.indices, expected)


def test_empty_concatenate():
    merged = IndexDescriptor.concatenate([])
    assert isinstance(merged, ContinuousIndex)
    assert merged.count == 0


def test_single_concatenate():
    c1 = ContinuousIndex.chunk(0, 10)
    merged = IndexDescriptor.concatenate([c1])
    assert merged is c1
