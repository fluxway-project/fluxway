"""
Tests for receive_shards' single-pass TUUID dispatch logic.

Validates that the merged loop correctly reconstructs per-operator bundles
from a flat RPC payload list, handling all meta_state bit combinations:
  bit 0 (1): ragged tensor — payload contains an extra lengths array
  bit 1 (2): materialized origin indices — payload contains an extra index array
  bit 0+1 (3): both, three payload slots per TUUID

Each test compares the new single-loop implementation against the original
two-loop reference to guarantee behavioral equivalence.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reference implementation (original two-loop logic from worker.py)
# ---------------------------------------------------------------------------


def _old_dispatch(
    tuuids,
    payloads,
    counts,
    global_shapes,
    meta_tensor_state_list,
    slice_compact_desc_params,
    get_operators_for_tuuid,
):
    num = len(tuuids)
    meta_tensor_state_list = meta_tensor_state_list or [0] * num
    global_shapes = global_shapes or [None] * num

    op_to_indices = {}
    for i, u in enumerate(tuuids):
        for op_name in get_operators_for_tuuid(u):
            op_to_indices.setdefault(op_name, []).append(i)

    per_tuuid_payloads = []
    per_tuuid_counts = []
    cursor = 0
    for i in range(num):
        meta_state = meta_tensor_state_list[i]
        n = 1 + bool(meta_state & 1) + bool(meta_state & 2)
        per_tuuid_payloads.append(payloads[cursor : cursor + n])
        per_tuuid_counts.append(counts[cursor : cursor + n])
        cursor += n

    result = {}
    for op_name, indices in op_to_indices.items():
        sub_payloads, sub_counts = [], []
        for i in indices:
            sub_payloads.extend(per_tuuid_payloads[i])
            sub_counts.extend(per_tuuid_counts[i])
        result[op_name] = {
            "uuids": [tuuids[i] for i in indices],
            "payloads": sub_payloads,
            "counts": sub_counts,
            "shapes": [global_shapes[i] for i in indices],
            "meta": [meta_tensor_state_list[i] for i in indices],
            "compact": (
                [slice_compact_desc_params[i] for i in indices]
                if slice_compact_desc_params
                else None
            ),
        }
    return result


# ---------------------------------------------------------------------------
# New implementation (single-loop from the refactored worker.py)
# ---------------------------------------------------------------------------


def _new_dispatch(
    tuuids,
    payloads,
    counts,
    global_shapes,
    meta_tensor_state_list,
    slice_compact_desc_params,
    get_operators_for_tuuid,
):
    num = len(tuuids)
    meta_tensor_state_list = meta_tensor_state_list or [0] * num
    global_shapes = global_shapes or [None] * num

    op_bundles = {}
    cursor = 0
    for i in range(num):
        u = tuuids[i]
        meta_state = meta_tensor_state_list[i]
        n = 1 + bool(meta_state & 1) + bool(meta_state & 2)
        tuuid_payloads = payloads[cursor : cursor + n]
        tuuid_counts = counts[cursor : cursor + n]
        cursor += n

        for op_name in get_operators_for_tuuid(u):
            if op_name not in op_bundles:
                op_bundles[op_name] = ([], [], [], [], [], [])
            b = op_bundles[op_name]
            b[0].append(u)
            b[1].extend(tuuid_payloads)
            b[2].extend(tuuid_counts)
            b[3].append(global_shapes[i])
            b[4].append(meta_state)
            b[5].append(
                slice_compact_desc_params[i] if slice_compact_desc_params else None
            )

    result = {}
    for op_name, (
        sub_uuids,
        sub_p,
        sub_c,
        sub_s,
        sub_m,
        sub_compact,
    ) in op_bundles.items():
        result[op_name] = {
            "uuids": sub_uuids,
            "payloads": sub_p,
            "counts": sub_c,
            "shapes": sub_s,
            "meta": sub_m,
            "compact": (
                sub_compact if any(c is not None for c in sub_compact) else None
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arrays_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _arrays_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return a == b


def _assert_bundles_equal(old, new):
    assert set(old.keys()) == set(new.keys()), (
        f"op keys differ: {set(old.keys())} vs {set(new.keys())}"
    )
    for op in old:
        for field in ("uuids", "counts", "shapes", "meta", "compact"):
            assert _arrays_equal(old[op][field], new[op][field]), (
                f"{op}.{field}: {old[op][field]} != {new[op][field]}"
            )
        assert _arrays_equal(old[op]["payloads"], new[op]["payloads"]), (
            f"{op}.payloads mismatch"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dense_multi_tuuid_multi_op():
    """3 dense TUUIDs dispatched across 2 operators."""
    dispatch = {100: ["op_a"], 200: ["op_b"], 300: ["op_a"]}
    tuuids = [100, 200, 300]
    payloads = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6])]
    counts = [3, 2, 1]
    meta = [0, 0, 0]
    shapes = [(10,), (5,), (1,)]

    old = _old_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    new = _new_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    _assert_bundles_equal(old, new)


def test_ragged_tuuid():
    """TUUID with ragged flag (meta_state & 1) consumes 2 payload slots."""
    dispatch = {100: ["op_a"], 200: ["op_a"]}
    tuuids = [100, 200]
    payloads = [np.array([1, 2, 3, 4]), np.array([2, 2]), np.array([5, 6, 7])]
    counts = [4, 2, 3]
    meta = [1, 0]  # 100 is ragged
    shapes = [(4,), (3,)]

    old = _old_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    new = _new_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    _assert_bundles_equal(old, new)


def test_materialized_origin_index():
    """TUUID with materialized origin indices (meta_state & 2) consumes 2 payload slots."""
    dispatch = {100: ["op_a"]}
    tuuids = [100]
    payloads = [np.array([10, 20, 30]), np.array([0, 1, 2])]
    counts = [3, 3]
    meta = [2]
    shapes = [(3,)]

    old = _old_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    new = _new_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    _assert_bundles_equal(old, new)


def test_ragged_and_origin_index():
    """TUUID with both ragged and origin index (meta_state = 3) consumes 3 payload slots."""
    dispatch = {100: ["op_a"]}
    tuuids = [100]
    payloads = [np.array([10, 20, 30, 40]), np.array([2, 2]), np.array([0, 1, 2, 3])]
    counts = [4, 2, 4]
    meta = [3]
    shapes = [(4,)]

    old = _old_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    new = _new_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    _assert_bundles_equal(old, new)


def test_fan_out_tuuid_to_multiple_ops():
    """A single TUUID consumed by two operators receives identical copies."""
    dispatch = {100: ["op_a", "op_b"], 200: ["op_b"]}
    tuuids = [100, 200]
    payloads = [np.array([1, 2, 3]), np.array([4, 5])]
    counts = [3, 2]
    meta = [0, 0]
    shapes = [(3,), (2,)]

    old = _old_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    new = _new_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    _assert_bundles_equal(old, new)

    # op_a gets only TUUID 100; op_b gets both
    assert new["op_a"]["uuids"] == [100]
    assert new["op_b"]["uuids"] == [100, 200]


def test_compact_desc_params_forwarded():
    """slice_compact_desc_params are correctly associated with each TUUID."""
    dispatch = {100: ["op_a"], 200: ["op_a"]}
    tuuids = [100, 200]
    payloads = [np.array([1, 2]), np.array([3, 4])]
    counts = [2, 2]
    meta = [0, 0]
    shapes = [(2,), (2,)]
    compact = [[0, 2], [2, 4]]

    old = _old_dispatch(
        tuuids, payloads, counts, shapes, meta, compact, dispatch.__getitem__
    )
    new = _new_dispatch(
        tuuids, payloads, counts, shapes, meta, compact, dispatch.__getitem__
    )
    _assert_bundles_equal(old, new)
    assert new["op_a"]["compact"] == [[0, 2], [2, 4]]


def test_empty_tuuid_list():
    """Empty input produces empty output."""
    old = _old_dispatch([], [], [], [], [], None, lambda u: [])
    new = _new_dispatch([], [], [], [], [], None, lambda u: [])
    assert old == new == {}


def test_mixed_meta_states_in_bundle():
    """Bundle with mixed dense/ragged/indexed TUUIDs in a single RPC call."""
    dispatch = {1: ["op_x"], 2: ["op_x"], 3: ["op_x"]}
    # TUUID 1: dense (1 slot), TUUID 2: ragged (2 slots), TUUID 3: both (3 slots)
    tuuids = [1, 2, 3]
    payloads = [
        np.array([10]),  # TUUID 1 data
        np.array([20, 21]),
        np.array([1, 1]),  # TUUID 2 data + lengths
        np.array([30, 31]),  # TUUID 3 data
        np.array([1, 1]),  # TUUID 3 lengths
        np.array([0, 1]),  # TUUID 3 origin indices
    ]
    counts = [1, 2, 2, 2, 2, 2]
    meta = [0, 1, 3]
    shapes = [(1,), (2,), (2,)]

    old = _old_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    new = _new_dispatch(
        tuuids, payloads, counts, shapes, meta, None, dispatch.__getitem__
    )
    _assert_bundles_equal(old, new)
