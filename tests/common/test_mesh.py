"""
Unit tests for DeviceMesh.
"""

import pytest
import numpy as np
from fluxway.common.mesh import DeviceMesh


def test_device_mesh_initialization():
    """Test basic initialization with default device IDs."""
    shape = (2, 4)
    axis_names = ("data", "model")
    mesh = DeviceMesh(shape, axis_names)

    assert mesh.shape == shape
    assert mesh.axis_names == axis_names
    # Default IDs should be 0..7
    expected_ids = np.arange(8).reshape(2, 4)
    np.testing.assert_array_equal(mesh.device_ids, expected_ids)

    assert mesh.get_axis_size("data") == 2
    assert mesh.get_axis_size("model") == 4


def test_device_mesh_inference_from_ids():
    """Test shape inference from multi-dimensional device_ids."""
    axis_names = ("x", "y")
    ids = np.array([[10, 11], [12, 13]])
    mesh = DeviceMesh(axis_names=axis_names, device_ids=ids)

    assert mesh.shape == (2, 2)
    np.testing.assert_array_equal(mesh.device_ids, ids)
    assert mesh.get_rank((1, 0)) == 12


def test_device_mesh_1d_inference():
    """Test inference from 1D ids when axis_names has length 1."""
    mesh = DeviceMesh(axis_names=("data",), device_ids=[0, 1, 2])
    assert mesh.shape == (3,)

    with pytest.raises(ValueError, match="Cannot infer multi-dimensional shape"):
        DeviceMesh(axis_names=("x", "y"), device_ids=[0, 1, 2, 3])


def test_device_mesh_required_args():
    """Test that at least one of shape or ids is required."""
    with pytest.raises(ValueError, match="axis_names must be provided"):
        DeviceMesh(device_shape=(2,))

    with pytest.raises(
        ValueError, match="Either device_shape or device_ids must be provided"
    ):
        DeviceMesh(axis_names=("x",))


def test_coordinate_mapping():
    """Test rank <-> coordinate mapping."""
    shape = (2, 2, 2)
    axis_names = ("a", "b", "c")
    mesh = DeviceMesh(shape, axis_names)

    # Check a few specific points
    # (0, 0, 0) -> 0
    assert mesh.get_coordinate(0) == (0, 0, 0)
    assert mesh.get_rank((0, 0, 0)) == 0

    # (1, 1, 1) -> 7 (last element)
    assert mesh.get_coordinate(7) == (1, 1, 1)
    assert mesh.get_rank((1, 1, 1)) == 7

    # (0, 1, 0) -> 2 (0*4 + 1*2 + 0)
    assert mesh.get_coordinate(2) == (0, 1, 0)
    assert mesh.get_rank((0, 1, 0)) == 2


def test_invalid_initialization():
    """Test error handling during initialization."""
    # Mismatch shape/names length
    with pytest.raises(ValueError, match="must have same length"):
        DeviceMesh((2, 2), ("x",))

    # Mismatch device_ids size
    with pytest.raises(ValueError, match="does not match device_shape"):
        DeviceMesh((2, 2), ("x", "y"), device_ids=[0, 1, 2])


def test_invalid_access():
    """Test error handling for invalid access."""
    mesh = DeviceMesh((2, 2), ("x", "y"))

    with pytest.raises(ValueError, match="Axis z not found"):
        mesh.get_axis_size("z")

    with pytest.raises(ValueError, match="Rank 100 not found"):
        mesh.get_coordinate(100)

    # Rank 0 exists, but coordinate dim mismatch
    # Python generic IndexError or custom? Implementation might raise IndexError or strict check
    # Let's check implementation behavior for dimensionality
    # If we pass (0,0,0) to 2D mesh, numpy indexing might fail or succeed depending on structure
    # The implementation as written accesses self.device_ids[coord]
    # For robust code, we should probably check lengths in get_rank too, but current test is good check.
    # We added explicit check in implementation: "Coordinate {coord} dim does not match"

    with pytest.raises(ValueError, match="dim does not match"):
        mesh.get_rank((0, 0, 0))
