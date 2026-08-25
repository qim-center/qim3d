import numpy as np
import pytest

import qim3d
from qim3d.segmentation._connected_components import connected_components


# Unit tests for connected_components()
@pytest.fixture(scope='module')
def setup_data():
    components = np.array(
        [[[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]]]
    )
    num_components = 4
    connected_components_ = connected_components(components)
    return connected_components_, components, num_components


def test_connected_components_property(setup_data):
    connected_components_, _, _ = setup_data
    components = np.array(
        [[[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [2, 2, 0, 0, 3, 0], [0, 0, 0, 4, 0, 0]]]
    )
    assert np.array_equal(connected_components_.get_cc(), components)


def test_num_connected_components_property(setup_data):
    connected_components, _, num_components = setup_data
    assert len(connected_components) == num_components


def test_get_connected_component_with_index(setup_data):
    connected_components, _, _ = setup_data
    expected_component = np.array(
        [[
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]],
        dtype=bool,
    )
    print(connected_components.get_cc(index=1))
    print(expected_component)
    assert np.array_equal(connected_components.get_cc(index=1), expected_component)


def test_get_connected_component_without_index(setup_data):
    connected_components, _, _ = setup_data
    component = connected_components.get_cc()
    assert np.any(component)


def test_get_connected_component_with_invalid_index(setup_data):
    connected_components, _, num_components = setup_data
    with pytest.raises(AssertionError):
        connected_components.get_cc(index=0)
    with pytest.raises(AssertionError):
        connected_components.get_cc(index=num_components + 1)

# Unit tests for watershed()
def test_watershed():
    # Create a small 3D binary volume with distinct objects
    bin_vol = np.zeros((10, 10, 10), dtype=np.uint8)
    bin_vol[2:4, 2:4, 2:4] = 1  # Object 1
    bin_vol[6:8, 6:8, 6:8] = 1  # Object 2

    # Apply the watershed function
    labeled_volume, num_labels = qim3d.segmentation.watershed(bin_vol, min_distance=2)

    # Check if the segmentation has been applied correctly
    assert bin_vol.shape == labeled_volume.shape, "Volume shape changed"
    assert num_labels == 2, f"Expected 2 objects, but found {num_labels}"

    # Check that the objects are correctly labeled
    assert np.unique(labeled_volume[2:4, 2:4, 2:4]) == [1], "Object 1 not labeled correctly"
    assert np.unique(labeled_volume[6:8, 6:8, 6:8]) == [2], "Object 2 not labeled correctly"

    # Check that the background is labeled as 0
    assert np.unique(labeled_volume[0, 0, 0]) == [0], "Background not labeled as 0"

    # Check that the labels are unique and consecutive
    unique_labels = np.unique(labeled_volume)
    assert np.array_equal(unique_labels, np.arange(num_labels + 1)), "Labels are not unique and consecutive"
