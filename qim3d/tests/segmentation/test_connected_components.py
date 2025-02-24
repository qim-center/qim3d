import numpy as np
import pytest

from qim3d.segmentation._connected_components import get_3d_cc


@pytest.fixture(scope='module')
def setup_data():
    components = np.array(
        [[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]]
    )
    num_components = 4
    connected_components = get_3d_cc(components)
    return connected_components, components, num_components


def test_connected_components_property(setup_data):
    connected_components, _, _ = setup_data
    components = np.array(
        [[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [2, 2, 0, 0, 3, 0], [0, 0, 0, 4, 0, 0]]
    )
    assert np.array_equal(connected_components.get_cc(), components)


def test_num_connected_components_property(setup_data):
    connected_components, _, num_components = setup_data
    assert len(connected_components) == num_components


def test_get_connected_component_with_index(setup_data):
    connected_components, _, _ = setup_data
    expected_component = np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
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
