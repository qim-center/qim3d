import numpy as np
import pytest

from qim3d.utils.connected_components import get_3d_connected_components


@pytest.fixture(scope="module")
def setup_data():
    components = np.array([[0,0,1,1,0,0],
                           [0,0,0,1,0,0],
                           [1,1,0,0,1,0],
                           [0,0,0,1,0,0]])
    num_components = 4
    connected_components = get_3d_connected_components(components)
    return connected_components, components, num_components

def test_connected_components_property(setup_data):
    connected_components, _, _ = setup_data
    components = np.array([[0,0,1,1,0,0],
                            [0,0,0,1,0,0],
                            [2,2,0,0,3,0],
                            [0,0,0,4,0,0]])
    assert np.array_equal(connected_components.connected_components, components)

def test_num_connected_components_property(setup_data):
    connected_components, _, num_components = setup_data
    assert connected_components.num_connected_components == num_components

def test_get_connected_component_with_index(setup_data):
    connected_components, _, _ = setup_data
    expected_component = np.array([[0,0,1,1,0,0],
                                    [0,0,0,1,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]], dtype=bool)
    print(connected_components.get_connected_component(index=1))
    print(expected_component)
    assert np.array_equal(connected_components.get_connected_component(index=1), expected_component)

def test_get_connected_component_without_index(setup_data):
    connected_components, _, _ = setup_data
    component = connected_components.get_connected_component()
    assert np.any(component)

def test_get_connected_component_with_invalid_index(setup_data):
    connected_components, _, num_components = setup_data
    with pytest.raises(AssertionError):
        connected_components.get_connected_component(index=0)
    with pytest.raises(AssertionError):
        connected_components.get_connected_component(index=num_components + 1)