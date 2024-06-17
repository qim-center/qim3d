import pytest
import torch
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pytest
from torch import ones

import qim3d
from qim3d.utils.internal_tools import temp_data

import matplotlib.pyplot as plt
import ipywidgets as widgets

# unit tests for grid overview
def test_grid_overview():
    random_tuple = (torch.ones(1,256,256),torch.ones(256,256))
    n_images = 10 
    train_set = [random_tuple for t in range(n_images)]

    fig = qim3d.viz.grid_overview(train_set,num_images=n_images)
    assert fig.get_figwidth() == 2*n_images


def test_grid_overview_tuple():
    random_tuple = (torch.ones(256,256),torch.ones(256,256))

    with pytest.raises(ValueError,match="Data elements must be tuples"):
        qim3d.viz.grid_overview(random_tuple,num_images=1)


# unit tests for grid prediction
def test_grid_pred():
    folder = 'folder_data'
    n = 4
    temp_data(folder,n = n)

    model = qim3d.models.UNet()
    augmentation = qim3d.utils.Augmentation()
    train_set,_,_ = qim3d.utils.prepare_datasets(folder,0.1,model,augmentation)

    in_targ_pred = qim3d.utils.models.inference(train_set,model)

    fig = qim3d.viz.grid_pred(in_targ_pred)
    
    assert (fig.get_figwidth(),fig.get_figheight()) == (2*(n),10)

    temp_data(folder,remove = True)


# unit tests for slices function
def test_slices_numpy_array_input():
    example_volume = np.ones((10, 10, 10))
    fig = qim3d.viz.slices(example_volume, n_slices=1)
    assert isinstance(fig, plt.Figure)

def test_slices_torch_tensor_input():
    example_volume = torch.ones((10,10,10))
    img_width = 3
    fig = qim3d.viz.slices(example_volume,n_slices = 1)
    assert isinstance(fig, plt.Figure)

def test_slices_wrong_input_format():
    input = 'not_a_volume'
    with pytest.raises(ValueError, match = 'Data type not supported'):
        qim3d.viz.slices(input)

def test_slices_not_volume():
    example_volume = np.ones((10,10))
    with pytest.raises(ValueError, match = 'The provided object is not a volume as it has less than 3 dimensions.'):
        qim3d.viz.slices(example_volume)

def test_slices_wrong_position_format1():
    example_volume = np.ones((10,10,10))
    with pytest.raises(ValueError, match = 'Position not recognized. Choose an integer, list of integers or one of the following strings: "start", "mid" or "end".'):
        qim3d.viz.slices(example_volume, position = 'invalid_slice')

def test_slices_wrong_position_format2():
    example_volume = np.ones((10,10,10))
    with pytest.raises(ValueError, match = 'Position not recognized. Choose an integer, list of integers or one of the following strings: "start", "mid" or "end".'):
        qim3d.viz.slices(example_volume, position = 1.5)

def test_slices_wrong_position_format3():
    example_volume = np.ones((10,10,10))
    with pytest.raises(ValueError, match = 'Position not recognized. Choose an integer, list of integers or one of the following strings: "start", "mid" or "end".'):
        qim3d.viz.slices(example_volume, position = [1, 2, 3.5])

def test_slices_invalid_axis_value():
    example_volume = np.ones((10,10,10))
    with pytest.raises(ValueError, match = "Invalid value for 'axis'. It should be an integer between 0 and 2"):
        qim3d.viz.slices(example_volume, axis = 3)

def test_slices_interpolation_option():
    example_volume = torch.ones((10, 10, 10))
    img_width = 3
    interpolation_method = 'bilinear'
    fig = qim3d.viz.slices(example_volume, n_slices=1, img_width=img_width, interpolation=interpolation_method)

    for ax in fig.get_axes():
        # Access the interpolation method used for each Axes object
        actual_interpolation = ax.images[0].get_interpolation()

        # Assert that the actual interpolation method matches the expected method
        assert actual_interpolation == interpolation_method

def test_slices_multiple_slices():
    example_volume = np.ones((10, 10, 10))
    img_width = 3
    n_slices = 3
    fig = qim3d.viz.slices(example_volume, n_slices=n_slices, img_width=img_width)
    # Add assertions for the expected number of subplots in the figure
    assert len(fig.get_axes()) == n_slices

def test_slices_axis_argument():
    # Non-symmetric input
    example_volume = np.arange(1000).reshape((10, 10, 10))
    img_width = 3

    # Call the function with different values of the axis
    fig_axis_0 = qim3d.viz.slices(example_volume, n_slices=1, img_width=img_width, axis=0)
    fig_axis_1 = qim3d.viz.slices(example_volume, n_slices=1, img_width=img_width, axis=1)
    fig_axis_2 = qim3d.viz.slices(example_volume, n_slices=1, img_width=img_width, axis=2)

    # Ensure that different axes result in different plots
    assert not np.allclose(fig_axis_0.get_axes()[0].images[0].get_array(), fig_axis_1.get_axes()[0].images[0].get_array())
    assert not np.allclose(fig_axis_1.get_axes()[0].images[0].get_array(), fig_axis_2.get_axes()[0].images[0].get_array())
    assert not np.allclose(fig_axis_2.get_axes()[0].images[0].get_array(), fig_axis_0.get_axes()[0].images[0].get_array())

# unit tests for slicer function
def test_slicer_with_numpy_array():
    # Create a sample NumPy array
    vol = np.random.rand(10, 10, 10)
    # Call the slicer function with the NumPy array
    slicer_obj = qim3d.viz.slicer(vol)
    # Assert that the slicer object is created successfully
    assert isinstance(slicer_obj, widgets.interactive)

def test_slicer_with_torch_tensor():
    # Create a sample PyTorch tensor
    vol = torch.rand(10, 10, 10)
    # Call the slicer function with the PyTorch tensor
    slicer_obj = qim3d.viz.slicer(vol)
    # Assert that the slicer object is created successfully
    assert isinstance(slicer_obj, widgets.interactive)

def test_slicer_with_different_parameters():
    # Test with different axis values
    for axis in range(3):
        slicer_obj = qim3d.viz.slicer(np.random.rand(10, 10, 10), axis=axis)
        assert isinstance(slicer_obj, widgets.interactive)

    # Test with different colormaps
    for cmap in ["viridis", "gray", "plasma"]:
        slicer_obj = qim3d.viz.slicer(np.random.rand(10, 10, 10), cmap=cmap)
        assert isinstance(slicer_obj, widgets.interactive)

    # Test with different image sizes
    for img_height, img_width in [(2, 2), (4, 4)]:
        slicer_obj = qim3d.viz.slicer(np.random.rand(10, 10, 10), img_height=img_height, img_width=img_width)
        assert isinstance(slicer_obj, widgets.interactive)

    # Test with show_position set to True and False
    for show_position in [True, False]:
        slicer_obj = qim3d.viz.slicer(np.random.rand(10, 10, 10), show_position=show_position)
        assert isinstance(slicer_obj, widgets.interactive)

# unit tests for orthogonal function
def test_orthogonal_with_numpy_array():
    # Create a sample NumPy array
    vol = np.random.rand(10, 10, 10)
    # Call the orthogonal function with the NumPy array
    orthogonal_obj = qim3d.viz.orthogonal(vol)
    # Assert that the orthogonal object is created successfully
    assert isinstance(orthogonal_obj, widgets.HBox)

def test_orthogonal_with_torch_tensor():
    # Create a sample PyTorch tensor
    vol = torch.rand(10, 10, 10)
    # Call the orthogonal function with the PyTorch tensor
    orthogonal_obj = qim3d.viz.orthogonal(vol)
    # Assert that the orthogonal object is created successfully
    assert isinstance(orthogonal_obj, widgets.HBox)

def test_orthogonal_with_different_parameters():
    # Test with different colormaps
    for cmap in ["viridis", "gray", "plasma"]:
        orthogonal_obj = qim3d.viz.orthogonal(np.random.rand(10, 10, 10), cmap=cmap)
        assert isinstance(orthogonal_obj, widgets.HBox)

    # Test with different image sizes
    for img_height, img_width in [(2, 2), (4, 4)]:
        orthogonal_obj = qim3d.viz.orthogonal(np.random.rand(10, 10, 10), img_height=img_height, img_width=img_width)
        assert isinstance(orthogonal_obj, widgets.HBox)

    # Test with show_position set to True and False
    for show_position in [True, False]:
        orthogonal_obj = qim3d.viz.orthogonal(np.random.rand(10, 10, 10), show_position=show_position)
        assert isinstance(orthogonal_obj, widgets.HBox)

def test_orthogonal_initial_slider_value():
    # Create a sample NumPy array
    vol = np.random.rand(10, 7, 19)
    # Call the orthogonal function with the NumPy array
    orthogonal_obj = qim3d.viz.orthogonal(vol)
    for idx,slicer in enumerate(orthogonal_obj.children):
        assert slicer.children[0].value == vol.shape[idx]//2

def test_orthogonal_slider_description():
    # Create a sample NumPy array
    vol = np.random.rand(10, 10, 10)
    # Call the orthogonal function with the NumPy array
    orthogonal_obj = qim3d.viz.orthogonal(vol)
    for idx,slicer in enumerate(orthogonal_obj.children):
        assert slicer.children[0].description == ['Z', 'Y', 'X'][idx]





# unit tests for local thickness visualization
def test_local_thickness_2d():
    blobs = qim3d.examples.blobs_256x256
    lt = qim3d.processing.local_thickness(blobs)
    fig = qim3d.viz.local_thickness(blobs, lt)

    # Assert that returned figure is a matplotlib figure
    assert isinstance(fig, plt.Figure)

def test_local_thickness_3d():
    fly = qim3d.examples.fly_150x256x256
    lt = qim3d.processing.local_thickness(fly)
    obj = qim3d.viz.local_thickness(fly, lt)

    # Assert that returned object is an interactive widget
    assert isinstance(obj, widgets.interactive)

def test_local_thickness_3d_max_projection():
    fly = qim3d.examples.fly_150x256x256
    lt = qim3d.processing.local_thickness(fly)
    fig = qim3d.viz.local_thickness(fly, lt, max_projection=True)

    # Assert that returned object is an interactive widget
    assert isinstance(fig, plt.Figure)
