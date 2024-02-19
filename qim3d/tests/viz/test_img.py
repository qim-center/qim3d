import torch
import numpy as np
import qim3d
import pytest
from qim3d.utils.internal_tools import temp_data

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


# unit tests for slice visualization
def test_slices_numpy_array_input():
    example_volume = np.ones((10, 10, 10))
    img_width = 3
    fig = qim3d.viz.slices(example_volume, n_slices=1, img_width=img_width)
    assert fig.get_figwidth() == img_width

def test_slices_torch_tensor_input():
    example_volume = torch.ones((10,10,10))
    img_width = 3
    fig = qim3d.viz.slices(example_volume,n_slices = 1, img_width = img_width)

    assert fig.get_figwidth() == img_width

def test_slices_wrong_input_format():
    input = 'not_a_volume'
    with pytest.raises(ValueError, match = 'Input must be a numpy.ndarray or torch.Tensor'):
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

def test_slices_show_title_option():
    example_volume = np.ones((10, 10, 10))
    img_width = 3
    fig = qim3d.viz.slices(example_volume, n_slices=1, img_width=img_width, show_title=False)
    # Assert that titles are not shown
    assert all(ax.get_title() == '' for ax in fig.get_axes())

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