import qim3d
import matplotlib.pyplot as plt
import pytest

from torch import ones
from qim3d.utils.internal_tools import temp_data

# unit tests for grid overview
def test_grid_overview():
    random_tuple = (ones(1,256,256),ones(256,256))
    n_images = 10 
    train_set = [random_tuple for t in range(n_images)]

    fig = qim3d.viz.grid_overview(train_set,num_images=n_images)
    assert fig.get_figwidth() == 2*n_images


def test_grid_overview_tuple():
    random_tuple = (ones(256,256),ones(256,256))

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
def test_slice_viz():
    example_volume = ones(10,10,10)
    img_width = 3
    fig = qim3d.viz.slice_viz(example_volume,img_width = img_width)

    assert fig.get_figwidth() == img_width


def test_slice_viz_not_volume():
    example_volume = ones(10,10)
    dim = example_volume.ndim
    with pytest.raises(ValueError, match = f"Given array is not a volume! Current dimension: {dim}"):
        qim3d.viz.slice_viz(example_volume)


def test_slice_viz_wrong_slice():
    example_volume = ones(10,10,10)
    with pytest.raises(ValueError, match = 'Position not recognized. Choose an integer, list, array or "start","mid","end".'):
        qim3d.viz.slice_viz(example_volume, position = 'invalid_slice')
