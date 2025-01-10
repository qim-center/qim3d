import qim3d
import pytest

from torch.utils.data.dataloader import DataLoader
from qim3d.tests import temp_data

# unit tests for Dataset()
def test_dataset():
    img_shape = (32,32)
    folder = 'folder_data'
    temp_data(folder, img_shape = img_shape)
    
    images = qim3d.ml.Dataset(folder)

    assert images[0][0].shape == img_shape

    temp_data(folder,remove=True)


# unit tests for check_resize()
def test_check_resize():
    h_adjust,w_adjust = qim3d.ml._data.check_resize(240,240,resize = 'crop',n_channels = 6)

    assert (h_adjust,w_adjust) == (192,192)

def test_check_resize_pad():
    h_adjust,w_adjust = qim3d.ml._data.check_resize(16,16,resize = 'padding',n_channels = 6)

    assert (h_adjust,w_adjust) == (64,64)

def test_check_resize_fail():

    with pytest.raises(ValueError,match="The size of the image is too small compared to the depth of the UNet. Choose a different 'resize' and/or a smaller model."):
        h_adjust,w_adjust = qim3d.ml._data.check_resize(16,16,resize = 'crop',n_channels = 6)


# unit tests for prepare_datasets()
def test_prepare_datasets():
    n = 3
    validation = 1/3
    
    folder = 'folder_data'
    img = temp_data(folder,n = n)

    my_model = qim3d.ml.models.UNet()
    my_augmentation = qim3d.ml.Augmentation(transform_test='light')
    train_set, val_set, test_set = qim3d.ml.prepare_datasets(folder,validation,my_model,my_augmentation)

    assert (len(train_set),len(val_set),len(test_set)) == (int((1-validation)*n), int(n*validation), n)

    temp_data(folder,remove=True)


# unit test for validation in prepare_datasets()
def test_validation():
    validation = 10
    
    with pytest.raises(ValueError,match = "The validation fraction must be a float between 0 and 1."):
        augment_class = qim3d.ml.prepare_datasets('folder',validation,'my_model','my_augmentation')


# unit test for prepare_dataloaders()
def test_prepare_dataloaders():
    folder = 'folder_data'
    temp_data(folder)

    batch_size = 1
    my_model = qim3d.ml.models.UNet()
    my_augmentation = qim3d.ml.Augmentation()
    train_set, val_set, test_set = qim3d.ml.prepare_datasets(folder,1/3,my_model,my_augmentation)

    _,val_loader,_ = qim3d.ml.prepare_dataloaders(train_set,val_set,test_set,
                                                                           batch_size,num_workers = 1,
                                                                           pin_memory = False)
    
    assert type(val_loader) == DataLoader

    temp_data(folder,remove=True)