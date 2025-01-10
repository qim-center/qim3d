import qim3d
import pytest
from torch import ones

from qim3d.tests import temp_data


# unit test for model summary()
def test_model_summary():
    n = 10
    img_shape = (32, 32)
    folder = "folder_data"
    temp_data(folder, img_shape=img_shape, n=n)

    unet = qim3d.ml.models.UNet(size="small")
    augment = qim3d.ml.Augmentation(transform_train=None)
    train_set, val_set, test_set = qim3d.ml.prepare_datasets(
        folder, 1 / 3, unet, augment
    )

    _, val_loader, _ = qim3d.ml.prepare_dataloaders(
        train_set, val_set, test_set, batch_size=1, num_workers=1, pin_memory=False
    )
    summary = qim3d.ml.model_summary(val_loader, unet)

    assert summary.input_size[0] == (1, 1) + img_shape

    temp_data(folder, remove=True)


# unit test for inference()
def test_inference():
    folder = "folder_data"
    temp_data(folder)

    unet = qim3d.ml.models.UNet(size="small")
    augment = qim3d.ml.Augmentation(transform_train=None)
    train_set, _, _ = qim3d.ml.prepare_datasets(folder, 1 / 3, unet, augment)

    _, targ, _ = qim3d.ml.inference(train_set, unet)

    assert tuple(targ[0].unique()) == (0, 1)

    temp_data(folder, remove=True)


# unit test for tuple ValueError().
def test_inference_tuple():
    folder = "folder_data"
    temp_data(folder)

    unet = qim3d.ml.models.UNet(size="small")

    data = [1, 2, 3]
    with pytest.raises(ValueError, match="Data items must be tuples"):
        qim3d.ml.inference(data, unet)

    temp_data(folder, remove=True)


# unit test for tensor ValueError().
def test_inference_tensor():
    folder = "folder_data"
    temp_data(folder)

    unet = qim3d.ml.models.UNet(size="small")

    data = [(1, 2)]
    with pytest.raises(ValueError, match="Data items must consist of tensors"):
        qim3d.ml.inference(data, unet)

    temp_data(folder, remove=True)


# unit test for dimension ValueError().
def test_inference_dim():
    folder = "folder_data"
    temp_data(folder)

    unet = qim3d.ml.models.UNet(size="small")

    data = [(ones(1), ones(1))]
    # need the r"" for special characters
    with pytest.raises(ValueError, match=r"Input image must be \(C,H,W\) format"):
        qim3d.ml.inference(data, unet)

    temp_data(folder, remove=True)


# unit test for train_model()
def test_train_model():
    folder = "folder_data"
    temp_data(folder)

    n_epochs = 1

    unet = qim3d.ml.models.UNet(size="small")
    augment = qim3d.ml.Augmentation(transform_train=None)
    hyperparams = qim3d.ml.Hyperparameters(unet, n_epochs=n_epochs)
    train_set, val_set, test_set = qim3d.ml.prepare_datasets(
        folder, 1 / 3, unet, augment
    )
    train_loader, val_loader, _ = qim3d.ml.prepare_dataloaders(
        train_set, val_set, test_set, batch_size=1, num_workers=1, pin_memory=False
    )

    train_loss, _ = qim3d.ml.train_model(
        unet, hyperparams, train_loader, val_loader, plot=False, return_loss=True
    )

    assert len(train_loss["loss"]) == n_epochs

    temp_data(folder, remove=True)
