import qim3d
import torch

# unit tests for UNet()
def test_starting_unet():
    unet = qim3d.ml.models.UNet()

    assert unet.size == 'medium'


def test_forward_pass():
    unet = qim3d.ml.models.UNet()

    # Size: B x C x H x W
    x = torch.ones([1,1,256,256])

    output = unet(x)
    assert x.shape == output.shape

# unit tests for Hyperparameters()
def test_hyper():
    unet = qim3d.ml.models.UNet()
    hyperparams = qim3d.ml.models.Hyperparameters(unet)

    assert hyperparams.n_epochs == 10

def test_hyper_dict():
    unet = qim3d.ml.models.UNet()
    hyperparams = qim3d.ml.models.Hyperparameters(unet)

    hyper_dict = hyperparams()

    assert type(hyper_dict) == dict    
