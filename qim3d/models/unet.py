"""UNet model and Hyperparameters class."""

import torch.nn as nn

from qim3d.utils.logger import log


class UNet(nn.Module):
    """
    2D UNet model for QIM imaging.

    This class represents a 2D UNet model designed for imaging segmentation tasks.

    Args:
        size (str, optional): Size of the UNet model. Must be one of 'small', 'medium', or 'large'. Defaults to 'medium'.
        dropout (float, optional): Dropout rate between 0 and 1. Defaults to 0.
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        up_kernel_size (int, optional): Up-convolution kernel size. Defaults to 3.
        activation (str, optional): Activation function. Defaults to 'PReLU'.
        bias (bool, optional): Whether to include bias in convolutions. Defaults to True.
        adn_order (str, optional): ADN (Activation, Dropout, Normalization) ordering. Defaults to 'NDA'.

    Raises:
        ValueError: If `size` is not one of 'small', 'medium', or 'large'.
    """

    def __init__(
        self,
        size="medium",
        dropout=0,
        kernel_size=3,
        up_kernel_size=3,
        activation="PReLU",
        bias=True,
        adn_order="NDA",
    ):
        super().__init__()
        if size not in ["small", "medium", "large"]:
            raise ValueError(
                f"Invalid model size: {size}. Size must be one of the following: 'small', 'medium', 'large'."
            )

        self.size = size
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.activation = activation
        self.bias = bias
        self.adn_order = adn_order

        self.model = self._model_choice()

    def _model_choice(self):
        from monai.networks.nets import UNet as monai_UNet

        if self.size == "small":
            self.channels = (64, 128, 256)
        elif self.size == "medium":
            self.channels = (64, 128, 256, 512, 1024)
        elif self.size == "large":
            self.channels = (64, 128, 256, 512, 1024, 2048)

        model = monai_UNet(
            spatial_dims=2,
            in_channels=1,  # TODO: check if image has 1 or multiple input channels
            out_channels=1,
            channels=self.channels,
            strides=(2,) * (len(self.channels) - 1),
            kernel_size=self.kernel_size,
            up_kernel_size=self.up_kernel_size,
            act=self.activation,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_order,
        )
        return model

    def forward(self, x):
        x = self.model(x)
        return x


class Hyperparameters:
    """
    Hyperparameters for QIM segmentation.

    Args:
        model (torch.nn.Module): PyTorch model.
        n_epochs (int, optional): Number of training epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        optimizer (str, optional): Optimizer algorithm. Must be one of 'Adam', 'SGD', 'RMSprop'. Defaults to 'Adam'.
        momentum (float, optional): Momentum value for SGD and RMSprop optimizers. Defaults to 0.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 0.
        loss_function (str, optional): Loss function criterion. Must be one of 'BCE', 'Dice', 'Focal', 'DiceCE'. Defaults to 'BCE'.

    Raises:
        ValueError: If `loss_function` is not one of 'BCE', 'Dice', 'Focal', 'DiceCE'.
        ValueError: If `optimizer` is not one of 'Adam', 'SGD', 'RMSprop'.

    Example:
        ```
        import qim3d

        # This examples shows how to define a UNet model and its hyperparameters.

        # Defining the model
        my_model = qim3d.models.UNet(size='medium')

        # Choosing the hyperparameters
        hyperparams = qim3d.models.Hyperparameters(model=my_model, n_epochs=20, learning_rate=0.001)

        params_dict = hyperparams() # Get the hyperparameters
        optimizer = params_dict['optimizer']
        criterion = params_dict['criterion']
        n_epochs  = params_dict['n_epochs']

        ```
    """

    def __init__(
        self,
        model,
        n_epochs=10,
        learning_rate=1e-3,
        optimizer="Adam",
        momentum=0,
        weight_decay=0,
        loss_function="Focal",
    ):

        # TODO: implement custom loss_functions? then add a check to see if loss works for segmentation.
        if loss_function not in ["BCE", "Dice", "Focal", "DiceCE"]:
            raise ValueError(
                f"Invalid loss function: {loss_function}. Loss criterion must "
                "be one of the following: 'BCE','Dice','Focal','DiceCE'."
            )
        # TODO: implement custom optimizer? and add check to see if valid.
        if optimizer not in ["Adam", "SGD", "RMSprop"]:
            raise ValueError(
                f"Invalid optimizer: {optimizer}. Optimizer must "
                "be one of the following: 'Adam', 'SGD', 'RMSprop'."
            )

        if (momentum != 0) and optimizer == "Adam":
            log.info(
                "Momentum isn't an input in the 'Adam' optimizer. "
                "Change optimizer to 'SGD' or 'RMSprop' to use momentum."
            )

        self.model = model
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.loss_function = loss_function

    def __call__(self):
        return self.model_params(
            self.model,
            self.n_epochs,
            self.optimizer,
            self.learning_rate,
            self.weight_decay,
            self.momentum,
            self.loss_function,
        )

    def model_params(
        self,
        model,
        n_epochs,
        optimizer,
        learning_rate,
        weight_decay,
        momentum,
        loss_function,
    ):

        optim = self._optimizer(model, optimizer, learning_rate, weight_decay, momentum)
        criterion = self._loss_functions(loss_function)

        hyper_dict = {
            "optimizer": optim,
            "criterion": criterion,
            "n_epochs": n_epochs,
        }
        return hyper_dict

    # selecting the optimizer
    def _optimizer(self, model, optimizer, learning_rate, weight_decay, momentum):
        from torch.optim import Adam, SGD, RMSprop

        if optimizer == "Adam":
            optim = Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "SGD":
            optim = SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "RMSprop":
            optim = RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        return optim

    # selecting the loss function
    def _loss_functions(self, loss_function):
        from monai.losses import FocalLoss, DiceLoss, DiceCELoss
        from torch.nn import BCEWithLogitsLoss

        if loss_function == "BCE":
            criterion = BCEWithLogitsLoss(reduction="mean")
        elif loss_function == "Dice":
            criterion = DiceLoss(sigmoid=True, reduction="mean")
        elif loss_function == "Focal":
            criterion = FocalLoss(reduction="mean")
        elif loss_function == "DiceCE":
            criterion = DiceCELoss(sigmoid=True, reduction="mean")
        return criterion
