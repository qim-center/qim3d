"""UNet model and Hyperparameters class."""

import torch
import torch.nn as nn

from qim3d.utils import log


class UNet(nn.Module):
    """
    3D UNet model designed for imaging segmentation tasks.

    Args:
        size (str, optional): Size of the UNet model. Must be one of 'xxsmall', 'xsmall', 'small', 'medium', 'large', 'xlarge', or 'xxlarge'. Default is 'medium'.
        dropout (float, optional): Dropout rate between 0 and 1. Default is 0.
        kernel_size (int, optional): Convolution kernel size. Default is 3.
        up_kernel_size (int, optional): Up-convolution kernel size. Default is 3.
        activation (str, optional): Activation function. Default is 'PReLU'.
        bias (bool, optional): Whether to include bias in convolutions. Default is True.
        adn_order (str, optional): ADN (Activation, Dropout, Normalization) ordering. Default is 'NDA'.

    Returns:
        model (torch.nn.Module): 3D UNet model.

    Raises:
        ValueError: If `size` is not one of 'small', 'medium', or 'large'.

    Example:
        ```python
        import qim3d

        model = qim3d.ml.models.UNet(size = 'small')
        ```

    """

    def __init__(
        self,
        size: str = 'medium',
        dropout: float = 0,
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        activation: str = 'PReLU',
        bias: bool = True,
        adn_order: str = 'NDA',
    ):
        super().__init__()

        self.size = size
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.activation = activation
        self.bias = bias
        self.adn_order = adn_order

        self.model = self._model_choice()

    def _model_choice(self) -> nn.Module:
        from monai.networks.nets import UNet as monai_UNet

        size_options = {
            'xxsmall': (4, 8),  # 2 layers
            'xsmall': (16, 32),  # 2 layers
            'small': (32, 64, 128),  # 3 layers
            'medium': (64, 128, 256),  # 3 layers
            'large': (64, 128, 256, 512, 1024),  # 5 layers
            'xlarge': (64, 128, 256, 512, 1024, 2048),  # 6 layers
            'xxlarge': (64, 128, 256, 512, 1024, 2048, 4096),  # 7 layers
        }

        if self.size in size_options:
            self.channels = size_options[self.size]
        else:
            message = (
                f"Unknown size '{self.size}'. Choose from {list(size_options.keys())}"
            )
            raise ValueError(message)

        model = monai_UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=self.channels,
            strides=(2,) * (len(self.channels) - 1),
            num_res_units=2,
            kernel_size=self.kernel_size,
            up_kernel_size=self.up_kernel_size,
            act=self.activation,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_order,
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class Hyperparameters:
    """
    Hyperparameters for training the 3D UNet model.

    Args:
        model (torch.nn.Module): PyTorch model.
        n_epochs (int, optional): Number of training epochs. Default is 10.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.
        optimizer (str, optional): Optimizer algorithm. Must be one of 'Adam', 'SGD', 'RMSprop'. Default is 'Adam'.
        momentum (float, optional): Momentum value for SGD and RMSprop optimizers. Default is 0.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Default is 0.
        loss_function (str, optional): Loss function criterion. Must be one of 'BCE', 'Dice', 'Focal', 'DiceCE'. Default is 'BCE'.

    Returns:
        hyperparameters (dict): Dictionary of hyperparameters.

    Raises:
        ValueError: If `loss_function` is not one of 'BCE', 'Dice', 'Focal', 'DiceCE'.
        ValueError: If `optimizer` is not one of 'Adam', 'SGD', 'RMSprop'.

    Example:
        ```python
        import qim3d

        # Set up the model and hyperparameters
        model = qim3d.ml.UNet(size = 'small')

        hyperparameters = qim3d.ml.Hyperparameters(
            model = model,
            n_epochs = 10,
            learning_rate = 5e-3,
            loss_function = 'DiceCE',
            weight_decay  = 1e-3
            )

        # Retrieve the hyperparameters
        parameters_dict = hyperparameters()

        optimizer = params_dict['optimizer']
        criterion = params_dict['criterion']
        n_epochs  = params_dict['n_epochs']
        ```

    """

    def __init__(
        self,
        model: nn.Module,
        n_epochs: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = 'Adam',
        momentum: float = 0,
        weight_decay: float = 0,
        loss_function: str = 'Focal',
    ):
        # TODO: Implement custom loss_functions? Then add a check to see if loss works for segmentation.
        if loss_function not in ['BCE', 'Dice', 'Focal', 'DiceCE']:
            msg = f'Invalid loss function: {loss_function}. Loss criterion must be one of the following: "BCE", "Dice", "Focal", "DiceCE".'
            raise ValueError(msg)

        # TODO: Implement custom optimizer? And add check to see if valid.
        if optimizer not in ['Adam', 'SGD', 'RMSprop']:
            msg = f'Invalid optimizer: {optimizer}. Optimizer must be one of the following: "Adam", "SGD", "RMSprop".'
            raise ValueError(msg)

        if (momentum != 0) and optimizer == 'Adam':
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
        model: nn.Module,
        n_epochs: int,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        loss_function: str,
    ) -> dict:
        optim = self._optimizer(model, optimizer, learning_rate, weight_decay, momentum)
        criterion = self._loss_functions(loss_function)

        hyper_dict = {
            'optimizer': optim,
            'criterion': criterion,
            'n_epochs': n_epochs,
        }
        return hyper_dict

    # Selecting the optimizer
    def _optimizer(
        self,
        model: nn.Module,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
    ) -> torch.optim.Optimizer:
        from torch.optim import SGD, Adam, RMSprop

        if optimizer == 'Adam':
            optim = Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == 'SGD':
            optim = SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == 'RMSprop':
            optim = RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        return optim

    # Selecting the loss function
    def _loss_functions(self, loss_function: str) -> torch.nn:
        from monai.losses import DiceCELoss, DiceLoss, FocalLoss
        from torch.nn import BCEWithLogitsLoss

        if loss_function == 'BCE':
            criterion = BCEWithLogitsLoss(reduction='mean')
        elif loss_function == 'Dice':
            criterion = DiceLoss(sigmoid=True, reduction='mean')
        elif loss_function == 'Focal':
            criterion = FocalLoss(reduction='mean')
        elif loss_function == 'DiceCE':
            criterion = DiceCELoss(sigmoid=True, reduction='mean')
        return criterion
