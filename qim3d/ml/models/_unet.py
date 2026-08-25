"""UNet model and Hyperparameters class."""

from qim3d.utils import log
from qim3d.utils._dependencies import optional_import

torch = optional_import('torch', extra='deep-learning')


class UNet(torch.nn.Module):
    """
    Constructs a 3D U-Net model for volumetric image segmentation.

    The U-Net architecture consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) that enables precise localization. This implementation wraps the [MONAI U-Net](https://docs.monai.io/en/stable/networks.html#unet) and provides simplified presets for model depth and width via the `size` argument.

    **Model Presets (Channels per Layer):**

    * **'xxsmall'**: (4, 8) - *Ultra-lightweight (2 layers)*
    * **'xsmall'**: (16, 32) - *Lightweight (2 layers)*
    * **'small'**: (32, 64, 128) - *Fast training (3 layers)*
    * **'medium'**: (64, 128, 256) - *Balanced performance (3 layers, default)*
    * **'large'**: (64, 128, 256, 512, 1024) - *High capacity (5 layers)*
    * **'xlarge'**: (64, 128, 256, 512, 1024, 2048) - *Very high capacity (6 layers)*
    * **'xxlarge'**: (64, 128, 256, 512, 1024, 2048, 4096) - *Maximum capacity (7 layers)*

    Args:
        size (str, optional): The complexity of the model. Must be one of 'xxsmall', 'xsmall', 'small', 'medium', 'large', 'xlarge', or 'xxlarge'. Defaults to 'medium'.
        dropout (float, optional): The dropout rate (0 to 1) applied to hidden layers to prevent overfitting. Defaults to 0.
        kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
        up_kernel_size (int, optional): The size of the up-convolution kernel. Defaults to 3.
        activation (str, optional): The activation function to use (e.g., 'RELU', 'PReLU', 'Sigmoid'). Defaults to 'PReLU'.
        bias (bool, optional): Whether to include bias terms in convolutions. Defaults to `True`.
        adn_order (str, optional): The ordering of Activation (A), Dropout (D), and Normalization (N) blocks. Defaults to 'NDA'.

    Returns:
        model (torch.nn.Module): The initialized 3D U-Net model.

    Raises:
        ValueError: If `size` is not a valid preset string.

    Example:
        ```python
        import qim3d

        # Initialize a small U-Net for quick experiments
        model = qim3d.ml.models.UNet(size='small', dropout=0.2)

        print(model)
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

    def _model_choice(self) -> torch.nn.Module:
        monai = optional_import('monai', extra='deep-learning')

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

        model = monai.networks.nets.UNet(
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
    Configuration wrapper for training parameters (Optimizer, Loss, Epochs).

    This class centralizes the setup of the training loop. It automatically instantiates the PyTorch optimizer and loss function based on string arguments, ensuring valid combinations and settings.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained (required to register parameters with the optimizer).
        n_epochs (int, optional): The number of complete passes through the training dataset. Defaults to 10.
        learning_rate (float, optional): The step size for the optimizer. Defaults to 1e-3.
        optimizer (str, optional): The optimization algorithm. Options: 'Adam', 'SGD', 'RMSprop'. Defaults to 'Adam'.
        momentum (float, optional): The momentum factor (only used for 'SGD' and 'RMSprop'). Accelerates gradient vectors in the right directions, leading to faster converging. Defaults to 0.
        weight_decay (float, optional): L2 penalty applied to the weights to prevent overfitting. Defaults to 0.
        loss_function (str, optional): The objective function to minimize. Options:
            * 'BCE': Binary Cross Entropy (with Logits).
            * 'Dice': Dice Loss (good for class imbalance).
            * 'Focal': Focal Loss (focuses on hard examples).
            * 'DiceCE': Weighted sum of Dice and Cross Entropy.
            Defaults to 'Focal'.

    Returns:
        hyperparameters (dict):
            A dictionary containing the initialized objects, accessible via the `()` operator:
            * 'optimizer': The torch.optim object.
            * 'criterion': The loss function module.
            * 'n_epochs': The integer number of epochs.

    Raises:
        ValueError: If `loss_function` or `optimizer` are not among the supported options.

    Example:
        ```python
        import qim3d

        # 1. Initialize model
        model = qim3d.ml.models.UNet(size='small')

        # 2. Define training configuration
        hyperparameters = qim3d.ml.Hyperparameters(
            model=model,
            n_epochs=10,
            learning_rate=5e-3,
            optimizer='Adam',
            loss_function='DiceCE'
        )

        # 3. Retrieve initialized objects for the training loop
        params_dict = hyperparameters()

        optimizer = params_dict['optimizer']
        criterion = params_dict['criterion']
        print(f"Ready to train for {params_dict['n_epochs']} epochs with {optimizer.__class__.__name__}")
        ```
    """

    def __init__(
        self,
        model: torch.nn.Module,
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
        model: torch.nn.Module,
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
        model: torch.nn.Module,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
    ) -> torch.optim.Optimizer:
        torch = optional_import('torch', extra='deep-learning')

        if optimizer == 'Adam':
            optim = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == 'SGD':
            optim = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == 'RMSprop':
            optim = torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        return optim

    # Selecting the loss function
    def _loss_functions(self, loss_function: str) -> torch.nn:
        monai = optional_import('monai', extra='deep-learning')
        torch = optional_import('torch', extra='deep-learning')

        if loss_function == 'BCE':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif loss_function == 'Dice':
            criterion = monai.losses.DiceLoss(sigmoid=True, reduction='mean')
        elif loss_function == 'Focal':
            criterion = monai.losses.FocalLoss(reduction='mean')
        elif loss_function == 'DiceCE':
            criterion = monai.losses.DiceCELoss(sigmoid=True, reduction='mean')
        return criterion
