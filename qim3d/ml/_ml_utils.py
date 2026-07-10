"""Tools performed with models."""

import os

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from qim3d.utils._dependencies import optional_import
from qim3d.utils._logger import log
from qim3d.viz._metrics import plot_metrics

from .models._unet import Hyperparameters

torch = optional_import('torch', extra='deep-learning')
torchinfo = optional_import('torchinfo', extra='deep-learning')


def train_model(
    model: torch.nn.Module,
    hyperparameters: Hyperparameters,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    checkpoint_directory: str = None,
    eval_every: int = 1,
    print_every: int = 1,
    plot: bool = True,
    return_loss: bool = False,
) -> tuple[tuple[float], tuple[float]]:
    """
    Executes the training loop for a PyTorch model.

    This function manages the iterative process of training. It handles:

    1.  **Training Steps**: Iterating through the training data, computing gradients (backpropagation), and updating model weights.
    2.  **Validation**: Periodically evaluating the model on unseen data to monitor for overfitting.
    3.  **Logging**: Printing loss values to track convergence.
    4.  **Checkpointing**: Saving the final model weights to disk.
    5.  **Visualization**: Plotting training and validation loss curves.

    The function automatically detects if a GPU (CUDA) is available and moves the model and data to the appropriate device.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        hyperparameters (Hyperparameters): A `qim3d.ml.Hyperparameters` object containing the optimizer, loss function, and epoch count.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation set.
        checkpoint_directory (str, optional): The directory where the final model weights (`.pth` file) will be saved. If `None`, the model is not saved to disk. Defaults to `None`.
        eval_every (int, optional): The number of epochs between validation runs. Defaults to 1 (validate every epoch).
        print_every (int, optional): The number of epochs between log updates. Defaults to 1 (log every epoch).
        plot (bool, optional): If `True`, displays a plot of the loss history after training finishes. Defaults to `True`.
        return_loss (bool, optional): If `True`, returns the history of loss values. Defaults to `False`.

    Returns:
        (train_loss, val_loss) (tuple[dict, dict] | None):
            Only returned if `return_loss` is `True`.
            * **train_loss**: A dictionary containing 'loss' (per epoch) and 'batch_loss' (per iteration).
            * **val_loss**: A dictionary containing 'loss' and 'batch_loss' for the validation set.

    Example:
        ```python
        import qim3d

        # 1. Setup components
        base_path = "dataset"
        model = qim3d.ml.models.UNet(size='xxsmall')
        hyperparameters = qim3d.ml.Hyperparameters(model, n_epochs=5)
        augmentation = qim3d.ml.Augmentation(resize='crop', transform_train='light')

        # 2. Prepare Data
        train_set, val_set, test_set = qim3d.ml.prepare_datasets(
            path=base_path,
            val_fraction=0.5,
            model=model,
            augmentation=augmentation
        )

        train_loader, val_loader, test_loader = qim3d.ml.prepare_dataloaders(
            train_set, val_set, test_set, batch_size=1
        )

        # 3. Train
        qim3d.ml.train_model(
            model=model,
            hyperparameters=hyperparameters,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_directory=base_path,
            plot=True
        )
        ```

    """
    # Get hyperparameters
    params_dict = hyperparameters()

    n_epochs = params_dict['n_epochs']
    optimizer = params_dict['optimizer']
    criterion = params_dict['criterion']

    # Choosing best device available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # Avoid logging twice
    log.propagate = False

    # Set up dictionaries to store training and validation losses
    train_loss = {'loss': [], 'batch_loss': []}
    val_loss = {'loss': [], 'batch_loss': []}

    with logging_redirect_tqdm():
        for epoch in tqdm(range(n_epochs), desc='Training epochs', unit='epoch'):
            epoch_loss = 0
            step = 0

            model.train()

            for data in train_loader:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                step += 1

                # Log and store batch training loss
                train_loss['batch_loss'].append(loss.detach().item())

            # Log and store average training loss per epoch
            epoch_loss = epoch_loss / step
            train_loss['loss'].append(epoch_loss)

            if epoch % eval_every == 0:
                eval_loss = 0
                step = 0

                model.eval()

                for data in val_loader:
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    eval_loss += loss.item()
                    step += 1

                    # Log and store batch validation loss
                    val_loss['batch_loss'].append(loss.item())

                # Log and store average validation loss
                eval_loss = eval_loss / step
                val_loss['loss'].append(eval_loss)

                if epoch % print_every == 0:
                    log.info(
                        f'Epoch {epoch: 3}, train loss: {train_loss["loss"][epoch]:.4f}, '
                        f'val loss: {val_loss["loss"][epoch]:.4f}'
                    )

    if checkpoint_directory:
        checkpoint_filename = f'model_{n_epochs}epochs.pth'
        checkpoint_path = os.path.join(checkpoint_directory, checkpoint_filename)

        # Save model checkpoint to .pth file
        torch.save(model.state_dict(), checkpoint_path)
        log.info(f'Model checkpoint saved at: {checkpoint_path}')

    if plot:
        plot_metrics(train_loss, val_loss, labels=['Train', 'Valid.'], show=True)

    if return_loss:
        return train_loss, val_loss


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Restores a model's state (weights and biases) from a saved checkpoint file.

    This function loads a dictionary of learned parameters from a `.pth` file and applies them to the provided model architecture. This is essential for:

    * **Inference**: Using a pre-trained model to make predictions on new data.
    * **Resuming Training**: Continuing the training process from a specific point.
    * **Transfer Learning**: Fine-tuning a pre-trained model on a new task.

    **Important:** The architecture of the `model` object must match the architecture used when the checkpoint was saved. If the shapes of the layers do not align, a runtime error will occur.

    Args:
        model (torch.nn.Module): The initialized PyTorch model architecture (e.g., a `UNet` instance).
        checkpoint_path (str): The file path to the `.pth` checkpoint.

    Returns:
        model (torch.nn.Module):
            The model with its weights updated from the file.

    Example:
        ```python
        import qim3d

        # 1. Define the architecture (must match the saved model)
        model = qim3d.ml.models.UNet(size='small')

        # 2. Path to the saved weights
        checkpoint_path = "dataset/model_5epochs.pth"

        # 3. Load the weights
        model = qim3d.ml.load_checkpoint(model, checkpoint_path)

        # The model is now ready for inference
        print("Checkpoint loaded successfully.")
        ```

    """
    model.load_state_dict(torch.load(checkpoint_path))
    log.info(f'Model checkpoint loaded from: {checkpoint_path}')

    return model


def model_summary(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
) -> torchinfo.ModelStatistics:
    """
    Generates a detailed summary of the model's architecture and parameter count.

    This function provides a comprehensive overview of the model, including the output shape of each layer, the number of trainable parameters, and the estimated memory usage. It automatically infers the input dimensions by sampling a single batch from the provided DataLoader.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
        dataloader (torch.utils.data.DataLoader): A DataLoader used to retrieve a sample batch for input shape inference.

    Returns:
        model_s (torchinfo.ModelStatistics):
            An object containing the model statistics. When printed, it displays a formatted table of layers and parameters.

    Example:
        ```python
        import qim3d

        # Define model and data components
        model = qim3d.ml.models.UNet(size='small')

        # ... (assume train_loader is already prepared) ...

        # Print model summary
        summary = qim3d.ml.model_summary(model, train_loader)
        print(summary)
        ```

    """
    images, _ = next(iter(dataloader))
    batch_size = tuple(images.shape)
    model_s = torchinfo.summary(model, batch_size, depth=torch.inf)

    return model_s


def test_model(
    model: torch.nn.Module,
    test_set: torch.utils.data.Dataset,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs inference on a test dataset to generate segmentation predictions.

    This function iterates through the provided `test_set`, applies the trained `model`, and post-processes the output. It automatically handles:

    1.  **Device Management**: Moves data to GPU if available.
    2.  **Batching**: Adds necessary batch dimensions for the model input.
    3.  **Activation**: Applies a Sigmoid function to convert raw model outputs (logits) into probabilities.
    4.  **Thresholding**: Converts probabilities into binary masks using the specified `threshold`.
    5.  **Format Conversion**: Returns inputs, targets, and predictions as NumPy arrays for easy analysis or visualization.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        test_set (torch.utils.data.Dataset): The dataset containing (image, label) pairs to evaluate.
        threshold (float, optional): The probability threshold for binary classification. Pixels with a probability higher than this value are classified as foreground (1). Defaults to 0.5.

    Returns:
        results (list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
            A list of tuples, where each tuple corresponds to one sample in the test set and contains:
            * **volume**: The original input image.
            * **target**: The ground truth label.
            * **pred**: The predicted binary segmentation mask.

    Raises:
        ValueError: If the items yielded by `test_set` are not PyTorch tensors.

    Example:
        ```python
        import qim3d
        import matplotlib.pyplot as plt

        # ... (Assume model and test_set are already prepared) ...

        # Run inference
        results = qim3d.ml.test_model(model=model, test_set=test_set)

        # Visualize the first result
        vol, target, pred = results[0]

        # Display the middle slice of the prediction
        mid_slice = pred.shape[0] // 2
        plt.imshow(pred[mid_slice], cmap='gray')
        plt.title("Predicted Segmentation")
        plt.show()
        ```

    """
    # Set model to evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # List to store results
    results = []

    for volume, target in test_set:
        if not isinstance(volume, torch.Tensor) or not isinstance(target, torch.Tensor):
            msg = 'Data items must consist of tensors'
            raise ValueError(msg)

        # Add batch and channel dimensions
        volume = volume.unsqueeze(0).to(device)  # Shape: [1, 1, D, H, W]
        target = target.unsqueeze(0).to(device)  # Shape: [1, 1, D, H, W]

        with torch.no_grad():
            # Get model predictions (logits)
            output = model(volume)

            # Convert logits to probabilities [0, 1]
            pred = torch.sigmoid(output)

            # Convert to binary mask by thresholding the probabilities
            pred = (pred > threshold).float()

            # Remove batch and channel dimensions
            volume = volume.squeeze().cpu().numpy()
            target = target.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()

        # TODO: Compute DICE score between target and prediction?

        # Append results to list
        results.append((volume, target, pred))

    return results
