"""Tools performed with models."""

import os

import torch
from torchinfo import ModelStatistics, summary
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from qim3d.utils._logger import log
from qim3d.viz._metrics import plot_metrics

from .models._unet import Hyperparameters


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
    Trains the specified model.

    The function trains the model using the data from the training and validation data loaders, according to the specified hyperparameters.
    Optionally, the final checkpoint of the trained model is saved as a .pth file, the loss curves are plotted, and the loss values are returned.

    Args:
        model (torch.nn.Module): PyTorch model.
        hyperparameters (class): Dictionary with n_epochs, optimizer and criterion.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        checkpoint_directory (str, optional): Directory to save model checkpoint. Default is None.
        eval_every (int, optional): Frequency of model evaluation. Default is every epoch.
        print_every (int, optional): Frequency of log for model performance. Default is every 5 epochs.
        plot (bool, optional): If True, plots the training and validation loss after the model is done training. Default is True.
        return_loss (bool, optional): If True, returns a dictionary with the history of the train and validation losses. Default is False.

    Returns:
        train_loss (dict): Dictionary with average losses and batch losses for training loop. Only returned when `return_loss = True`.
        val_loss (dict): Dictionary with average losses and batch losses for validation loop. Only returned when `return_loss = True`.

    Example:
        ```python
        import qim3d

        base_path = "dataset"
        model = qim3d.ml.models.UNet(size = 'xxsmall')
        augmentation =  qim3d.ml.Augmentation(resize = 'crop', transform_train = 'light')
        hyperparameters = qim3d.ml.Hyperparameters(model, n_epochs = 5)

        # Set up datasets and dataloaders
        train_set, val_set, test_set = qim3d.ml.prepare_datasets(
            path = base_path,
            val_fraction = 0.5,
            model = model,
            augmentation = augmentation
            )

        train_loader, val_loader, test_loader = qim3d.ml.prepare_dataloaders(
            train_set = train_set,
            val_set = val_set,
            test_set = test_set,
            batch_size = 1,
            )

        # Train model
        qim3d.ml.train_model(
            model = model,
            hyperparameters = hyperparameters,
            train_loader = train_loader,
            val_loader = val_loader,
            checkpoint_directory = base_path,
            plot = True)
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
                        f"Epoch {epoch: 3}, train loss: {train_loss['loss'][epoch]:.4f}, "
                        f"val loss: {val_loss['loss'][epoch]:.4f}"
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
    Loads a trained model checkpoint from a .pth file.

    Args:
        model (torch.nn.Module): The PyTorch model to load the checkpoint into.
        checkpoint_path (str): The path to the model checkpoint .pth file.

    Returns:
        model (torch.nn.Module): The model with the loaded checkpoint.

    Example:
        ```python
        import qim3d

        # Instantiate model architecture
        model = qim3d.ml.models.UNet(size = 'small')
        checkpoint_path = "dataset/model_5epochs.pth"

        # Load checkpoint into model
        model = qim3d.ml.load_checkpoint(model, checkpoint_path)
        ```

    """
    model.load_state_dict(torch.load(checkpoint_path))
    log.info(f'Model checkpoint loaded from: {checkpoint_path}')

    return model


def model_summary(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
) -> ModelStatistics:
    """
    Prints the summary of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        dataloader (torch.utils.data.DataLoader): The data loader used to determine the input shape.

    Returns:
        summary (str): Summary of the model architecture.

    Example:
        ```python
        import qim3d

        base_path = "dataset"
        model = qim3d.ml.models.UNet(size = 'small')
        augmentation =  qim3d.ml.Augmentation(resize = 'crop', transform_train = 'light')

        # Set up datasets and dataloaders
        train_set, val_set, test_set = qim3d.ml.prepare_datasets(
            path = base_path,
            val_fraction = 0.5,
            model = model,
            augmentation = augmentation
            )

        train_loader, val_loader, test_loader = qim3d.ml.prepare_dataloaders(
            train_set = train_set,
            val_set = val_set,
            test_set = test_set,
            batch_size = 1,
            )

        # Get model summary
        summary = qim3d.ml.model_summary(model, train_loader)
        print(summary)
        ```

    """
    images, _ = next(iter(dataloader))
    batch_size = tuple(images.shape)
    model_s = summary(model, batch_size, depth=torch.inf)

    return model_s


def test_model(
    model: torch.nn.Module,
    test_set: torch.utils.data.Dataset,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs inference on input data using the specified model.

    The input data should be in the form of a list, where each item is a tuple containing the input image tensor and the corresponding target label tensor.
    The function checks the format and validity of the input data, ensures the model is in evaluation mode,
    and generates predictions using the model. The input images, target labels, and predicted labels are returned as a tuple.

    Args:
        model (torch.nn.Module): The trained model used for predicting segmentations.
        test_set (torch.utils.data.Dataset): A test dataset containing input images and ground truth label data.
        threshold (float): The threshold value used to binarize the model predictions.

    Returns:
        results (list): List of tuples (volume, target, pred) containing the input images, target labels, and predicted labels.

    Raises:
        ValueError: If the data items do not consist of tensors.

    Notes:
        - The function assumes that the model is not already in evaluation mode (`model.eval()`).

    Example:
        ```python
        import qim3d

        base_path = "dataset"
        model = qim3d.ml.models.UNet(size = 'small')
        augmentation =  qim3d.ml.Augmentation(resize = 'crop', transform_train = 'light')
        hyperparameters = qim3d.ml.Hyperparameters(model, n_epochs = 10)

        # Set up datasets and dataloaders
        train_set, val_set, test_set = qim3d.ml.prepare_datasets(
            path = base_path,
            val_fraction = 0.5,
            model = model,
            augmentation = augmentation
            )

        train_loader, val_loader, test_loader = qim3d.ml.prepare_dataloaders(
            train_set = train_set,
            val_set = val_set,
            test_set = test_set,
            batch_size = 1,
            )

        # Train model
        qim3d.ml.train_model(
            model = model,
            hyperparameters = hyperparameters,
            train_loader = train_loader,
            val_loader = val_loader,
            plot = True)

        # Test model
        results = qim3d.ml.test_model(
            model = model,
            test_set = test_set
            )

        # Get the result of the first test image
        volume, target, pred = results[0]
        qim3d.viz.slices_grid(pred, num_slices = 5)
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
