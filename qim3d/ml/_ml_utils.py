""" Tools performed with models."""

import torch
import numpy as np

from torchinfo import summary, ModelStatistics
from qim3d.utils._logger import log
from qim3d.viz._metrics import plot_metrics

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from .models._unet import Hyperparameters

def train_model(
    model: torch.nn.Module,
    hyperparameters: Hyperparameters,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    eval_every: int = 1,
    print_every: int = 5,
    plot: bool = True,
    return_loss: bool = False,
) -> tuple[tuple[float], tuple[float]]:
    """Function for training Neural Network models.

    Args:
        model (torch.nn.Module): PyTorch model.
        hyperparameters (class): Dictionary with n_epochs, optimizer and criterion.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        eval_every (int, optional): Frequency of model evaluation. Defaults to every epoch.
        print_every (int, optional): Frequency of log for model performance. Defaults to every 5 epochs.
        plot (bool, optional): If True, plots the training and validation loss after the model is done training.
        return_loss (bool, optional), If True, returns a dictionary with the history of the train and validation losses.

    Returns:
        if return_loss = True:
            tuple:
                train_loss (dict): Dictionary with average losses and batch losses for training loop.
                val_loss (dict): Dictionary with average losses and batch losses for validation loop.

    Example:
        import qim3d
        from qim3d.ml import train_model

        # defining the model.
        model = qim3d.ml.UNet()

        # choosing the hyperparameters
        hyperparameters = qim3d.ml.models.Hyperparameters(model)

        # DataLoaders
        train_loader = MyTrainLoader()
        val_loader = MyValLoader()

        # training the model.
        train_loss,val_loss = qim3d.ml.train_model(model, hyperparameters, train_loader, val_loader)
    """
    params_dict = hyperparameters()
    n_epochs = params_dict["n_epochs"]
    optimizer = params_dict["optimizer"]
    criterion = params_dict["criterion"]

    # Choosing best device available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Avoid logging twice.
    log.propagate = False

    train_loss = {"loss": [], "batch_loss": []}
    val_loss = {"loss": [], "batch_loss": []}
    with logging_redirect_tqdm():
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            step = 0

            model.train()

            for data in train_loader:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                step += 1

                # Log and store batch training loss.
                train_loss["batch_loss"].append(loss.detach().item())

            # Log and store average training loss per epoch.
            epoch_loss = epoch_loss / step
            train_loss["loss"].append(epoch_loss)

            if epoch % eval_every == 0:
                eval_loss = 0
                step = 0

                model.eval()

                for data in val_loader:
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device).unsqueeze(1)

                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    eval_loss += loss.item()
                    step += 1

                    # Log and store batch validation loss.
                    val_loss["batch_loss"].append(loss.item())

                # Log and store average validation loss.
                eval_loss = eval_loss / step
                val_loss["loss"].append(eval_loss)

                if epoch % print_every == 0:
                    log.info(
                        f"Epoch {epoch: 3}, train loss: {train_loss['loss'][epoch]:.4f}, "
                        f"val loss: {val_loss['loss'][epoch]:.4f}"
                    )

    if plot:
        plot_metrics(train_loss, val_loss, labels=["Train", "Valid."], show=True)

    if return_loss:
        return train_loss, val_loss


def model_summary(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module) -> ModelStatistics:
    """Prints the summary of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        dataloader (torch.utils.data.DataLoader): The data loader used to determine the input shape.

    Returns:
        str: Summary of the model architecture.

    Example:
        model = MyModel()
        dataloader = DataLoader(dataset, batch_size=32)
        summary = model_summary(model, dataloader)
        print(summary)
    """
    images, _ = next(iter(dataloader))
    batch_size = tuple(images.shape)
    model_s = summary(model, batch_size, depth=torch.inf)

    return model_s


def inference(data: torch.utils.data.Dataset, model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs inference on input data using the specified model.

    Performs inference on the input data using the provided model. The input data should be in the form of a list,
    where each item is a tuple containing the input image tensor and the corresponding target label tensor.

    The function checks the format and validity of the input data, ensures the model is in evaluation mode,
    and generates predictions using the model. The input images, target labels, and predicted labels are returned
    as a tuple.

    Args:
        data (torch.utils.data.Dataset): A Torch dataset containing input image and
            ground truth label data.
        model (torch.nn.Module): The trained network model used for predicting segmentations.

    Returns:
        tuple: A tuple containing the input images, target labels, and predicted labels.

    Raises:
        ValueError: If the data items are not tuples or data items do not consist of tensors.
        ValueError: If the input image is not in (C, H, W) format.

    Notes:
        - The function does not assume the model is already in evaluation mode (model.eval()).

    Example:
        import qim3d
        dataset = MySegmentationDataset()
        model = MySegmentationModel()
        qim3d.ml.inference(data,model)
    """

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if data have the right format
    if not isinstance(data[0], tuple):
        raise ValueError("Data items must be tuples")

    # Check if data is torch tensors
    for element in data[0]:
        if not isinstance(element, torch.Tensor):
            raise ValueError("Data items must consist of tensors")

    # Check if input image is (C,H,W) format
    if data[0][0].dim() == 3 and (data[0][0].shape[0] in [1, 3]):
        pass
    else:
        raise ValueError("Input image must be (C,H,W) format")

    model.to(device)
    model.eval()

    # Make new list such that possible augmentations remain identical for all three rows
    plot_data = [data[idx] for idx in range(len(data))]

    # Create input and target batch
    inputs = torch.stack([item[0] for item in plot_data], dim=0).to(device)
    targets = torch.stack([item[1] for item in plot_data], dim=0)

    # Get output predictions
    with torch.no_grad():
        outputs = model(inputs)

    # Prepare data for plotting
    inputs = inputs.cpu().squeeze()
    targets = targets.squeeze()
    if outputs.shape[1] == 1:
        preds = (
            outputs.cpu().squeeze() > 0.5
        )  # TODO: outputs from model are not between [0,1] yet, need to implement that
    else:
        preds = outputs.cpu().argmax(axis=1)

    # if there is only one image
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        preds = preds.unsqueeze(0)

    return inputs, targets, preds


def volume_inference(volume: np.ndarray, model: torch.nn.Module, threshold:float = 0.5) -> np.ndarray:
    """
    Compute on the entire volume
    Args:
        volume (numpy.ndarray): A 3D numpy array representing the input volume.
        model (torch.nn.Module): The trained network model used for inference.
        threshold (float): The threshold value used to binarize the model predictions.
    Returns:
        numpy.ndarray: A 3D numpy array representing the model predictions for each slice of the input volume.
    Raises:
        ValueError: If the input volume is not a 3D numpy array.
    """
    if len(volume.shape) != 3:
        raise ValueError("Input volume must be a 3D numpy array")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    inference_vol = np.zeros_like(volume)

    for idx in np.arange(len(volume)):
        input_with_channel = np.expand_dims(volume[idx], axis=0)
        input_tensor = torch.tensor(input_with_channel, dtype=torch.float32).to(device)
        input_tensor = input_tensor.unsqueeze(0)
        output = model(input_tensor) > threshold
        output = output.cpu() if device == "cuda" else output
        output_detached = output.detach()
        output_numpy = output_detached.numpy()[0, 0, :, :]
        inference_vol[idx] = output_numpy

    return inference_vol
