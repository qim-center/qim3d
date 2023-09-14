""" Tools performed with models."""
import torch
import matplotlib.pyplot as plt

from torchinfo import summary
from qim3d.io.logger import log, level
from qim3d.viz.visualizations import plot_metrics

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def train_model(model, hyperparameters, train_loader, val_loader, eval_every = 1, print_every = 5, plot = True):
    """ Function for training Neural Network models.
    
    Args:
        model (torch.nn.Module): PyTorch model.
        hyperparameters (class): dictionary with n_epochs, optimizer and criterion.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        eval_every (int, optional): frequency of model evaluation. Defaults to every epoch.
        print_every (int, optional): frequency of log for model performance. Defaults to every 5 epochs.

    Returns:
        tuple:
            train_loss (dict): dictionary with average losses and batch losses for training loop.
            val_loss (dict): dictionary with average losses and batch losses for validation loop.
        
    Example:
        # defining the model.
        model = qim3d.utils.UNet()
        
        # choosing the hyperparameters
        hyperparameters = qim3d.utils.hyperparameters(model)

        # DataLoaders
        train_loader = MyTrainLoader()
        val_loader = MyValLoader()

        # training the model.
        train_loss,val_loss = train_model(model, hyperparameters, train_loader, val_loader)
    """
    params_dict = hyperparameters()
    n_epochs = params_dict['n_epochs']
    optimizer = params_dict['optimizer']
    criterion = params_dict['criterion']

    # Choosing best device available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    
    # Avoid logging twice.
    log.propagate = False
    
    train_loss = {'loss' : [],'batch_loss': []}
    val_loss = {'loss' : [], 'batch_loss' : []}
    with logging_redirect_tqdm():
        for epoch in tqdm(range(n_epochs)): 
            epoch_loss = 0
            step = 0
    
            model.train()
            
            for data in train_loader:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device).type(torch.cuda.FloatTensor).unsqueeze(1)
    
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
    
                # Backpropagation
                loss.backward()
                optimizer.step()
    
                epoch_loss += loss.detach().item()
                step += 1

                # Log and store batch training loss.
                train_loss['batch_loss'].append(loss.detach().item())
    
            # Log and store average training loss per epoch.
            epoch_loss = epoch_loss / step
            train_loss['loss'].append(epoch_loss)
        
            if epoch % eval_every ==0:
                eval_loss = 0
                step = 0
                
                model.eval()
        
                for data in val_loader:
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device).type(torch.cuda.FloatTensor).unsqueeze(1)
                
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    eval_loss += loss.item()
                    step += 1

                    # Log and store batch validation loss.
                    val_loss['batch_loss'].append(loss.item())
                
                # Log and store average validation loss.
                eval_loss = eval_loss / step
                val_loss['loss'].append(eval_loss)
                
                if epoch % print_every == 0:
                    log.info(
                        f"Epoch {epoch: 3}, train loss: {train_loss['loss'][epoch]:.4f}, "
                        f"val loss: {val_loss['loss'][epoch]:.4f}"
                    )

    if plot:
        fig = plt.figure(figsize=(16, 6), constrained_layout = True)
        plot_metrics(train_loss, label = 'Train')
        plot_metrics(val_loss,color = 'orange', label = 'Valid.')
        fig.show()


def model_summary(dataloader,model):
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    images,_ = next(iter(dataloader)) 
    batch_size = tuple(images.shape)
    model_s = summary(model,batch_size,depth = torch.inf)
    
    return model_s


def inference(data,model):
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
        dataset = MySegmentationDataset()
        model = MySegmentationModel()
        inference(data,model)
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
        preds = outputs.cpu().squeeze() > 0.5 # TODO: outputs from model are not between [0,1] yet, need to implement that
    else:
        preds = outputs.cpu().argmax(axis=1)

    # if there is only one image
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        preds = preds.unsqueeze(0)
    
    return inputs,targets,preds