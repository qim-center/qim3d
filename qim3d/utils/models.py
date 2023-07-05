""" Tools performed with trained models."""
import torch

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
        preds = outputs.cpu().squeeze() > 0.5
    else:
        preds = outputs.cpu().argmax(axis=1)

    # if there is only one image
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        preds = preds.unsqueeze(0)
    
    return inputs,targets,preds