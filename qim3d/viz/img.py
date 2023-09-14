""" Provides a collection of visualization functions."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import torch
import numpy as np
from qim3d.io.logger import log
import qim3d.io

def grid_overview(data, num_images=7, cmap_im="gray", cmap_segm="viridis", alpha=0.5):
    """Displays an overview grid of images, labels, and masks (if they exist).

    Labels are the annotated target segmentations
    Masks are applied to the output and target prior to the loss calculation in case of
    sparse labeled data

    Args:
        data (list or torch.utils.data.Dataset): A list of tuples or Torch dataset containing image,
            label, (and mask data).
        num_images (int, optional): The maximum number of images to display.
            Defaults to 7.
        cmap_im (str, optional): The colormap to be used for displaying input images.
            Defaults to 'gray'.
        cmap_segm (str, optional): The colormap to be used for displaying labels.
            Defaults to 'viridis'.
        alpha (float, optional): The transparency level of the label and mask overlays.
            Defaults to 0.5.

    Raises:
        ValueError: If the data elements are not tuples.

    Notes:
        - If the image data is RGB, the color map is ignored and the user is informed.
        - The number of displayed images is limited to the minimum between `num_images`
            and the length of the data.
        - The grid layout and dimensions vary based on the presence of a mask.

    Returns:
        None

    Example:
        data = [(image1, label1, mask1), (image2, label2, mask2)]
        grid_overview(data, num_images=5, cmap_im='viridis', cmap_segm='hot', alpha=0.8)
    """

    # Check if data has a mask
    has_mask = len(data[0]) > 2 and data[0][-1] is not None

    # Check if image data is RGB and inform the user if it's the case
    if len(data[0][0].squeeze().shape) > 2:
        log.info("Input images are RGB: color map is ignored")

    # Check if dataset have at least specified number of images
    if len(data) < num_images:
        log.warning(
            "Not enough images in the dataset. Changing num_images=%d to num_images=%d",
            num_images,
            len(data),
        )
        num_images = len(data)

    # Adapt segmentation cmap so that background is transparent
    colors_segm = cm.get_cmap(cmap_segm)(np.linspace(0, 1, 256))
    colors_segm[:128, 3] = 0
    custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors_segm)

    # Check if data have the right format
    if not isinstance(data[0], tuple):
        raise ValueError("Data elements must be tuples")

    # Define row titles
    row_titles = ["Input images", "Ground truth segmentation", "Mask"]

    # Make new list such that possible augmentations remain identical for all three rows
    plot_data = [data[idx] for idx in range(num_images)]

    fig = plt.figure(figsize=(2 * num_images, 9 if has_mask else 6), constrained_layout=True)

    # create 2 (3) x 1 subfigs
    subfigs = fig.subfigures(nrows=3 if has_mask else 2, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(row_titles[row], fontsize=22)

        # create 1 x num_images subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=num_images)
        for col, ax in enumerate(np.atleast_1d(axs)):
            if row in [1, 2]:  # Ground truth segmentation and mask
                ax.imshow(plot_data[col][0].squeeze(), cmap=cmap_im)
                ax.imshow(plot_data[col][row].squeeze(), cmap=custom_cmap, alpha=alpha)
                ax.axis("off")
            else:
                ax.imshow(plot_data[col][row].squeeze(), cmap=cmap_im)
                ax.axis("off")
    fig.show()


def grid_pred(in_targ_preds, num_images=7, cmap_im="gray", cmap_segm="viridis", alpha=0.5):
    """Displays a grid of input images, predicted segmentations, ground truth segmentations, and their comparison.

    Displays a grid of subplots representing different aspects of the input images and segmentations.
    The grid includes the following rows:
        - Row 1: Input images
        - Row 2: Predicted segmentations overlaying input images
        - Row 3: Ground truth segmentations overlaying input images
        - Row 4: Comparison between true and predicted segmentations overlaying input images

    Each row consists of `num_images` subplots, where each subplot corresponds to an image from the dataset.
    The function utilizes various color maps for visualization and applies transparency to the segmentations.

    Args:
        in_targ_preds (tuple): A tuple containing input images, target segmentations, and predicted segmentations.
        num_images (int, optional): Number of images to display. Defaults to 7.
        cmap_im (str, optional): Color map for input images. Defaults to "gray".
        cmap_segm (str, optional): Color map for segmentations. Defaults to "viridis".
        alpha (float, optional): Alpha value for transparency. Defaults to 0.5.

    Returns:
        None

    Raises:
        None

    Example:
        dataset = MySegmentationDataset()
        model = MySegmentationModel()
        in_targ_preds = qim3d.utils.models.inference(dataset,model)
        grid_pred(in_targ_preds, cmap_im='viridis', alpha=0.5)        
    """
    
    # Check if dataset have at least specified number of images
    if len(in_targ_preds[0]) < num_images:
        log.warning(
            "Not enough images in the dataset. Changing num_images=%d to num_images=%d",
            num_images,
            len(in_targ_preds[0]),
        )
        num_images = len(in_targ_preds[0])

    # Take only the number of images from in_targ_preds
    inputs,targets,preds = [items[:num_images] for items in in_targ_preds]
    
    # Adapt segmentation cmap so that background is transparent
    colors_segm = cm.get_cmap(cmap_segm)(np.linspace(0, 1, 256))
    colors_segm[:128, 3] = 0
    custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors_segm)
    
    N = num_images
    H = inputs[0].shape[-2]
    W = inputs[0].shape[-1]

    comp_rgb = torch.zeros((N,4,H,W))
    comp_rgb[:,1,:,:] = targets.logical_and(preds)
    comp_rgb[:,0,:,:] = targets.logical_xor(preds)
    comp_rgb[:,3,:,:] = targets.logical_or(preds)

    row_titles = [
        "Input images",
        "Predicted segmentation",
        "Ground truth segmentation",
        "True vs. predicted segmentation",
    ]

    fig = plt.figure(figsize=(2 * num_images, 10), constrained_layout=True)

    # create 3 x 1 subfigs
    subfigs = fig.subfigures(nrows=4, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(row_titles[row], fontsize=22)

        # create 1 x num_images subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=num_images)
        for col, ax in enumerate(np.atleast_1d(axs)):
            if row == 0:
                ax.imshow(inputs[col], cmap=cmap_im)
                ax.axis("off")

            elif row == 1:  # Predicted segmentation
                ax.imshow(inputs[col], cmap=cmap_im)
                ax.imshow(preds[col], cmap=custom_cmap, alpha=alpha)
                ax.axis("off")
            elif row == 2:  # Ground truth segmentation
                ax.imshow(inputs[col], cmap=cmap_im)
                ax.imshow(
                    targets[col], cmap=custom_cmap, alpha=alpha
                )
                ax.axis("off")
            else:
                ax.imshow(inputs[col], cmap=cmap_im)
                ax.imshow(comp_rgb[col].permute(1, 2, 0), alpha=alpha)
                ax.axis("off")

    fig.show()

def slice_viz(input, position = 'mid', cmap="viridis", axis=False, img_height=2, img_width=2):
    """ Displays one or several slices from a 3d array.

    Args:
        input (str, numpy.ndarray): Path to the file or 3-dimensional array.
        position (str, int, list, array, optional): One or several slicing levels.
        cmap (str, optional): Specifies the color map for the image.
        axis (bool, optional): Specifies whether the axes should be included.

    Raises:
        ValueError: If provided string for 'position' argument is not valid (not upper, middle or bottom).

    Usage:
        image_path = '/my_image_path/my_image.tif'
        slice_viz(image_path)
    """
    
    # Filepath input
    if isinstance(input,str):
        vol = qim3d.io.load(input) # Function has its own ValueErrors
        dim = vol.ndim
        
        if dim == 3:
            pass
        else:
            raise ValueError(f"Given array is not a volume! Current dimension: {dim}")
            
        
    # Numpy array input
    elif isinstance(input,np.ndarray):
        dim = input.ndim
        
        if dim == 3:
            vol = input
        else:
            raise ValueError(f"Given array is not a volume! Current dimension: {dim}")

    # Position is a string
    if isinstance(position,str):
        if position.lower() in ['mid','middle']:
            height = [int(vol.shape[-1]/2)]
        elif position.lower() in ['top','upper', 'start']:
            height = [0]
        elif position.lower() in ['bot','bottom', 'end']:
            height = [vol.shape[-1]-1]
        else:
            raise ValueError('Position not recognized. Choose an integer, list, array or "start","mid","end".')
    
    # Position is an integer
    elif isinstance(position,int):
        height = [position]

    # Position is a list or array of integers
    elif isinstance(position,(list,np.ndarray)):
        height = position

    num_images = len(height)
    
    fig = plt.figure(figsize=(img_width * num_images, img_height), constrained_layout = True)
    axs = fig.subplots(nrows = 1, ncols = num_images)
    
    for col, ax in enumerate(np.atleast_1d(axs)):
        ax.imshow(vol[:,:,height[col]],cmap = cmap)
        ax.set_title(f'Slice {height[col]}', fontsize=8)
        if not axis:
            ax.axis('off')
    
    fig.show()