"""Visualization tools"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from qim3d.utils.logger import log


def plot_metrics(
    *metrics,
    linestyle="-",
    batch_linestyle="dotted",
    labels: list = None,
    figsize: tuple = (16, 6),
    show=False
):
    """
    Plots the metrics over epochs and batches.

    Args:
        *metrics: Variable-length argument list of dictionary containing the metrics per epochs and per batches.
        linestyle (str, optional): The style of the epoch metric line. Defaults to '-'.
        batch_linestyle (str, optional): The style of the batch metric line. Defaults to 'dotted'.
        labels (list[str], optional): Labels for the plotted lines. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size (width, height) in inches. Defaults to (16, 6).
        show (bool, optional): If True, displays the plot. Defaults to False.

    Returns:
        fig (matplotlib.figure.Figure): plot with metrics.

    Example:
        train_loss = {'epoch_loss' : [...], 'batch_loss': [...]}
        val_loss = {'epoch_loss' : [...], 'batch_loss': [...]}
        plot_metrics(train_loss,val_loss, labels=['Train','Valid.'])
    """
    import seaborn as snb

    if labels == None:
        labels = [None] * len(metrics)
    elif len(metrics) != len(labels):
        raise ValueError("The number of metrics doesn't match the number of labels.")

    # plotting parameters
    snb.set_style("darkgrid")
    snb.set(font_scale=1.5)
    plt.rcParams["lines.linewidth"] = 2

    fig = plt.figure(figsize=figsize)

    palette = snb.color_palette(None, len(metrics))

    for i, metric in enumerate(metrics):
        metric_name = list(metric.keys())[0]
        epoch_metric = metric[list(metric.keys())[0]]
        batch_metric = metric[list(metric.keys())[1]]

        x_axis = np.linspace(0, len(epoch_metric) - 1, len(batch_metric))

        plt.plot(epoch_metric, linestyle=linestyle, color=palette[i], label=labels[i])
        plt.plot(
            x_axis, batch_metric, linestyle=batch_linestyle, color=palette[i], alpha=0.4
        )

    if labels[0] != None:
        plt.legend()

    plt.ylabel(metric_name)
    plt.xlabel("epoch")

    # reset plotting parameters
    snb.set_style("white")

    if show:
        plt.show()
    plt.close()

    return fig


def grid_overview(
    data, num_images=7, cmap_im="gray", cmap_segm="viridis", alpha=0.5, show=False
):
    """Displays an overview grid of images, labels, and masks (if they exist).

    Labels are the annotated target segmentations
    Masks are applied to the output and target prior to the loss calculation in case of
    sparse labeled data

    Args:
        data (list or torch.utils.data.Dataset): A list of tuples or Torch dataset containing image, label, (and mask data).
        num_images (int, optional): The maximum number of images to display. Defaults to 7.
        cmap_im (str, optional): The colormap to be used for displaying input images. Defaults to 'gray'.
        cmap_segm (str, optional): The colormap to be used for displaying labels. Defaults to 'viridis'.
        alpha (float, optional): The transparency level of the label and mask overlays. Defaults to 0.5.
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.

    Raises:
        ValueError: If the data elements are not tuples.


    Returns:
        fig (matplotlib.figure.Figure): The figure with an overview of the images and their labels.

    Example:
        ```python
        data = [(image1, label1, mask1), (image2, label2, mask2)]
        grid_overview(data, num_images=5, cmap_im='viridis', cmap_segm='hot', alpha=0.8)
        ```

    Notes:
        - If the image data is RGB, the color map is ignored and the user is informed.
        - The number of displayed images is limited to the minimum between `num_images`
            and the length of the data.
        - The grid layout and dimensions vary based on the presence of a mask.
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
    colors_segm = colormaps.get_cmap(cmap_segm)(np.linspace(0, 1, 256))
    colors_segm[:128, 3] = 0
    custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors_segm)

    # Check if data have the right format
    if not isinstance(data[0], tuple):
        raise ValueError("Data elements must be tuples")

    # Define row titles
    row_titles = ["Input images", "Ground truth segmentation", "Mask"]

    # Make new list such that possible augmentations remain identical for all three rows
    plot_data = [data[idx] for idx in range(num_images)]

    fig = plt.figure(
        figsize=(2 * num_images, 9 if has_mask else 6), constrained_layout=True
    )

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

    if show:
        plt.show()
    plt.close()

    return fig


def grid_pred(
    in_targ_preds,
    num_images=7,
    cmap_im="gray",
    cmap_segm="viridis",
    alpha=0.5,
    show=False,
):
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
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.

    Returns:
        fig (matplotlib.figure.Figure): The figure with images, labels and the label prediction from the trained models.

    Raises:
        None

    Example:
        dataset = MySegmentationDataset()
        model = MySegmentationModel()
        in_targ_preds = qim3d.utils.models.inference(dataset,model)
        grid_pred(in_targ_preds, cmap_im='viridis', alpha=0.5)
    """
    import torch

    # Check if dataset have at least specified number of images
    if len(in_targ_preds[0]) < num_images:
        log.warning(
            "Not enough images in the dataset. Changing num_images=%d to num_images=%d",
            num_images,
            len(in_targ_preds[0]),
        )
        num_images = len(in_targ_preds[0])

    # Take only the number of images from in_targ_preds
    inputs, targets, preds = [items[:num_images] for items in in_targ_preds]

    # Adapt segmentation cmap so that background is transparent
    colors_segm = colormaps.get_cmap(cmap_segm)(np.linspace(0, 1, 256))
    colors_segm[:128, 3] = 0
    custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors_segm)

    N = num_images
    H = inputs[0].shape[-2]
    W = inputs[0].shape[-1]

    comp_rgb = torch.zeros((N, 4, H, W))
    comp_rgb[:, 1, :, :] = targets.logical_and(preds)
    comp_rgb[:, 0, :, :] = targets.logical_xor(preds)
    comp_rgb[:, 3, :, :] = targets.logical_or(preds)

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
                ax.imshow(targets[col], cmap=custom_cmap, alpha=alpha)
                ax.axis("off")
            else:
                ax.imshow(inputs[col], cmap=cmap_im)
                ax.imshow(comp_rgb[col].permute(1, 2, 0), alpha=alpha)
                ax.axis("off")

    if show:
        plt.show()
    plt.close()

    return fig


def vol_masked(vol, vol_mask, viz_delta=128):
    """
    Applies masking to a volume based on a binary volume mask.

    This function takes a volume array `vol` and a corresponding binary volume mask `vol_mask`.
    It computes the masked volume where pixels outside the mask are set to the background value,
    and pixels inside the mask are set to foreground.


    Args:
        vol (ndarray): The input volume as a NumPy array.
        vol_mask (ndarray): The binary mask volume as a NumPy array with the same shape as `vol`.
        viz_delta (int, optional): Value added to the volume before applying the mask to visualize masked regions.
            Defaults to 128.

    Returns:
        ndarray: The masked volume with the same shape as `vol`, where pixels outside the mask are set
            to the background value (negative).


    """

    background = (vol.astype("float") + viz_delta) * (1 - vol_mask) * -1
    foreground = (vol.astype("float") + viz_delta) * vol_mask
    vol_masked_result = background + foreground

    return vol_masked_result
