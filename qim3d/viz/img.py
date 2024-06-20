""" 
Provides a collection of visualization functions.
"""

import math
from typing import List, Optional, Union

import dask.array as da
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

import qim3d
from qim3d.io.logger import log


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


def slices(
    vol: Union[np.ndarray, torch.Tensor],
    axis: int = 0,
    position: Optional[Union[str, int, List[int]]] = None,
    n_slices: int = 5,
    max_cols: int = 5,
    cmap: str = "viridis",
    img_height: int = 2,
    img_width: int = 2,
    show: bool = False,
    show_position: bool = True,
    interpolation: Optional[str] = "none",
    img_size = None,
    **imshow_kwargs,
) -> plt.Figure:
    """Displays one or several slices from a 3d volume.

    By default if `position` is None, slices plots `n_slices` linearly spaced slices.
    If `position` is given as a string or integer, slices will plot an overview with `n_slices` figures around that position.
    If `position` is given as a list, `n_slices` will be ignored and the slices from `position` will be plotted.

    Args:
        vol (np.ndarray or torch.Tensor): The 3D volume to be sliced.
        axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        position (str, int, list, optional): One or several slicing levels. If None, linearly spaced slices will be displayed. Defaults to None.
        n_slices (int, optional): Defines how many slices the user wants to be displayed. Defaults to 5.
        max_cols (int, optional): The maximum number of columns to be plotted. Defaults to 5.
        cmap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        img_height(int, optional): Height of the figure.
        img_width(int, optional): Width of the figure.
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.
        show_position (bool, optional): If True, displays the position of the slices. Defaults to True.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        fig (matplotlib.figure.Figure): The figure with the slices from the 3d array.

    Raises:
        ValueError: If the input is not a numpy.ndarray or torch.Tensor.
        ValueError: If the axis to slice along is not a valid choice, i.e. not an integer between 0 and the number of dimensions of the volume minus 1.
        ValueError: If the file or array is not a volume with at least 3 dimensions.
        ValueError: If the `position` keyword argument is not a integer, list of integers or one of the following strings: "start", "mid" or "end".

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.shell_225x128x128
        qim3d.viz.slices(vol, n_slices=15)
        ```
        ![Grid of slices](assets/screenshots/viz-slices.png)
    """
    if img_size:
        img_height = img_size
        img_width = img_size

    # Numpy array or Torch tensor input
    if not isinstance(vol, (np.ndarray, torch.Tensor, da.core.Array)):
        raise ValueError("Data type not supported")

    if vol.ndim < 3:
        raise ValueError(
            "The provided object is not a volume as it has less than 3 dimensions."
        )

    if isinstance(vol, da.core.Array):
        vol = vol.compute()
        
    # Ensure axis is a valid choice
    if not (0 <= axis < vol.ndim):
        raise ValueError(
            f"Invalid value for 'axis'. It should be an integer between 0 and {vol.ndim - 1}."
        )

    # Get total number of slices in the specified dimension
    n_total = vol.shape[axis]

    # Position is not provided - will use linearly spaced slices
    if position is None:
        slice_idxs = np.linspace(0, n_total - 1, n_slices, dtype=int)
    # Position is a string
    elif isinstance(position, str) and position.lower() in ["start", "mid", "end"]:
        if position.lower() == "start":
            slice_idxs = _get_slice_range(0, n_slices, n_total)
        elif position.lower() == "mid":
            slice_idxs = _get_slice_range(n_total // 2, n_slices, n_total)
        elif position.lower() == "end":
            slice_idxs = _get_slice_range(n_total - 1, n_slices, n_total)
    #  Position is an integer
    elif isinstance(position, int):
        slice_idxs = _get_slice_range(position, n_slices, n_total)
    # Position is a list of integers
    elif isinstance(position, list) and all(isinstance(idx, int) for idx in position):
        slice_idxs = position
    else:
        raise ValueError(
            'Position not recognized. Choose an integer, list of integers or one of the following strings: "start", "mid" or "end".'
        )

    # Make grid
    nrows = math.ceil(n_slices / max_cols)
    ncols = min(n_slices, max_cols)

    # Generate figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * img_height, nrows * img_width),
        constrained_layout=True,
    )
    if nrows == 1:
        axs = [axs]  # Convert to a list for uniformity

    # Convert to NumPy array in order to use the numpy.take method
    if isinstance(vol, torch.Tensor):
        vol = vol.numpy()
    elif isinstance(vol, da.core.Array):
        vol = vol.compute()

    # Run through each ax of the grid
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(np.atleast_1d(ax_row)):
            slice_idx = i * max_cols + j
            try:
                slice_img = vol.take(slice_idxs[slice_idx], axis=axis)
                ax.imshow(
                    slice_img, cmap=cmap, interpolation=interpolation, **imshow_kwargs
                )

                if show_position:
                    ax.text(
                        0.0,
                        1.0,
                        f"slice {slice_idxs[slice_idx]} ",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=8,
                        va="top",
                        ha="left",
                        bbox=dict(facecolor="#303030", linewidth=0, pad=0),
                    )

                    ax.text(
                        1.0,
                        0.0,
                        f"axis {axis} ",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=8,
                        va="bottom",
                        ha="right",
                        bbox=dict(facecolor="#303030", linewidth=0, pad=0),
                    )

            except IndexError:
                # Not a problem, because we simply do not have a slice to show
                pass

            # Hide the axis, so that we have a nice grid
            ax.axis("off")

    if show:
        plt.show()

    plt.close()

    return fig


def _get_slice_range(position: int, n_slices: int, n_total):
    """Helper function for `slices`. Returns the range of slices to be displayed around the given position."""
    start_idx = position - n_slices // 2
    end_idx = (
        position + n_slices // 2 if n_slices % 2 == 0 else position + n_slices // 2 + 1
    )
    slice_idxs = np.arange(start_idx, end_idx)

    if slice_idxs[0] < 0:
        slice_idxs = np.arange(0, n_slices)
    elif slice_idxs[-1] > n_total:
        slice_idxs = np.arange(n_total - n_slices, n_total)

    return slice_idxs


def slicer(
    vol: Union[np.ndarray, torch.Tensor],
    axis: int = 0,
    cmap: str = "viridis",
    img_height: int = 3,
    img_width: int = 3,
    show_position: bool = False,
    interpolation: Optional[str] = "none",
    img_size = None,
    **imshow_kwargs,
) -> widgets.interactive:
    """Interactive widget for visualizing slices of a 3D volume.

    Args:
        vol (np.ndarray or torch.Tensor): The 3D volume to be sliced.
        axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        cmap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        img_height (int, optional): Height of the figure. Defaults to 3.
        img_width (int, optional): Width of the figure. Defaults to 3.
        show_position (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        slicer_obj (widgets.interactive): The interactive widget for visualizing slices of a 3D volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.slicer(vol)
        ```
        ![viz slicer](assets/screenshots/viz-slicer.gif)
    """

    if img_size:
        img_height = img_size
        img_width = img_size

    # Create the interactive widget
    def _slicer(position):
        fig = slices(
            vol,
            axis=axis,
            cmap=cmap,
            img_height=img_height,
            img_width=img_width,
            show_position=show_position,
            interpolation=interpolation,
            position=position,
            n_slices=1,
            show=True,
            **imshow_kwargs,
        )
        return fig

    position_slider = widgets.IntSlider(
        value=vol.shape[axis] // 2,
        min=0,
        max=vol.shape[axis] - 1,
        description="Slice",
        continuous_update=True,
    )
    slicer_obj = widgets.interactive(_slicer, position=position_slider)
    slicer_obj.layout = widgets.Layout(align_items="flex-start")

    return slicer_obj


def orthogonal(
    vol: Union[np.ndarray, torch.Tensor],
    cmap: str = "viridis",
    img_height: int = 3,
    img_width: int = 3,
    show_position: bool = False,
    interpolation: Optional[str] = None,
    img_size = None,
):
    """Interactive widget for visualizing orthogonal slices of a 3D volume.

    Args:
        vol (np.ndarray or torch.Tensor): The 3D volume to be sliced.
        cmap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        img_height(int, optional): Height of the figure.
        img_width(int, optional): Width of the figure.
        show_position (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        orthogonal_obj (widgets.HBox): The interactive widget for visualizing orthogonal slices of a 3D volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fly_150x256x256
        qim3d.viz.orthogonal(vol, cmap="magma")
        ```
        ![viz orthogonal](assets/screenshots/viz-orthogonal.gif)
    """

    if img_size:
        img_height = img_size
        img_width = img_size

    z_slicer = slicer(
        vol,
        axis=0,
        cmap=cmap,
        img_height=img_height,
        img_width=img_width,
        show_position=show_position,
        interpolation=interpolation,
    )
    y_slicer = slicer(
        vol,
        axis=1,
        cmap=cmap,
        img_height=img_height,
        img_width=img_width,
        show_position=show_position,
        interpolation=interpolation,
    )
    x_slicer = slicer(
        vol,
        axis=2,
        cmap=cmap,
        img_height=img_height,
        img_width=img_width,
        show_position=show_position,
        interpolation=interpolation,
    )

    z_slicer.children[0].description = "Z"
    y_slicer.children[0].description = "Y"
    x_slicer.children[0].description = "X"

    return widgets.HBox([z_slicer, y_slicer, x_slicer])


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
    vol_masked = background + foreground

    return vol_masked

def interactive_fade_mask(vol: np.ndarray, axis: int = 0):
    """ Interactive widget for visualizing the effect of edge fading on a 3D volume.

    This can be used to select the best parameters before applying the mask.

    Args:
        vol (np.ndarray): The volume to apply edge fading to.
        axis (int, optional): The axis along which to apply the fading. Defaults to 0.

    Example:
        ```python
        import qim3d
        vol = qim3d.examples.cement_128x128x128
        qim3d.viz.interactive_fade_mask(vol) 
        ```
        ![operations-edge_fade_before](assets/screenshots/viz-fade_mask.gif)  

    """

    # Create the interactive widget
    def _slicer(position, decay_rate, ratio, geometry, invert):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        axes[0].imshow(vol[position, :, :], cmap='viridis')
        axes[0].set_title('Original')
        axes[0].axis('off')

        mask = qim3d.processing.operations.fade_mask(np.ones_like(vol), decay_rate=decay_rate, ratio=ratio, geometry=geometry, axis=axis, invert=invert)
        axes[1].imshow(mask[position, :, :], cmap='viridis')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        masked_vol = qim3d.processing.operations.fade_mask(vol, decay_rate=decay_rate, ratio=ratio,  geometry=geometry, axis=axis, invert=invert)
        axes[2].imshow(masked_vol[position, :, :], cmap='viridis')
        axes[2].set_title('Masked')
        axes[2].axis('off')

        return fig
    
    shape_dropdown = widgets.Dropdown(
        options=['sphere', 'cilinder'],
        value='sphere',  # default value
        description='Geometry',
    )

    position_slider = widgets.IntSlider(
        value=vol.shape[0] // 2,
        min=0,
        max=vol.shape[0] - 1,
        description="Slice",
        continuous_update=False,
    )
    decay_rate_slider = widgets.FloatSlider(
        value=10,
        min=1,
        max=50,
        step=1.0,
        description="Decay Rate",
        continuous_update=False,
    )
    ratio_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=1,
        step=0.01, 
        description="Ratio",
        continuous_update=False,
    )

    # Create the Checkbox widget
    invert_checkbox = widgets.Checkbox(
        value=False,  # default value
        description='Invert'
    )

    slicer_obj = widgets.interactive(_slicer, position=position_slider, decay_rate=decay_rate_slider, ratio=ratio_slider, geometry=shape_dropdown, invert=invert_checkbox)
    slicer_obj.layout = widgets.Layout(align_items="flex-start")

    return slicer_obj
