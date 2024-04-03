""" 
Provides a collection of visualization functions.
"""

import math
from typing import List, Optional, Union, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

import qim3d.io
from qim3d.io.logger import log
from qim3d.utils.cc import CC
from qim3d.viz.colormaps import objects


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

        img = qim3d.examples.shell_225x128x128
        qim3d.viz.slices(img, n_slices=15)
        ```

    """

    # Numpy array or Torch tensor input
    if not isinstance(vol, (np.ndarray, torch.Tensor)):
        raise ValueError("Input must be a numpy.ndarray or torch.Tensor")

    if vol.ndim < 3:
        raise ValueError(
            "The provided object is not a volume as it has less than 3 dimensions."
        )

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

    # Convert Torch tensor to NumPy array in order to use the numpy.take method
    if isinstance(vol, torch.Tensor):
        vol = vol.numpy()

    # Run through each ax of the grid
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(np.atleast_1d(ax_row)):
            slice_idx = i * max_cols + j
            try:
                slice_img = vol.take(slice_idxs[slice_idx], axis=axis)
                ax.imshow(slice_img, cmap=cmap, interpolation=interpolation, **imshow_kwargs)

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
    **imshow_kwargs,
) -> widgets.interactive:
    """Interactive widget for visualizing slices of a 3D volume.

    Args:
        vol (np.ndarray or torch.Tensor): The 3D volume to be sliced.
        axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        cmap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        img_height(int, optional): Height of the figure. Defaults to 3.
        img_width(int, optional): Width of the figure. Defaults to 3.
        show_position (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        slicer_obj (widgets.interactive): The interactive widget for visualizing slices of a 3D volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.slicer(vol, cmap="magma")
        ```
    """

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

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.orthogonal(vol)
        ```
    """

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


def plot_cc(
    connected_components: CC,
    component_indexs: list | tuple = None,
    max_cc_to_plot=32,
    overlay=None,
    crop=False,
    show=True,
    **kwargs,
) -> list[plt.Figure]:
    """
    Plot the connected components of an image.

    Parameters:
        connected_components (CC): The connected components object.
        components (list | tuple, optional): The components to plot. If None the first max_cc_to_plot=32 components will be plotted. Defaults to None.
        max_cc_to_plot (int, optional): The maximum number of connected components to plot. Defaults to 32.
        overlay (optional): Overlay image. Defaults to None.
        crop (bool, optional): Whether to crop the image to the cc. Defaults to False.
        show (bool, optional): Whether to show the figure. Defaults to True.
        **kwargs: Additional keyword arguments to pass to `qim3d.viz.slices`.

    Returns:
        figs (list[plt.Figure]): List of figures, if `show=False`.
    """
    # if no components are given, plot the first max_cc_to_plot=32 components
    if component_indexs is None:
        if len(connected_components) > max_cc_to_plot:
            log.warning(
                f"More than {max_cc_to_plot} connected components found. Only the first {max_cc_to_plot} will be plotted. Change max_cc_to_plot to plot more components."
            )
        component_indexs = range(
            1, min(max_cc_to_plot + 1, len(connected_components) + 1)
        )
        
    figs = []
    for component in component_indexs:
        if overlay is not None:
            assert (overlay.shape == connected_components.shape), f"Overlay image must have the same shape as the connected components. overlay.shape=={overlay.shape} != connected_components.shape={connected_components.shape}."

            # plots overlay masked to connected component
            if crop:
                # Crop the overlay image based on the bounding box of the component
                bb = connected_components.get_bounding_box(component)[0]
                cc = connected_components.get_cc(component, crop=True)
                overlay_crop = overlay[bb]
                # use cc as mask for overlay_crop, where all values in cc set to 0 should be masked out, cc contains integers
                overlay_crop = np.where(cc == 0, 0, overlay_crop)
                fig = slices(overlay_crop, show=show, **kwargs)
            else:
                cc = connected_components.get_cc(component, crop=False)
                overlay_crop = np.where(cc == 0, 0, overlay)
                fig = slices(overlay_crop, show=show, **kwargs)
        else:
            # assigns discrete color map to each connected component if not given 
            if "cmap" not in kwargs:
                kwargs["cmap"] = qim3dCmap(len(component_indexs))
        
            # Plot the connected component without overlay
            fig = slices(connected_components.get_cc(component, crop=crop), show=show, **kwargs)

        figs.append(fig)

    if not show:
        return figs

    return


def local_thickness(
    image: np.ndarray,
    image_lt: np.ndarray,
    max_projection: bool = False,
    axis: int = 0,
    slice_idx: Optional[Union[int, float]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (15, 5),
) -> Union[plt.Figure, widgets.interactive]:
    """Visualizes the local thickness of a 2D or 3D image.

    Args:
        image (np.ndarray): 2D or 3D NumPy array representing the image/volume.
        image_lt (np.ndarray): 2D or 3D NumPy array representing the local thickness of the input
            image/volume.
        max_projection (bool, optional): If True, displays the maximum projection of the local
            thickness. Only used for 3D images. Defaults to False.
        axis (int, optional): The axis along which to visualize the local thickness.
            Unused for 2D images.
            Defaults to 0.
        slice_idx (int or float, optional): The initial slice to be visualized. The slice index
            can afterwards be changed. If value is an integer, it will be the index of the slice
            to be visualized. If value is a float between 0 and 1, it will be multiplied by the
            number of slices and rounded to the nearest integer. If None, the middle slice will
            be used for 3D images. Unused for 2D images. Defaults to None.
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (15, 5).

    Raises:
        ValueError: If the slice index is not an integer or a float between 0 and 1.

    Returns:
        If the input is 3D, returns an interactive widget. Otherwise, returns a matplotlib figure.

    Example:
        image_lt = qim3d.processing.local_thickness(image)
        qim3d.viz.local_thickness(image, image_lt, slice_idx=10)
    """

    def _local_thickness(image, image_lt, show, figsize, axis=None, slice_idx=None):
        if slice_idx is not None:
            image = image.take(slice_idx, axis=axis)
            image_lt = image_lt.take(slice_idx, axis=axis)

        fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained")

        axs[0].imshow(image, cmap="gray")
        axs[0].set_title("Original image")
        axs[0].axis("off")

        axs[1].imshow(image_lt, cmap="viridis")
        axs[1].set_title("Local thickness")
        axs[1].axis("off")

        plt.colorbar(
            axs[1].imshow(image_lt, cmap="viridis"), ax=axs[1], orientation="vertical"
        )

        axs[2].hist(image_lt[image_lt > 0].ravel(), bins=32, edgecolor="black")
        axs[2].set_title("Local thickness histogram")
        axs[2].set_xlabel("Local thickness")
        axs[2].set_ylabel("Count")

        if show:
            plt.show()

        plt.close()

        return fig

    # Get the middle slice if the input is 3D
    if len(image.shape) == 3:
        if max_projection:
            if slice_idx is not None:
                log.warning(
                    "slice_idx is not used for max_projection. It will be ignored."
                )
            image = image.max(axis=axis)
            image_lt = image_lt.max(axis=axis)
            return _local_thickness(image, image_lt, show, figsize)
        else:
            if slice_idx is None:
                slice_idx = image.shape[axis] // 2
            elif isinstance(slice_idx, float):
                if slice_idx < 0 or slice_idx > 1:
                    raise ValueError(
                        "Values of slice_idx of float type must be between 0 and 1."
                    )
                slice_idx = int(slice_idx * image.shape[0]) - 1
            slide_idx_slider = widgets.IntSlider(
                min=0,
                max=image.shape[axis] - 1,
                step=1,
                value=slice_idx,
                description="Slice index",
                layout=widgets.Layout(width="450px"),
            )
            widget_obj = widgets.interactive(
                _local_thickness,
                image=widgets.fixed(image),
                image_lt=widgets.fixed(image_lt),
                show=widgets.fixed(True),
                figsize=widgets.fixed(figsize),
                axis=widgets.fixed(axis),
                slice_idx=slide_idx_slider,
            )
            widget_obj.layout = widgets.Layout(align_items="center")
            if show:
                display(widget_obj)
            return widget_obj
    else:
        if max_projection:
            log.warning(
                "max_projection is only used for 3D images. It will be ignored."
            )
        if slice_idx is not None:
            log.warning("slice_idx is only used for 3D images. It will be ignored.")
        return _local_thickness(image, image_lt, show, figsize)


