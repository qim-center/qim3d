"""Provides a collection of visualization functions."""

import inspect
import io
import os
import math
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Iterable

import dask.array as da
import imageio.v2 as imageio
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import plotly.colors
import plotly.graph_objects as go
import pyvista as pv
import seaborn as sns
import skimage.measure
import zarr
from IPython.display import HTML, Image, Video, clear_output, display
from ipywidgets import widgets
from ipywidgets.widgets import Output, Widget
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

import qim3d
import qim3d.operations
from qim3d.utils import log
from qim3d.utils._decorators import coarseness

# For progress bar in Jupyter notebooks
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

ColormapLike = str | matplotlib.colors.Colormap


@coarseness('volume')
def slices_grid(
    volume: np.ndarray,
    slice_axis: int = 0,
    slice_positions: str | int | list[int] | None = None,
    n_slices: int = 15,
    max_columns: int = 5,
    colormap: str = 'magma',
    min_value: float = None,
    max_value: float = None,
    image_size: int = None,
    image_height: int = 2,
    image_width: int = 2,
    display_figure: bool = False,
    display_positions: bool = True,
    interpolation: str | None = None,
    colorbar: bool = False,
    colorbar_style: str = 'small',
    mask:np.ndarray = None,
    mask_alpha:float = 0.4,
    mask_colormap:str = 'gray',
    **matplotlib_imshow_kwargs,
) -> matplotlib.figure.Figure:
    """
    Displays one or several slices from a 3d volume.

    By default if `slice_positions` is None, slices_grid plots `n_slices` linearly spaced slices.
    If `slice_positions` is given as a string or integer, slices_grid will plot an overview with `n_slices` figures around that position.
    If `slice_positions` is given as a list, `n_slices` will be ignored and the slices from `slice_positions` will be plotted.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        slice_axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        slice_positions (int or list[int] or str or None, optional): One or several slicing levels. If None, linearly spaced slices will be displayed. Defaults to None.
        n_slices (int, optional): Defines how many slices the user wants to be displayed. Defaults to 15.
        max_columns (int, optional): The maximum number of columns to be plotted. Defaults to 5.
        colormap (str or matplotlib.colors.LinearSegmentedColormap, optional): Specifies the color map for the image. Defaults to "magma".
        min_value (float, optional): Together with max_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        max_value (float, optional): Together with min_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        image_size (int, optional): Size of the figure. If set, image_height and image_width are ignored.
        image_height (int, optional): Height of the figure.
        image_width (int, optional): Width of the figure.
        display_figure (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.
        display_positions (bool, optional): If True, displays the position of the slices. Defaults to True.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.
        colorbar (bool, optional): Adds a colorbar positioned in the top-right for the corresponding colormap and data range. Defaults to False.
        colorbar_style (str, optional): Determines the style of the colorbar. Option 'small' is height of one image row. Option 'large' spans full height of image grid. Defaults to 'small'.
        **matplotlib_imshow_kwargs (Any): Additional keyword arguments to pass to the `matplotlib.pyplot.imshow` function.

    Returns:
        fig (matplotlib.figure.Figure): The figure with the slices from the 3d array.

    Raises:
        ValueError: If the input is not a numpy.ndarray or da.Array.
        ValueError: If the slice_axis to slice along is not a valid choice, i.e. not an integer between 0 and the number of dimensions of the volume minus 1.
        ValueError: If the file or array is not a volume with at least 3 dimensions.
        ValueError: If the `position` keyword argument is not a integer, list of integers or one of the following strings: "start", "mid" or "end".
        ValueError: If the colorbar_style keyword argument is not one of the following strings: 'small' or 'large'.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.shell_225x128x128
        qim3d.viz.slices_grid(vol, n_slices=15)
        ```
        ![Grid of slices](../../assets/screenshots/viz-slices.png)

    """
    if image_size:
        image_height = image_size
        image_width = image_size

    # If we pass python None to the imshow function, it will set to
    # default value 'antialiased'
    if interpolation is None:
        interpolation = 'none'

    # Numpy array or Torch tensor input
    if not isinstance(volume, np.ndarray | da.Array):
        msg = 'Data type not supported'
        raise ValueError(msg)

    if volume.ndim < 3:
        msg = 'The provided object is not a volume as it has less than 3 dimensions.'
        raise ValueError(msg)

    colorbar_style_options = ['small', 'large']
    if colorbar_style not in colorbar_style_options:
        msg = f"Value '{colorbar_style}' is not valid for colorbar style. Please select from {colorbar_style_options}."
        raise ValueError(msg)

    if isinstance(volume, da.Array):
        volume = volume.compute()

    # Ensure axis is a valid choice
    if not (0 <= slice_axis < volume.ndim):
        msg = f"Invalid value for 'slice_axis'. It should be an integer between 0 and {volume.ndim - 1}."
        raise ValueError(msg)

    # Here we deal with the case that the user wants to use the objects colormap directly
    if (
        type(colormap) == matplotlib.colors.LinearSegmentedColormap
        or colormap == 'segmentation'
    ):
        num_labels = volume.max()

        if colormap == 'segmentation':
            colormap = qim3d.viz.colormaps.segmentation(num_labels)
        # If min_value and max_value are not set like this, then in case the
        # number of objects changes on new slice, objects might change
        # colors. So when using a slider, the same object suddently
        # changes color (flickers), which is confusing and annoying.
        min_value = 0
        max_value = num_labels

    # Get total number of slices in the specified dimension
    n_total = volume.shape[slice_axis]

    # Position is not provided - will use linearly spaced slices
    if slice_positions is None:
        slice_idxs = np.linspace(0, n_total - 1, n_slices, dtype=int)
    # Position is a string
    elif isinstance(slice_positions, str) and slice_positions.lower() in [
        'start',
        'mid',
        'end',
    ]:
        if slice_positions.lower() == 'start':
            slice_idxs = _get_slice_range(0, n_slices, n_total)
        elif slice_positions.lower() == 'mid':
            slice_idxs = _get_slice_range(n_total // 2, n_slices, n_total)
        elif slice_positions.lower() == 'end':
            slice_idxs = _get_slice_range(n_total - 1, n_slices, n_total)
    #  Position is an integer
    elif isinstance(slice_positions, int):
        slice_idxs = _get_slice_range(slice_positions, n_slices, n_total)
    # Position is a list of integers
    elif isinstance(slice_positions, list) and all(map(lambda x:isinstance(x, int), slice_positions)):
        slice_idxs = np.array(slice_positions)
        if any(slice_idxs < 0):
            dim = volume.shape[slice_axis]
            slice_idxs[np.where(slice_idxs < 0)] += dim
        n_slices = len(slice_idxs)
            

    else:
        msg = 'Position not recognized. Choose an integer, list of integers or one of the following strings: "start", "mid" or "end".'
        raise ValueError(msg)

    # Make grid
    nrows = math.ceil(n_slices / max_columns)
    ncols = min(n_slices, max_columns)

    # Generate figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * image_height, nrows * image_width),
        constrained_layout=True,
    )

    if nrows == 1:
        axs = [axs]  # Convert to a list for uniformity

    # Convert to NumPy array in order to use the numpy.take method
    if isinstance(volume, da.Array):
        volume = volume.compute()

    if colorbar:
        # In this case, we want the vrange to be constant across the
        # slices, which makes them all comparable to a single colorbar.
        new_min_value = min_value if min_value is not None else np.min(volume)
        new_max_value = max_value if max_value is not None else np.max(volume)

    # Run through each ax of the grid
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(np.atleast_1d(ax_row)):
            slice_idx = i * max_columns + j
            try:
                slice_img = volume.take(slice_idxs[slice_idx], axis=slice_axis)
                slice_mask = (
                    None
                    if mask is None
                    else mask.take(slice_idxs[slice_idx], axis=slice_axis)
                )

                if not colorbar:
                    # If min_value is higher than the highest value in the
                    # image ValueError is raised. We don't want to
                    # override the values because next slices might be okay
                    new_min_value = (
                        None
                        if (
                            isinstance(min_value, float | int)
                            and min_value > np.max(slice_img)
                        )
                        else min_value
                    )
                    new_max_value = (
                        None
                        if (
                            isinstance(max_value, float | int)
                            and max_value < np.min(slice_img)
                        )
                        else max_value
                    )

                ax.imshow(
                    slice_img,
                    cmap=colormap,
                    interpolation=interpolation,
                    vmin=new_min_value,
                    vmax=new_max_value,
                    **matplotlib_imshow_kwargs,
                )
                if slice_mask is not None:
                    ax.imshow(slice_mask, cmap = mask_colormap, alpha = mask_alpha)

                if display_positions:
                    ax.text(
                        0.0,
                        1.0,
                        f'slice {slice_idxs[slice_idx]} ',
                        transform=ax.transAxes,
                        color='white',
                        fontsize=8,
                        va='top',
                        ha='left',
                        bbox={'facecolor': '#303030', 'linewidth': 0, 'pad': 0},
                    )

                    ax.text(
                        1.0,
                        0.0,
                        f'axis {slice_axis} ',
                        transform=ax.transAxes,
                        color='white',
                        fontsize=8,
                        va='bottom',
                        ha='right',
                        bbox={'facecolor': '#303030', 'linewidth': 0, 'pad': 0},
                    )

            except IndexError:
                # Not a problem, because we simply do not have a slice to show
                pass

            # Hide the axis, so that we have a nice grid
            ax.axis('off')

    if colorbar:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            fig.tight_layout()

        norm = matplotlib.colors.Normalize(
            vmin=new_min_value, vmax=new_max_value, clip=True
        )
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)

        if colorbar_style == 'small':
            # Figure coordinates of top-right axis
            tr_pos = np.atleast_1d(axs[0])[-1].get_position()
            # The width is divided by ncols to make it the same relative size to the images
            colorbar_ax = fig.add_axes(
                [tr_pos.x1 + 0.05 / ncols, tr_pos.y0, 0.05 / ncols, tr_pos.height]
            )
            fig.colorbar(mappable=mappable, cax=colorbar_ax, orientation='vertical')
        elif colorbar_style == 'large':
            # Figure coordinates of bottom- and top-right axis
            br_pos = np.atleast_1d(axs[-1])[-1].get_position()
            tr_pos = np.atleast_1d(axs[0])[-1].get_position()
            # The width is divided by ncols to make it the same relative size to the images
            colorbar_ax = fig.add_axes(
                [
                    br_pos.xmax + 0.05 / ncols,
                    br_pos.y0 + 0.0015,
                    0.05 / ncols,
                    (tr_pos.y1 - br_pos.y0) - 0.0015,
                ]
            )
            fig.colorbar(mappable=mappable, cax=colorbar_ax, orientation='vertical')

    if display_figure:
        plt.show()

    plt.close()

    return fig


def _get_slice_range(position: int, n_slices: int, n_total: int) -> np.ndarray:
    """Helper function for `slices`. Returns the range of slices to be displayed around the given position."""
    start_idx = position - n_slices // 2
    end_idx = (
        position + n_slices // 2
        if n_slices % 2 == 0
        else position + n_slices // 2 + 1
    )
    slice_idxs = np.arange(start_idx, end_idx)

    if slice_idxs[0] < 0:
        slice_idxs = np.arange(0, n_slices)
    elif slice_idxs[-1] > n_total:
        slice_idxs = np.arange(n_total - n_slices, n_total)

    return slice_idxs


@coarseness('volume')
def slicer(
    volume: np.ndarray,
    slice_axis: int = 0,
    colormap: str = 'magma',
    min_value: float = None,
    max_value: float = None,
    image_height: int = 3,
    image_width: int = 3,
    display_positions: bool = False,
    interpolation: str | None = None,
    image_size: int = None,
    colorbar: str = None,
    mask:np.ndarray = None,
    mask_alpha:float = 0.4,
    mask_colormap = 'gray',
    default_position:float|int = 0.5,
    **matplotlib_imshow_kwargs,
) -> widgets.interactive:
    """
    Interactive widget for visualizing slices of a 3D volume.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        slice_axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        colormap (str or matplotlib.colors.LinearSegmentedColormap, optional): Specifies the color map for the image. Defaults to 'magma'.
        min_value (float, optional): Together with max_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        max_value (float, optional): Together with min_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        image_height (int, optional): Height of the figure. Defaults to 3.
        image_width (int, optional): Width of the figure. Defaults to 3.
        display_positions (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.
        image_size (int, optional): Size of the figure. If set, image_height and image_width are ignored. Defaults to None.
        colorbar (str, optional): Controls the options for color bar. If None, no color bar is included. If 'volume', the color map range is constant for each slice. If 'slices', the color map range changes dynamically according to the slice. Defaults to None.
        mask (np.ndarray, optional): Overlays the image with this segmentation mask. Defaults to None.
        mask_alpha (float, optional): Sets the alpha of the overlaying mask. Defaults to 0.4.
        mask_colormap (str, optional): Sets the color map of the overlaying mask. Defaults to 'gray'.
        default_position (float|int, optional): Set the x slicer to this slice after reload. If float, it should be between 0 and 1 to set position relative to shape. If int, it sets the exact slice. Defaults to 0.5.
        **matplotlib_imshow_kwargs (Any): Additional keyword arguments to pass to the `matplotlib.pyplot.imshow` function.

    Returns:
        slicer_obj (widgets.interactive): The interactive widget for visualizing slices of a 3D volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.slicer(vol)
        ```
        ![viz slicer](../../assets/screenshots/viz-slicer.gif)

    """

    if image_size:
        image_height = image_size
        image_width = image_size

    colorbar_options = [None, 'slices', 'volume']
    if colorbar not in colorbar_options:
        msg = (
            f"Unrecognized value '{colorbar}' for parameter colorbar. "
            f'Expected one of {colorbar_options}.'
        )
        raise ValueError(msg)
    show_colorbar = colorbar is not None
    if colorbar == 'slices':
        # Precompute the minimum and maximum along each slice for faster widget sliding.
        non_slice_axes = tuple(i for i in range(volume.ndim) if i != slice_axis)
        slice_mins = np.min(volume, axis=non_slice_axes)
        slice_maxs = np.max(volume, axis=non_slice_axes)

    # Create the interactive widget
    def _slicer(slice_positions: int) -> Figure:
        if colorbar == 'slices':
            dynamic_min = slice_mins[slice_positions]
            dynamic_max = slice_maxs[slice_positions]
        else:
            dynamic_min = min_value
            dynamic_max = max_value

        fig = slices_grid(
            volume,
            slice_axis=slice_axis,
            colormap=colormap,
            min_value=dynamic_min,
            max_value=dynamic_max,
            image_height=image_height,
            image_width=image_width,
            display_positions=display_positions,
            interpolation=interpolation,
            slice_positions=slice_positions,
            n_slices=1,
            display_figure=True,
            colorbar=show_colorbar,
            mask = mask,
            mask_alpha = mask_alpha,
            mask_colormap = mask_colormap,
            **matplotlib_imshow_kwargs,
        )
        return fig

    if isinstance(default_position, float):
        default_position = int(default_position * (volume.shape[slice_axis] - 1))
    if isinstance(default_position, int):
        if default_position < 0:
            default_position = volume.shape[slice_axis] - default_position
        default_position = np.clip(
            default_position, a_min=0, a_max=volume.shape[slice_axis] - 1
        )
    else:
        default_position = volume.shape[slice_axis] // 2

    position_slider = widgets.IntSlider(
        value=default_position,
        min=0,
        max=volume.shape[slice_axis] - 1,
        description='Slice',
        continuous_update=True,
    )
    slicer_obj = widgets.interactive(_slicer, slice_positions=position_slider)
    slicer_obj.layout = widgets.Layout(align_items='flex-start')

    return slicer_obj


@coarseness('volume')
def slicer_orthogonal(
    volume: np.ndarray,
    colormap: str = 'magma',
    min_value: float = None,
    max_value: float = None,
    image_height: int = 3,
    image_width: int = 3,
    display_positions: bool = False,
    interpolation: str | None = None,
    image_size: int = None,
    colorbar:str = None,
    mask:np.ndarray = None,
    mask_alpha:float = 0.4,
    mask_colormap:str = 'gray',
    default_z:float|int = 0.5,
    default_y:float|int = 0.5,
    default_x:float|int = 0.5,
) -> widgets.interactive:
    """
    Interactive widget for visualizing orthogonal slices of a 3D volume.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        colormap (str or matplotlib.colors.LinearSegmentedColormap, optional): Specifies the color map for the image. Defaults to "magma".
        min_value (float, optional): Together with max_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        max_value (float, optional): Together with min_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        image_height (int, optional): Height of the figure.
        image_width (int, optional): Width of the figure.
        display_positions (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.
        image_size (int, optional): Size of the figure. If set, image_height and image_width are ignored. Defaults to None.
        colorbar (str, optional): Controls the options for color bar. If None, no color bar is included. If 'volume', the color map range is constant for each slice. If 'slices', the color map range changes dynamically according to the slice. Defaults to None.
        mask (np.ndarray, optional): Overlays the image with this segmentation mask. Defaults to None.
        mask_alpha (float, optional): Sets the alpha of the overlaying mask. Defaults to 0.4.
        mask_colormap (str, optional): Sets the color map of the overlaying mask. Defaults to 'gray'.
        default_x (float|int, optional): Set the x slicer to this slice after reload. If float, it should be between 0 and 1 to set position relative to shape. If int, it sets the exact slice. Defaults to 0.5.
        default_y (float|int, optional): Set the x slicer to this slice after reload. If float, it should be between 0 and 1 to set position relative to shape. If int, it sets the exact slice. Defaults to 0.5.
        default_z (float|int, optional): Set the x slicer to this slice after reload. If float, it should be between 0 and 1 to set position relative to shape. If int, it sets the exact slice. Defaults to 0.5.

    Returns:
        slicer_orthogonal_obj (widgets.HBox): The interactive widget for visualizing orthogonal slices of a 3D volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fly_150x256x256
        qim3d.viz.slicer_orthogonal(vol, colormap="magma")
        ```
        ![viz slicer_orthogonal](../../assets/screenshots/viz-orthogonal.gif)

    """

    if image_size:
        image_height = image_size
        image_width = image_size

    get_slicer_for_axis = lambda slice_axis, default_position: slicer(
        volume,
        slice_axis=slice_axis,
        colormap=colormap,
        min_value=min_value,
        max_value=max_value,
        image_height=image_height,
        image_width=image_width,
        display_positions=display_positions,
        interpolation=interpolation,
        colorbar=colorbar,
        mask = mask,
        mask_alpha = mask_alpha,
        mask_colormap = mask_colormap,
        default_position=default_position
    )

    z_slicer = get_slicer_for_axis(slice_axis=0, default_position=default_z)
    y_slicer = get_slicer_for_axis(slice_axis=1, default_position=default_y)
    x_slicer = get_slicer_for_axis(slice_axis=2, default_position=default_x)

    z_slicer.children[0].description = 'Z'
    y_slicer.children[0].description = 'Y'
    x_slicer.children[0].description = 'X'

    return widgets.HBox([z_slicer, y_slicer, x_slicer])


@coarseness('volume')
def fade_mask(
    volume: np.ndarray,
    axis: int = 0,
    colormap: str = 'magma',
    min_value: float = None,
    max_value: float = None,
) -> widgets.interactive:
    """
    Interactive widget for visualizing the effect of edge fading on a 3D volume.

    This can be used to select the best parameters before applying the mask.

    Args:
        volume (np.ndarray): The volume to apply edge fading to.
        axis (int, optional): The axis along which to apply the fading. Defaults to 0.
        colormap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        min_value (float or None, optional): Together with max_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        max_value (float or None, optional): Together with min_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None

    Returns:
        slicer_obj (widgets.HBox): The interactive widget for visualizing fade mask on slices of a 3D volume.

    Example:
        ```python
        import qim3d
        vol = qim3d.examples.cement_128x128x128
        qim3d.viz.fade_mask(vol)
        ```
        ![operations-edge_fade_before](../../assets/screenshots/viz-fade_mask.gif)

    """

    # Create the interactive widget
    def _slicer(
        position: int,
        decay_rate: float,
        ratio: float,
        geometry: str,
        invert: bool,
    ) -> Figure:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        slice_img = volume[position, :, :]
        # If min_value is higher than the highest value in the image ValueError is raised
        # We don't want to override the values because next slices might be okay
        new_min_value = (
            None
            if (isinstance(min_value, float | int) and min_value > np.max(slice_img))
            else min_value
        )
        new_max_value = (
            None
            if (isinstance(max_value, float | int) and max_value < np.min(slice_img))
            else max_value
        )

        axes[0].imshow(
            slice_img, cmap=colormap, vmin=new_min_value, vmax=new_max_value
        )
        axes[0].set_title('Original')
        axes[0].axis('off')

        mask = qim3d.operations.fade_mask(
            np.ones_like(volume),
            decay_rate=decay_rate,
            ratio=ratio,
            geometry=geometry,
            axis=axis,
            invert=invert,
        )
        axes[1].imshow(mask[position, :, :], cmap=colormap)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        masked_volume = qim3d.operations.fade_mask(
            volume,
            decay_rate=decay_rate,
            ratio=ratio,
            geometry=geometry,
            axis=axis,
            invert=invert,
        )
        # If min_value is higher than the highest value in the image ValueError is raised
        # We don't want to override the values because next slices might be okay
        slice_img = masked_volume[position, :, :]
        new_min_value = (
            None
            if (isinstance(min_value, float | int) and min_value > np.max(slice_img))
            else min_value
        )
        new_max_value = (
            None
            if (isinstance(max_value, float | int) and max_value < np.min(slice_img))
            else max_value
        )
        axes[2].imshow(
            slice_img, cmap=colormap, vmin=new_min_value, vmax=new_max_value
        )
        axes[2].set_title('Masked')
        axes[2].axis('off')

        return fig

    shape_dropdown = widgets.Dropdown(
        options=['spherical', 'cylindrical'],
        value='spherical',  # default value
        description='Geometry',
    )

    position_slider = widgets.IntSlider(
        value=volume.shape[0] // 2,
        min=0,
        max=volume.shape[0] - 1,
        description='Slice',
        continuous_update=False,
    )
    decay_rate_slider = widgets.FloatSlider(
        value=10,
        min=1,
        max=50,
        step=1.0,
        description='Decay Rate',
        continuous_update=False,
    )
    ratio_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=1,
        step=0.01,
        description='Ratio',
        continuous_update=False,
    )

    # Create the Checkbox widget
    invert_checkbox = widgets.Checkbox(
        value=False,
        description='Invert',  # default value
    )

    slicer_obj = widgets.interactive(
        _slicer,
        position=position_slider,
        decay_rate=decay_rate_slider,
        ratio=ratio_slider,
        geometry=shape_dropdown,
        invert=invert_checkbox,
    )
    slicer_obj.layout = widgets.Layout(align_items='flex-start')

    return slicer_obj


def chunks(zarr_path: str, **kwargs) -> widgets.VBox:
    """
    Launch an interactive chunk explorer for a 3D or 5D OME-Zarr/Zarr dataset.

    Args:
        zarr_path (str):
            Path to the OME-Zarr/Zarr dataset.

        **kwargs (Any):
            Additional keyword arguments that are **selectively** forwarded
            only to the visualization method that supports them. Any key
            not accepted by the chosen method is ignored.

            The visualization methods available in this tool are:

            - `slicer` → calls `qim3d.viz.slicer`
            - `slices` → calls `qim3d.viz.slices_grid`
            - `volume` → calls `qim3d.viz.volumetric`

            Users select the desired method via the dropdown menu in the widget.

    Raises:
        ValueError: If the dataset's dimensionality is not 3 or 5.

    Returns:
        chunk_explorer (widgets.VBox): A widget containing dropdowns for selecting the OME-Zarr scale, chunk coordinates along each axis, and visualization method.

    Example:
        ```python
        import qim3d

        # Visualize interactive chunks explorer
        qim3d.viz.chunks('path/to/zarr/dataset.zarr')
        ```
        ![interactive chunks explorer](../../assets/screenshots/chunks_explorer.gif)

    """
    # Opens the Zarr dataset - doesn't load to memory yet
    zarr_data = zarr.open(zarr_path, mode='r')

    title = widgets.HTML('<h2>Chunk Explorer</h2>')
    info_label = widgets.HTML(value='Chunk info will be displayed here')

    def get_num_chunks(shape: Sequence[int], chunk_size: Sequence[int]) -> list[int]:
        return [(s + chunk_size[i] - 1) // chunk_size[i] for i, s in enumerate(shape)]

    def _filter_kwargs(
        function: Callable[..., Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter kwargs to only include those that are accepted by the function."""
        sig = inspect.signature(function)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    def load_and_visualize(
        key: int,
        *coords: int,
        visualization_method: Literal['slicer', 'slices', 'volume'],
        **inner_kwargs: object,
    ) -> Widget | Figure | Output:
        key = _path_from_dropdown(key)
        arr = da.from_zarr(zarr_data) if isinstance(zarr_data, zarr.Array) else da.from_zarr(zarr_data[key])
        shape = arr.shape
        chunksz = arr.chunks

        if arr.ndim == 3:
            z_idx, y_idx, x_idx = coords
            slices = (
                slice(
                    z_idx * chunksz[0][0], min((z_idx + 1) * chunksz[0][0], shape[0])
                ),
                slice(
                    y_idx * chunksz[1][0], min((y_idx + 1) * chunksz[1][0], shape[1])
                ),
                slice(
                    x_idx * chunksz[2][0], min((x_idx + 1) * chunksz[2][0], shape[2])
                ),
            )
            chunk = arr[slices].compute()
        elif arr.ndim == 5:
            t_idx, c_idx, z_idx, y_idx, x_idx = coords
            slices = (
                slice(
                    t_idx * chunksz[0][0], min((t_idx + 1) * chunksz[0][0], shape[0])
                ),
                slice(
                    c_idx * chunksz[1][0], min((c_idx + 1) * chunksz[1][0], shape[1])
                ),
                slice(
                    z_idx * chunksz[2][0], min((z_idx + 1) * chunksz[2][0], shape[2])
                ),
                slice(
                    y_idx * chunksz[3][0], min((y_idx + 1) * chunksz[3][0], shape[3])
                ),
                slice(
                    x_idx * chunksz[4][0], min((x_idx + 1) * chunksz[4][0], shape[4])
                ),
            )
            chunk = arr[slices].compute()
            chunk = chunk[0, 0, ...]
        else:
            msg = f'Unsupported ndim={arr.ndim}'
            raise ValueError(msg)

        mins, maxs, means = chunk.min(), chunk.max(), chunk.mean()
        ranges = [f'{sl.start}-{sl.stop}' for sl in slices]
        coords_str = ', '.join(str(c) for c in coords)
        info_html = (
            f"<div style='font-size:14px; margin-left:32px'>"
            f"<h3 style='margin:0'>Chunk Info</h3>"
            f'<pre>'
            f'shape      : {chunk.shape}\n'
            f'coords     : ({coords_str})\n'
            f'ranges     : {ranges}\n'
            f'dtype      : {chunk.dtype}\n'
            f'min / max  : {mins:.0f} / {maxs:.0f}\n'
            f'mean value : {means:.0f}\n'
            f'</pre></div>'
        )
        info_label.value = info_html

        if visualization_method == 'slicer':
            kw = _filter_kwargs(qim3d.viz.slicer, inner_kwargs)
            return qim3d.viz.slicer(chunk, **kw)
        if visualization_method == 'slices':
            out = widgets.Output()
            with out:
                kw = _filter_kwargs(qim3d.viz.slices_grid, inner_kwargs)
                fig = qim3d.viz.slices_grid(chunk, **kw)
                display(fig)
            return out
        # volume
        out = widgets.Output()
        with out:
            kw = _filter_kwargs(qim3d.viz.volumetric, inner_kwargs)
            vol = qim3d.viz.volumetric(chunk, show=False, **kw)
            display(vol)
        return out
    
    def _path_from_dropdown(string:str):
        return string.split('(')[0].strip()
    
    if isinstance(zarr_data, zarr.Group): 
        scale_opts = [f'{key} {zarr_data[key].shape}' for key in sorted(zarr_data.keys())]
    elif isinstance(zarr_data, zarr.Array):
        scale_opts = [f'{zarr_data.shape}',]
    drop_style = {'description_width': '120px'}
    scale_dd = widgets.Dropdown(
        options=scale_opts, description='Scale:', style=drop_style
    )

    # first_shape = zarr_data[0].shape
    if isinstance(zarr_data, zarr.Array):
        first_shape = zarr_data.shape
        chunks = zarr_data.chunks
    else:
        first_array = zarr_data[_path_from_dropdown(scale_opts[0])]
        first_shape = first_array.shape
        chunks = first_array.chunks


    if len(first_shape) == 3:
        axis_names = ['Z', 'Y', 'X']
    elif len(first_shape) == 5:
        axis_names = ['T', 'C', 'Z', 'Y', 'X']
    else:
        msg = f'Only 3D or 5D supported, got ndim={len(first_shape)}'
        raise ValueError(msg)

    counts0 = get_num_chunks(first_shape, chunks)
    axis_dds = []
    for name, cnt in zip(axis_names, counts0):
        dd = widgets.Dropdown(
            options=list(range(cnt)), value=0, description=f'{name}:', style=drop_style
        )
        axis_dds.append(dd)

    method_dd = widgets.Dropdown(
        options=['slicer', 'slices', 'volume'],
        value='slicer',
        description='Viz:',
        style=drop_style,
    )

    def disable_observers() -> None:
        for dd in (*axis_dds, method_dd):
            dd.unobserve(_update_vis, names='value')

    def enable_observers() -> None:
        for dd in (*axis_dds, method_dd):
            dd.observe(_update_vis, names='value')

    def _update_coords(key: str) -> None:
        disable_observers()
        key = _path_from_dropdown(key)
        shp = zarr_data[key].shape
        cnts = get_num_chunks(shp, zarr_data[key].chunks)
        for dd, c in zip(axis_dds, cnts):
            dd.options = list(range(c))
            dd.disabled = c == 1
            dd.value = 0
        enable_observers()
        _update_vis()

    def _update_vis(*_) -> None:
        coords = [dd.value for dd in axis_dds]
        widget = load_and_visualize(
            scale_dd.value, *coords, visualization_method=method_dd.value, **kwargs
        )
        container.children = [title, controls_with_info, widget]

    scale_dd.observe(lambda change: _update_coords(scale_dd.value), names='value')
    enable_observers()

    initial = load_and_visualize(
        scale_dd.value,
        *[dd.value for dd in axis_dds],
        visualization_method=method_dd.value,
        **kwargs,
    )

    control_box = widgets.VBox([scale_dd, *axis_dds, method_dd])
    controls_with_info = widgets.HBox([control_box, info_label])
    container = widgets.VBox([title, controls_with_info, initial])
    return container


@coarseness('volume')
def histogram(
    volume: np.ndarray,
    coarseness: int | list[int] = 1,
    ignore_zero: bool = True,
    bins: int | str = 'auto',
    slice_index: int | str | None = None,
    slice_axis: int = 0,
    vertical_line: int | Iterable = None,
    vertical_line_colormap: str | Iterable = 'qim',
    kde: bool = False,
    log_scale: bool = False,
    despine: bool = True,
    show_title: bool = True,
    color: str = 'qim3d',
    edgecolor: str | None = None,
    figsize: tuple[float, float] = (8, 4.5),
    bin_style: Literal['bars', 'step', 'poly'] = 'step',
    return_fig: bool = False,
    show: bool = True,
    ax: plt.Axes | None = None,
    **sns_kwargs: str | float | bool,
) -> plt.Figure | plt.Axes | None:
    """
    Plots a histogram of voxel intensities from a 3D volume, with options to show a specific slice or the entire volume.

    Utilizes [seaborn.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html) for visualization.

    Args:
        volume (np.ndarray): A 3D NumPy array representing the volume to be visualized.
        coarseness (int or list[int], optional): A positive integer representing the coarseness of the subsampling. A value of 1 (default) uses the original volume, a value of 2 uses every second element along each axis and so on. Used to reduce the needed computation.
        ignore_zero (bool, optional): Specifies if zero-values in the volume should be ignored.
        bins (Union[int, str], optional): Number of histogram bins or a binning strategy (e.g., "auto"). Default is "auto".
        slice_axis (int, optional): Axis along which to take a slice. Default is 0.
        slice_index (Union[int, str], optional): Specifies the slice to visualize. If an integer, it represents the slice index along the selected axis.
                                               If "middle", the function uses the middle slice. If None, the entire volume is visualized. Default is None.
        vertical_line (Union[int, Iterable], optional): Intensity value for a vertical line(s) to be drawn on the histogram. Default is None.
        vertical_line_colormap (Union[str, Iterable], optional): Colors for vertical lines. If string, it should be a valid colormap. If iterable, it should be list of valid colors. Default is 'qim'.
        kde (bool, optional): Whether to overlay a kernel density estimate.
        log_scale (bool, optional): Whether to use a logarithmic scale on the y-axis. Default is False.
        despine (bool, optional): If True, removes the top and right spines from the plot for cleaner appearance. Default is True.
        show_title (bool, optional): If True, displays a title with slice information. Default is True.
        color (str, optional): Color for the histogram bars. If "qim3d", defaults to the qim3d color. Default is "qim3d".
        edgecolor (str, optional): Color for the edges of the histogram bars. Default is None.
        figsize (tuple, optional): Size of the figure (width, height). Default is (8, 4.5).
        bin_style (str, optional): Type of histogram to draw ('bars', 'step', or 'poly'). Default is "step".
        return_fig (bool, optional): If True, returns the figure object instead of showing it directly. Default is False.
        show (bool, optional): If True, displays the plot. If False, suppresses display. Default is True.
        ax (matplotlib.axes.Axes, optional): Axes object where the histogram will be plotted. Default is None.
        **sns_kwargs: Additional keyword arguments for `seaborn.histplot`.

    Returns:
        Optional[matplotlib.figure.Figure or matplotlib.axes.Axes]:
            If `return_fig` is True, returns the generated figure object.
            If `return_fig` is False and `ax` is provided, returns the `Axes` object.
            Otherwise, returns None.

    Raises:
        ValueError: If `slice_axis` is not a valid axis index (0, 1, or 2).
        ValueError: If `slice_index` is an integer and is out of range for the specified axis.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.histogram(vol)
        ```
        ![viz histogram](../../assets/screenshots/viz-histogram-vol.png)

    Example: Histogram from a single slice
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.histogram(vol, slice_index=100, slice_axis=1, bin_style='bars', edgecolor='white')
        ```
        ![viz histogram](../../assets/screenshots/viz-histogram-slice.png)

    Example: Using coarsness for faster computation
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.histogram(vol, coarseness=2, kde=True, log_scale=True)
        ```
        ![viz histogram](../../assets/screenshots/viz-histogram-coarse.png)

    """
    if not (0 <= slice_axis < volume.ndim):
        msg = f'Axis must be an integer between 0 and {volume.ndim - 1}.'
        raise ValueError(msg)

    title_suffixes = []

    if slice_index == 'middle':
        slice_index = volume.shape[slice_axis] // 2

    if slice_index is not None:
        if 0 <= slice_index < volume.shape[slice_axis]:
            img_slice = np.take(volume, indices=slice_index, axis=slice_axis)
            data = img_slice.ravel()
            title = f'Intensity histogram of slice #{slice_index} {img_slice.shape} along axis {slice_axis}'
        else:
            msg = f'Slice index out of range. Must be between 0 and {volume.shape[slice_axis] - 1}.'
            raise ValueError(msg)
    else:
        data = volume.ravel()
        title = f'Intensity histogram for volume {volume.shape}'

    if ignore_zero:
        data = data[data > 0]
        title_suffixes.append('zero-values ignored')

    if title_suffixes:
        title += ' (' + ', '.join(title_suffixes) + ')'

    # Use provided Axes or create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if log_scale:
        ax.set_yscale('log')

    if color == 'qim3d':
        color = qim3d.viz.colormaps.qim(1.0)

    sns.histplot(
        data,
        bins=bins,
        kde=kde,
        color=color,
        element=bin_style,
        edgecolor=edgecolor,
        ax=ax,  # Plot directly on the specified Axes
        **sns_kwargs,
    )

    if vertical_line is not None:
        
        if isinstance(vertical_line_colormap, str):
            colors = matplotlib.colormaps[vertical_line_colormap]
        elif isinstance(vertical_line_colormap, Iterable):
            colors = lambda x: vertical_line_colormap[x]


        if isinstance(vertical_line, (float, int)):
            ax.axvline(
                x=vertical_line,
                color=colors(0),
                linestyle='--',
                linewidth=2,
            )
        elif isinstance(vertical_line, Iterable):
            for index, line_position in enumerate(vertical_line):
                if isinstance(vertical_line_colormap, str):
                    index = index/(max(len(vertical_line)-1, 1))
                ax.axvline(
                    x=line_position,
                    color=colors(index),
                    linestyle='--',
                    linewidth=2,
                )

    if despine:
        sns.despine(
            fig=None,
            ax=ax,
            top=True,
            right=True,
            left=False,
            bottom=False,
            offset={'left': 0, 'bottom': 18},
            trim=True,
        )

    ax.set_xlabel('Voxel Intensity')
    ax.set_ylabel('Frequency')

    if show_title:
        ax.set_title(title, fontsize=10)

    # Handle show and return
    if show and fig is not None:
        plt.show()

    if return_fig:
        return fig
    elif ax is not None:
        return ax


class _LineProfile:
    def __init__(
        self,
        volume: np.ndarray,
        slice_axis: int,
        slice_index: int,
        vertical_position: int,
        horizontal_position: int,
        angle: float,
        fraction_range: tuple[float, float],
        ylim: Literal['auto', 'full', 'manual'] | tuple[float, float],
    ):
        self.volume = volume
        self.slice_axis = slice_axis

        self.data_min = self.volume.min()
        self.data_max = self.volume.max()
        self.data_span = self.data_max - self.data_min
        if isinstance(ylim, str):
            self.ylim_style = ylim
            self.ylim = [self.data_min, self.data_max]
        else:
            self.ylim_style = 'manual'
            self.ylim = ylim

        self.dims = np.array(volume.shape)
        self.pad = 1  # Padding on pivot point to avoid border issues
        self.cmap = [matplotlib.cm.plasma, matplotlib.cm.spring][1]

        self.initialize_widgets()
        self.update_slice_axis(slice_axis)
        self.slice_index_widget.value = slice_index
        self.x_widget.value = horizontal_position
        self.y_widget.value = vertical_position
        self.angle_widget.value = angle
        self.line_fraction_widget.value = [fraction_range[0], fraction_range[1]]

    def update_slice_axis(self, slice_axis: int) -> None:
        self.slice_axis = slice_axis
        self.slice_index_widget.max = self.volume.shape[slice_axis] - 1
        self.slice_index_widget.value = self.volume.shape[slice_axis] // 2

        self.x_max, self.y_max = np.delete(self.dims, self.slice_axis) - 1
        self.x_widget.max = self.x_max - self.pad
        self.x_widget.value = self.x_max // 2
        self.y_widget.max = self.y_max - self.pad
        self.y_widget.value = self.y_max // 2

    def update_ylim(self, ax: plt.Axes, ylim_style: str) -> None:
        self.ylim_widget.layout.display = 'none'
        if ylim_style == 'full':
            pad = 0.05
            ax.set_ylim(
                self.data_min - pad * self.data_span,
                self.data_max + pad * self.data_span,
            )
        elif ylim_style == 'manual':
            ax.set_ylim(self.ylim[0], self.ylim[1])
            self.ylim_widget.layout.display = 'flex'

    def initialize_widgets(self) -> None:
        layout = widgets.Layout(width='300px', height='auto')

        # Line options
        self.x_widget = widgets.IntSlider(
            min=self.pad, step=1, description='', layout=layout
        )
        self.y_widget = widgets.IntSlider(
            min=self.pad, step=1, description='', layout=layout
        )
        self.angle_widget = widgets.IntSlider(
            min=0, max=360, step=1, value=0, description='', layout=layout
        )
        self.line_fraction_widget = widgets.FloatRangeSlider(
            min=0, max=1, step=0.01, value=[0, 1], description='', layout=layout
        )

        # Slice options
        self.slice_axis_widget = widgets.Dropdown(
            options=[0, 1, 2], value=self.slice_axis, description='Slice axis'
        )
        self.slice_axis_widget.layout.width = '250px'

        self.slice_index_widget = widgets.IntSlider(
            min=0, step=1, description='Slice index', layout=layout
        )
        self.slice_index_widget.layout.width = '400px'

        # y-limit
        self.ylim_style_widget = widgets.Dropdown(
            options=['auto', 'full', 'manual'],
            value=self.ylim_style,
            description='y-limit style',
        )

        num_steps = 30
        self.ymin_widget = widgets.FloatText(
            description='y-min',
            value=self.ylim[0],
            layout=widgets.Layout(width='150px'),
            step=self.data_span / num_steps,
        )
        self.ymax_widget = widgets.FloatText(
            description='y-max',
            value=self.ylim[1],
            layout=widgets.Layout(width='150px'),
            step=self.data_span / num_steps,
        )
        self.ylim_widget = widgets.HBox([self.ymin_widget, self.ymax_widget])
        self.ylim_widget.layout = widgets.Layout(width='310px')
        if self.ylim_style != 'manual':
            self.ylim_widget.layout.display = 'none'

    def calculate_line_endpoints(
        self, x: float, y: float, angle: float
    ) -> tuple[list[float], list[float]]:
        """Line is parameterized as: [x + t*np.cos(angle), y + t*np.sin(angle)]."""
        if np.isclose(angle, 0):
            return [0, y], [self.x_max, y]
        elif np.isclose(angle, np.pi / 2):
            return [x, 0], [x, self.y_max]
        elif np.isclose(angle, np.pi):
            return [self.x_max, y], [0, y]
        elif np.isclose(angle, 3 * np.pi / 2):
            return [x, self.y_max], [x, 0]
        elif np.isclose(angle, 2 * np.pi):
            return [0, y], [self.x_max, y]

        t_left = -x / np.cos(angle)
        t_bottom = -y / np.sin(angle)
        t_right = (self.x_max - x) / np.cos(angle)
        t_top = (self.y_max - y) / np.sin(angle)
        t_values = np.array([t_left, t_top, t_right, t_bottom])
        t_pos = np.min(t_values[t_values > 0])
        t_neg = np.max(t_values[t_values < 0])

        src = [x + t_neg * np.cos(angle), y + t_neg * np.sin(angle)]
        dst = [x + t_pos * np.cos(angle), y + t_pos * np.sin(angle)]
        return src, dst

    def update(
        self,
        slice_axis: int,
        slice_index: int,
        x: int,
        y: int,
        angle_deg: float,
        fraction_range: tuple[float, float],
        ylim_style: str,
        ymin: float,
        ymax: float,
    ) -> None:
        if slice_axis != self.slice_axis:
            self.update_slice_axis(slice_axis)
            x = self.x_widget.value
            y = self.y_widget.value
            slice_index = self.slice_index_widget.value

        self.ylim[0] = ymin
        self.ylim[1] = ymax

        clear_output(wait=True)

        image = np.take(self.volume, slice_index, slice_axis)
        angle = np.radians(angle_deg)
        src, dst = (
            np.array(point, dtype='float32')
            for point in self.calculate_line_endpoints(x, y, angle)
        )

        # Rescale endpoints
        line_vec = dst - src
        dst = src + fraction_range[1] * line_vec
        src = src + fraction_range[0] * line_vec

        y_pline = skimage.measure.profile_line(image, src, dst)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Image with color-gradiented line
        num_segments = 100
        x_seg = np.linspace(src[0], dst[0], num_segments)
        y_seg = np.linspace(src[1], dst[1], num_segments)
        segments = np.stack(
            [
                np.column_stack([y_seg[:-2], x_seg[:-2]]),
                np.column_stack([y_seg[2:], x_seg[2:]]),
            ],
            axis=1,
        )
        norm = plt.Normalize(vmin=0, vmax=num_segments - 1)
        colors = self.cmap(norm(np.arange(num_segments - 1)))
        lc = matplotlib.collections.LineCollection(segments, colors=colors, linewidth=2)

        ax[0].imshow(image, cmap='gray')
        ax[0].add_collection(lc)
        # pivot point
        ax[0].plot(y, x, marker='s', linestyle='', color='cyan', markersize=4)
        ax[0].set_xlabel(f'axis {np.delete(np.arange(3), self.slice_axis)[1]}')
        ax[0].set_ylabel(f'axis {np.delete(np.arange(3), self.slice_axis)[0]}')

        # Profile intensity plot
        norm = plt.Normalize(0, vmax=len(y_pline) - 1)
        x_pline = np.arange(len(y_pline))
        points = np.column_stack((x_pline, y_pline))[:, np.newaxis, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = matplotlib.collections.LineCollection(
            segments, cmap=self.cmap, norm=norm, array=x_pline[:-1], linewidth=2
        )

        ax[1].add_collection(lc)
        ax[1].autoscale()
        self.update_ylim(ax[1], ylim_style)
        ax[1].set_xlabel('Distance along line')
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()

    def build_interactive(self) -> widgets.VBox:
        # Group widgets into two columns
        title_style = (
            'text-align:center; font-size:16px; font-weight:bold; margin-bottom:5px;'
        )
        title_column1 = widgets.HTML(
            f"<div style='{title_style}'>Line parameterization</div>"
        )
        title_column2 = widgets.HTML(
            f"<div style='{title_style}'>Slice selection & plot options</div>"
        )

        # Make label widgets instead of descriptions which have different lengths.
        label_layout = widgets.Layout(width='120px')
        label_x = widgets.Label('Vertical position', layout=label_layout)
        label_y = widgets.Label('Horizontal position', layout=label_layout)
        label_angle = widgets.Label('Angle (°)', layout=label_layout)
        label_fraction = widgets.Label('Fraction range', layout=label_layout)

        row_x = widgets.HBox([label_x, self.x_widget])
        row_y = widgets.HBox([label_y, self.y_widget])
        row_angle = widgets.HBox([label_angle, self.angle_widget])
        row_fraction = widgets.HBox([label_fraction, self.line_fraction_widget])

        controls_column1 = widgets.VBox(
            [title_column1, row_x, row_y, row_angle, row_fraction]
        )
        controls_column2 = widgets.VBox(
            [
                title_column2,
                self.slice_axis_widget,
                self.slice_index_widget,
                self.ylim_style_widget,
                self.ylim_widget,
            ]
        )
        controls = widgets.HBox([controls_column1, controls_column2])

        interactive_plot = widgets.interactive_output(
            self.update,
            {
                'slice_axis': self.slice_axis_widget,
                'slice_index': self.slice_index_widget,
                'x': self.x_widget,
                'y': self.y_widget,
                'angle_deg': self.angle_widget,
                'fraction_range': self.line_fraction_widget,
                'ylim_style': self.ylim_style_widget,
                'ymin': self.ymin_widget,
                'ymax': self.ymax_widget,
            },
        )

        return widgets.VBox([controls, interactive_plot])


@coarseness('volume')
def line_profile(
    volume: np.ndarray,
    slice_axis: int = 0,
    slice_index: int | str = 'middle',
    vertical_position: int | str = 'middle',
    horizontal_position: int | str = 'middle',
    angle: int = 0,
    fraction_range: tuple[float, float] = (0.00, 1.00),
    y_limits: str | tuple[float, float] = 'auto',
) -> widgets.interactive:
    """
    Returns an interactive widget for visualizing the intensity profiles of lines on slices.

    Args:
        volume (np.ndarray): The 3D volume of interest.
        slice_axis (int, optional): Specifies the initial axis along which to slice.
        slice_index (int or str, optional): Specifies the initial slice index along slice_axis.
        vertical_position (int or str, optional): Specifies the initial vertical position of the line's pivot point.
        horizontal_position (int or str, optional): Specifies the initial horizontal position of the line's pivot point.
        angle (int or float, optional): Specifies the initial angle (°) of the line around the pivot point. A float will be converted to an int. A value outside the range will be wrapped modulo.
        fraction_range (tuple or list, optional): Specifies the fraction of the line segment to use from border to border. Both the start and the end should be in the range [0.0, 1.0].
        y_limits (str or tuple or list, optional): Specifies the behaviour of the limits on the y-axis of the intensity value plot. Option 'full' fixes to the volume's data range. Option 'auto' automatically adapts to the intensities on the current line. A manual range can be specified by passing a tuple or list of length 2. Defaults to 'auto'.

    Returns:
        widget (widgets.widget_box.VBox): The interactive widget.


    Example:
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.line_profile(vol)
        ```
        ![viz histogram](../../assets/screenshots/viz-line_profile.gif)

    """

    def parse_position(
        pos: int | str,
        pos_range: tuple[int, int],
        name: str,
    ) -> int:
        if isinstance(pos, int):
            if not pos_range[0] <= pos < pos_range[1]:
                msg = (
                    f'Value for {name} must be inside [{pos_range[0]}, {pos_range[1]}]'
                )
                raise ValueError(msg)
            return pos
        elif isinstance(pos, str):
            pos = pos.lower()
            if pos == 'start':
                return pos_range[0]
            elif pos == 'middle':
                return pos_range[0] + (pos_range[1] - pos_range[0]) // 2
            elif pos == 'end':
                return pos_range[1]
            else:
                msg = (
                    f"Invalid string '{pos}' for {name}. "
                    "Must be 'start', 'middle', or 'end'."
                )
                raise ValueError(msg)
        else:
            msg = 'Axis position must be of type int or str.'
            raise TypeError(msg)

    if not isinstance(volume, np.ndarray | da.Array):
        msg = 'Data type for volume not supported.'
        raise ValueError(msg)
    if volume.ndim != 3:
        msg = 'Volume must be 3D.'
        raise ValueError(msg)

    dims = volume.shape
    slice_index = parse_position(slice_index, (0, dims[slice_axis] - 1), 'slice_index')
    # the omission of the ends for the pivot point is due to border issues.
    vertical_position = parse_position(
        vertical_position, (1, np.delete(dims, slice_axis)[0] - 2), 'vertical_position'
    )
    horizontal_position = parse_position(
        horizontal_position,
        (1, np.delete(dims, slice_axis)[1] - 2),
        'horizontal_position',
    )

    if not isinstance(angle, float | int):
        msg = 'Invalid type for angle.'
        raise ValueError(msg)
    angle = round(angle) % 360

    if not (
        0.0 <= fraction_range[0] <= 1.0
        and 0.0 <= fraction_range[1] <= 1.0
        and fraction_range[0] <= fraction_range[1]
    ):
        msg = 'Invalid values for fraction_range.'
        raise ValueError(msg)

    if isinstance(y_limits, str):
        if y_limits not in ['auto', 'full']:
            msg = 'Invalid string value for y_limits.'
            raise ValueError(msg)
    else:
        y_limits = [*y_limits]

    lp = _LineProfile(
        volume,
        slice_axis,
        slice_index,
        vertical_position,
        horizontal_position,
        angle,
        fraction_range,
        y_limits,
    )
    return lp.build_interactive()


@coarseness('volume')
def threshold(
    volume: np.ndarray,
    colormap: str = 'magma',
    min_value: float = None,
    max_value: float = None,
) -> widgets.VBox:
    """
    An interactive interface to explore thresholding on a
    3D volume slice-by-slice. Users can either manually set the threshold value
    using a slider or select an automatic thresholding method from `skimage`.

    The visualization includes the original image slice, a binary mask showing regions above the
    threshold and an overlay combining the binary mask and the original image.

    Args:
        volume (np.ndarray): 3D volume to threshold.
        colormap (str, optional): Colormap for the original image. Defaults to 'viridis'.
        min_value (float, optional): Minimum value for the colormap. Defaults to None.
        max_value (float, optional): Maximum value for the colormap. Defaults to None.

    Returns:
        slicer_obj (widgets.VBox): The interactive widget for thresholding a 3D volume.

    Interactivity:
        - **Manual Thresholding**:
            Select 'Manual' from the dropdown menu to manually adjust the threshold
            using the slider.
        - **Automatic Thresholding**:
            Choose a method from the dropdown menu to apply an automatic thresholding
            algorithm. Available methods include:
            - Otsu
            - Isodata
            - Li
            - Mean
            - Minimum
            - Triangle
            - Yen

            The threshold slider will display the computed value and will be disabled
            in this mode.


        ```python
        import qim3d

        # Load a sample volume
        vol = qim3d.examples.bone_128x128x128

        # Visualize interactive thresholding
        qim3d.viz.threshold(vol)
        ```
        ![interactive threshold](../../assets/screenshots/interactive_thresholding.gif)

    """

    # Centralized state dictionary to track current parameters
    state = {
        'position': volume.shape[0] // 2,
        'method': 'Manual',
    }

    if np.issubdtype(volume.dtype, np.integer):
        step = 1
        state['threshold'] = int((volume.min() + volume.max()) / 2)
    elif np.issubdtype(volume.dtype, np.floating):
        step = (volume.max() - volume.min())/1000
        state['threshold'] = (volume.min() + volume.max()) / 2
    else:
        pass

    threshold_methods = {
        'Otsu': threshold_otsu,
        'Isodata': threshold_isodata,
        'Li': threshold_li,
        'Mean': threshold_mean,
        'Minimum': threshold_minimum,
        'Triangle': threshold_triangle,
        'Yen': threshold_yen,
    }

    # Create an output widget to display the plot
    output = widgets.Output()

    # Function to update the state and trigger visualization
    def update_state(change: dict[str, Any]) -> None:
        # Update state based on widget values
        state['position'] = position_slider.value
        state['method'] = method_dropdown.value

        if state['method'] == 'Manual':
            state['threshold'] = threshold_slider.value
            threshold_slider.disabled = False
        else:
            threshold_func = threshold_methods.get(state['method'])
            if threshold_func:
                slice_img = volume[state['position'], :, :]
                computed_threshold = threshold_func(slice_img)
                state['threshold'] = computed_threshold

                # Programmatically update the slider without triggering callbacks
                threshold_slider.unobserve_all()
                threshold_slider.value = computed_threshold
                threshold_slider.disabled = True
                threshold_slider.observe(update_state, names='value')
            else:
                msg = f"Unsupported thresholding method: {state['method']}"
                raise ValueError(msg)

        # Trigger visualization
        update_visualization()

    # Visualization function
    def update_visualization() -> None:
        slice_img = np.take(volume, state['position'], axis = slice_axis)
        with output:
            output.clear_output(wait=True)  # Clear previous plot
            fig, axes = plt.subplots(1, 4, figsize=(25, 5))

            # Original image
            new_min_value = (
                None
                if (isinstance(min_value, float | int) and min_value > np.max(slice_img))
                else min_value
            )
            new_max_value = (
                None
                if (isinstance(max_value, float | int) and max_value < np.min(slice_img))
                else max_value
            )
            axes[0].imshow(slice_img, cmap=colormap, vmin=new_min_value, vmax=new_max_value)
            axes[0].set_title('Original')
            axes[0].axis('off')

            # Histogram
            histogram(
                volume=volume,
                bins=32,
                slice_index=state['position'],
                vertical_line=state['threshold'],
                kde=False,
                ax=axes[1],
                show=False,
            )
            thr = state['threshold']
            if isinstance(step, float):
                thr = f'{thr:.3f}'
            else:
                thr = int(thr)
            axes[1].set_title(f"Histogram with Threshold = {thr}")

            # Binary mask
            mask = slice_img >= state['threshold']
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('Binary mask')
            axes[2].axis('off')

            # Overlay
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[:, :, 0] = mask
            masked_volume = qim3d.operations.overlay_rgb_images(
                background=slice_img,
                foreground=mask_rgb,
            )
            axes[3].imshow(masked_volume, vmin=new_min_value, vmax=new_max_value)
            axes[3].set_title('Overlay')
            axes[3].axis('off')

            plt.show()

    # Widgets
    position_slider = widgets.IntSlider(
        value=state['position'],
        min=0,
        max=volume.shape[0] - 1,
        description='Slice',
    )

    threshold_slider = widgets.FloatSlider(
        value=state['threshold'],
        min=volume.min(),
        max=volume.max(),
        step = step,
        description='Threshold',
        readout_format = 'd' if step == 1 else '.3f'
    )

    method_dropdown = widgets.Dropdown(
        options=[
            'Manual',
            'Otsu',
            'Isodata',
            'Li',
            'Mean',
            'Minimum',
            'Triangle',
            'Yen',
        ],
        value=state['method'],
        description='Method',
    )

    # Attach the state update function to widgets
    position_slider.observe(update_state, names='value')
    threshold_slider.observe(update_state, names='value')
    method_dropdown.observe(update_state, names='value')

    # Layout
    controls_left = widgets.VBox([position_slider, threshold_slider])
    controls_right = widgets.VBox([method_dropdown])
    controls_layout = widgets.HBox(
        [controls_left, controls_right],
        layout=widgets.Layout(justify_content='flex-start'),
    )
    interactive_ui = widgets.VBox([controls_layout, output])
    update_visualization()

    return interactive_ui


class _VolumeComparison:
    def __init__(
        self,
        volume1: np.ndarray | da.Array,
        volume2: np.ndarray | da.Array,
        slice_axis: int,
        slice_index: int,
        k3d: bool = False,
    ) -> None:
        self.volume1 = volume1
        self.volume2 = volume2
        self.slice_axis = slice_axis
        self.slice_index = slice_index
        self.k3d = k3d

        self.k3d_output1 = widgets.Output()
        self.k3d_output2 = widgets.Output()
        self.k3d_output3 = widgets.Output()
        self.plt_output1 = widgets.Output()
        self.plt_output2 = widgets.Output()
        self.plt_output3 = widgets.Output()

        self.update_comp_plot = False
        self.update_plots = False
        self.comparison_type = 'difference'  # Default comparison type

        self.create_colormap()
        self.initialize_widgets()
        self.update_slice_axis(slice_axis)
        self.slice_index_widget.value = slice_index
        if self.k3d:
            self.initialize_k3d_plots()

    def create_colormap(self) -> None:
        # Combine Blues and Reds colormaps
        blues = plt.cm.Blues_r(np.linspace(0.0, 1, 256))
        reds = plt.cm.Reds(np.linspace(0.0, 1, 256))
        colors = np.vstack((blues, reds))
        self.diff_cmap = LinearSegmentedColormap.from_list('blue_red', colors)

    def initialize_k3d_plots(self) -> None:
        self.k3d_plot1 = qim3d.viz.volumetric(
            self.volume1, show=False, colormap='Reds'
        )
        self.k3d_plot2 = qim3d.viz.volumetric(
            self.volume2, show=False, colormap='Blues'
        )
        alpha = [
            [0.0, 1.0],
            [0.35, 0.0],
            [0.65, 0.0],
            [1, 1.0],
        ]
        self.k3d_plot3 = qim3d.viz.volumetric(
            (self.volume1 - self.volume2),
            show=False,
            colormap=self.diff_cmap,
            opacity_function=alpha,
        )

    def update_slice_axis(self, slice_axis: int) -> None:
        self.slice_axis = slice_axis
        self.slice_index_widget.max = self.volume1.shape[slice_axis] - 1
        self.slice_index_widget.value = self.volume1.shape[slice_axis] // 2

    def initialize_widgets(self) -> None:
        layout = widgets.Layout(width='300px', height='auto')
        self.color_range_widget = widgets.FloatRangeSlider(
            min=0, max=1, step=0.01, value=[0, 1], layout=layout
        )
        self.comparison_type_widget = widgets.Dropdown(
            options=['difference', 'absolute difference', 'quadratic difference'],
            value=self.comparison_type,
        )

        # Slice related
        self.slice_axis_widget = widgets.Dropdown(
            options=[0, 1, 2], value=self.slice_axis, description='Slice axis'
        )
        self.slice_axis_widget.layout.width = '250px'

        self.slice_index_widget = widgets.IntSlider(
            min=0, step=1, description='Slice index', layout=layout
        )
        self.slice_index_widget.layout.width = '400px'

    def slice_cmap(
        self, colormap: str | LinearSegmentedColormap, color_range: tuple[float, float]
    ) -> matplotlib.colors.ListedColormap:
        if isinstance(colormap, str):
            colormap = matplotlib.colormaps[colormap].resampled(256)
        black = np.array([0, 0, 0, 1])
        sampled_colors = colormap(np.linspace(0, 1, 256))
        sampled_colors[: round(color_range[0] * 256), :] = black
        sampled_colors[round(color_range[1] * 256) :, :] = black
        newcmp = matplotlib.colors.ListedColormap(sampled_colors)
        return newcmp

    def update(
        self,
        slice_axis: int,
        slice_index: int,
        comparison_type: str,
        color_range: tuple[float, float],
    ) -> None:
        if slice_axis != self.slice_axis:
            self.update_slice_axis(slice_axis)
            slice_index = self.slice_index_widget.value

        slice1 = np.take(self.volume1, slice_index, axis=slice_axis).astype(float)
        slice2 = np.take(self.volume2, slice_index, axis=slice_axis).astype(float)

        norm1 = matplotlib.colors.Normalize(
            vmin=min(slice1.min(), slice2.min()), vmax=max(slice1.max(), slice2.max())
        )

        if comparison_type == 'difference':
            newcmp = self.slice_cmap(self.diff_cmap, color_range)
            cmap1 = self.slice_cmap('Reds', color_range)
            cmap2 = self.slice_cmap('Blues', color_range)

            comparison = slice1 - slice2
            vrange = [comparison.min(), comparison.max()]
            # In the special cases add small epsilon since TwoSlopeNorm requires vmin, vcenter, vmax to be in strictly ascending order.
            eps = 1e-8
            norm2 = matplotlib.colors.TwoSlopeNorm(
                vmin=min(0.0 - eps, vrange[0]),
                vcenter=0.0,
                vmax=max(0.0 + eps, vrange[1]),
            )
            # Create opacity function for k3d comparison plot
            alpha = [
                [0.0, 1.0],
                [0.35, 0.0],
                [0.65, 0.0],
                [1, 1.0],
            ]
            # Create comparison k3d plot
            if self.comparison_type != comparison_type and self.k3d:
                comparison_k3d = self.volume1 - self.volume2

        else:
            newcmp = self.slice_cmap('magma', color_range)
            cmap1 = newcmp
            cmap2 = newcmp
            alpha = []

            if comparison_type == 'absolute difference':
                comparison = np.abs(slice1 - slice2)
                if self.comparison_type != comparison_type and self.k3d:
                    comparison_k3d = np.abs(self.volume1 - self.volume2)

            elif comparison_type == 'quadratic difference':
                comparison = (slice1 - slice2) ** 2
                if self.comparison_type != comparison_type and self.k3d:
                    comparison_k3d = (self.volume1 - self.volume2) ** 2

            vrange = [0.0, comparison.max()]
            norm2 = matplotlib.colors.Normalize(vmin=vrange[0], vmax=vrange[1])

        if self.comparison_type != comparison_type and self.k3d:
            # Update difference k3d plot
            self.k3d_plot3 = qim3d.viz.volumetric(
                comparison_k3d,
                show=False,
                colormap=self.diff_cmap
                if comparison_type == 'difference'
                else 'magma',
                opacity_function=alpha,
            )
            self.update_comp_plot = True

            if (
                comparison_type == 'difference'
                or (
                    'quadratic' in comparison_type
                    and 'absolute' not in self.comparison_type
                )
                or (
                    'absolute' in comparison_type
                    and 'quadratic' not in self.comparison_type
                )
            ):
                # Update k3d plots 1 and 2 if colormap change is necessary (difference <-> quadratic/absolute)
                self.k3d_plot1 = qim3d.viz.volumetric(
                    self.volume1,
                    show=False,
                    colormap='Blues' if comparison_type == 'difference' else 'magma',
                )
                self.k3d_plot2 = qim3d.viz.volumetric(
                    self.volume2,
                    show=False,
                    colormap='Reds' if comparison_type == 'difference' else 'magma',
                )
                self.update_plots = True

        # Overwrite the comparison type in instance
        self.comparison_type = comparison_type

        # Create plots
        fig_1, ax_1 = plt.subplots(figsize=(4, 4))
        im1 = ax_1.imshow(slice1, norm=norm1, cmap=cmap1)
        ax_1.set_title('Volume1')
        divider1 = make_axes_locatable(ax_1)
        cax1 = divider1.append_axes('bottom', size='5%', pad=0.3)
        fig_1.colorbar(im1, cax=cax1, orientation='horizontal')
        fig_1.tight_layout()
        self.fig1 = fig_1
        plt.close(fig_1)

        fig_2, ax_2 = plt.subplots(figsize=(4, 4))
        im2 = ax_2.imshow(slice2, norm=norm1, cmap=cmap2)
        ax_2.set_title('Volume2')
        divider2 = make_axes_locatable(ax_2)
        cax2 = divider2.append_axes('bottom', size='5%', pad=0.3)
        fig_2.colorbar(im2, cax=cax2, orientation='horizontal')
        fig_2.tight_layout()
        self.fig2 = fig_2
        plt.close(fig_2)

        fig_3, ax_3 = plt.subplots(figsize=(4, 4))
        im3 = ax_3.imshow(comparison, norm=norm2, cmap=newcmp)
        ax_3.set_title(comparison_type)
        divider3 = make_axes_locatable(ax_3)
        cax3 = divider3.append_axes('bottom', size='5%', pad=0.3)
        fig_3.colorbar(im3, cax=cax3, orientation='horizontal')
        fig_3.tight_layout()
        self.fig3 = fig_3
        plt.close(fig_3)

        # Visualize plt plot 1
        self.plt_output1.clear_output(wait=True)
        with self.plt_output1:
            plt.figure(self.fig1)
            plt.show()

        # Visualize plt plot 2
        self.plt_output2.clear_output(wait=True)
        with self.plt_output2:
            plt.figure(self.fig2)
            plt.show()

        # Visualize plt plot 3
        self.plt_output3.clear_output(wait=True)
        with self.plt_output3:
            plt.figure(self.fig3)
            plt.show()

        if self.update_plots:
            # Visualize k3d plot 1 (if there are changes)
            self.k3d_output1.clear_output(wait=True)
            with self.k3d_output1:
                display(self.k3d_plot1)
            # Visualize k3d plot 1 (if there are changes)
            self.k3d_output2.clear_output(wait=True)
            with self.k3d_output2:
                display(self.k3d_plot2)
            self.update_plots = False

        # Visualize k3d plot 3 (if there are changes)
        if self.update_comp_plot:
            self.k3d_output3.clear_output(wait=True)
            with self.k3d_output3:
                display(self.k3d_plot3)
            self.update_comp_plot = False

    def build_interactive(self) -> widgets.VBox:
        # Group widgets into two columns
        title_style = (
            'text-align:center; font-size:16px; font-weight:bold; margin-bottom:5px;'
        )
        title_column1 = widgets.HTML(
            f"<div style='{title_style}'>Comparison options</div>"
        )
        title_column2 = widgets.HTML(
            f"<div style='{title_style}'>Slice selection and y-limit options</div>"
        )

        # Make label widgets instead of descriptions which have different lengths.
        label_layout = widgets.Layout(width='120px')
        label_comparison_type = widgets.Label('Comparison type', layout=label_layout)
        label_color_range = widgets.Label('Color range fraction', layout=label_layout)

        row_comparison_type = widgets.HBox(
            [label_comparison_type, self.comparison_type_widget]
        )
        row_color_range = widgets.HBox([label_color_range, self.color_range_widget])

        controls_column1 = widgets.VBox(
            [title_column1, row_comparison_type, row_color_range]
        )
        controls_column2 = widgets.VBox(
            [title_column2, self.slice_axis_widget, self.slice_index_widget]
        )
        controls = widgets.HBox([controls_column2, controls_column1])

        interactive_plot = widgets.interactive_output(
            self.update,
            {
                'slice_axis': self.slice_axis_widget,
                'slice_index': self.slice_index_widget,
                'comparison_type': self.comparison_type_widget,
                'color_range': self.color_range_widget,
            },
        )

        # Create height on plt outputs to prevent flickering
        plt_layout = widgets.Layout(
            height='400px',
            width='99%',
            overflow='hidden',
        )

        fig_layout = widgets.Layout(
            width='400px',
            height='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch',
            justify_content='space-between',
        )
        self.plt_output1.layout = plt_layout
        self.plt_output2.layout = plt_layout
        self.plt_output3.layout = plt_layout

        if self.k3d:
            k3d_layout = widgets.Layout(
                width='99%',  # Slightly smaller
                height='400px',
                overflow='hidden',
                flex='1 1 0%',
            )

            self.k3d_output1.layout = k3d_layout
            self.k3d_output2.layout = k3d_layout
            self.k3d_output3.layout = k3d_layout

            fig1 = widgets.VBox(
                [self.plt_output1, self.k3d_output1],
                layout=fig_layout,
            )
            fig2 = widgets.VBox(
                [self.plt_output2, self.k3d_output2],
                layout=fig_layout,
            )
            fig3 = widgets.VBox(
                [self.plt_output3, self.k3d_output3],
                layout=fig_layout,
            )

            with self.k3d_output1:
                display(self.k3d_plot1)
            with self.k3d_output2:
                display(self.k3d_plot2)
            with self.k3d_output3:
                display(self.k3d_plot3)
        else:  # if volumetric visualization should not  be shown
            fig1 = widgets.VBox(
                [self.plt_output1],
                layout=fig_layout,
            )
            fig2 = widgets.VBox(
                [self.plt_output2],
                layout=fig_layout,
            )
            fig3 = widgets.VBox(
                [self.plt_output3],
                layout=fig_layout,
            )

        # Update visualization and return the widgets
        self.plt_output1.clear_output(wait=True)
        self.plt_output2.clear_output(wait=True)
        self.plt_output3.clear_output(wait=True)
        with self.plt_output1:
            plt.figure(self.fig1)
            plt.show()
        with self.plt_output2:
            plt.figure(self.fig2)
            plt.show()
        with self.plt_output3:
            plt.figure(self.fig3)
            plt.show()

        figs = widgets.HBox([fig1, fig2, fig3])
        return widgets.VBox([controls, interactive_plot, figs])


@coarseness('volume1', 'volume2')
def compare_volumes(
    volume1: np.ndarray,
    volume2: np.ndarray,
    slice_axis: int = 0,
    slice_index: int = None,
    volumetric_visualization: bool = False,
) -> widgets.interactive:
    """
    Returns an interactive widget for comparing two volumes along slices.

    Args:
        volume1 (np.ndarray): The first volume.
        volume2 (np.ndarray): The second volume.
        slice_axis (int, optional): Specifies the initial axis along which to slice.
        slice_index (int, optional): Specifies the initial index along slice_axis.
        volumetric_visualization (bool, optional): Defines if k3d plots should also be shown.

    Returns:
        widget (widgets.widget_box.VBox): The interactive widget.



    Example:
        ```python
        import qim3d

        vol1 = qim3d.generate.volume(noise_scale=0.020, dtype='float32')
        vol2 = qim3d.generate.volume(noise_scale=0.021, dtype='float32')

        qim3d.viz.compare_volumes(vol1, vol2, volumetric_visualization=True)

        ```
        ![volume_comparison](../../assets/screenshots/viz-compare_volumes.png)

    """

    if volume1.ndim != 3:
        msg = 'Volume must be 3D.'
        raise ValueError(msg)
    if volume1.shape != volume2.shape:
        msg = 'Volumes must have the same shape.'
        raise ValueError(msg)

    if np.issubdtype(volume1.dtype, np.unsignedinteger) and np.issubdtype(
        volume2.dtype, np.unsignedinteger
    ):
        log.warning(
            'Volumes have unsigned integer datatypes. Beware of over-/underflow.'
        )

    if slice_axis not in (0, 1, 2):
        msg = 'Invalid slice_axis.'
        raise ValueError(msg)

    if slice_index is None:
        slice_index = volume1.shape[slice_axis] // 2
    if not isinstance(slice_index, int):
        msg = 'slice_index must be an integer.'
        raise ValueError(msg)

    vc = _VolumeComparison(
        volume1, volume2, slice_axis, slice_index, volumetric_visualization
    )
    return vc.build_interactive()


class IsoSurface:
    def __init__(self, volume: np.ndarray, colormap: str = 'magma') -> None:
        # keep a float32 copy to save half the RAM up front
        self.vol_full = np.transpose(volume, (1, 2, 0)).astype(np.float32)
        self.min_value = self.vol_full.min()
        self.max_value = self.vol_full.max()
        self.cmap = colormap
        self._resolution_cache = {}

        self.out = widgets.Output()
        self._build_widgets()
        self._init_figure()  # FigureWidget – persistent
        self._display_ui()

    # ---------- widgets ----------
    def _build_widgets(self) -> None:
        self.thr = widgets.IntSlider(
            value=int(self.max_value / 2),
            min=self.min_value,
            max=self.max_value,
            step=1,
            description='Threshold',
            continuous_update=False,
        )
        self.resolution = widgets.IntSlider(
            value=64,
            min=32,
            max=96,
            step=1,
            description='Resolution',
            continuous_update=False,
        )
        self.trans = widgets.FloatSlider(
            value=0,
            min=0,
            max=1,
            step=0.1,
            description='Transparency',
            continuous_update=False,
        )
        self.cmapw = widgets.Dropdown(
            options=[
                'Blackbody',
                'Bluered',
                'Blues',
                'Cividis',
                'Earth',
                'Electric',
                'Greens',
                'Greys',
                'Hot',
                'Jet',
                'Magma',
                'Picnic',
                'Portland',
                'Rainbow',
                'RdBu',
                'Reds',
                'Viridis',
                'YlGnBu',
                'YlOrRd',
            ],
            value=self.cmap,
            description='Colormap',
        )
        self.grid = widgets.Checkbox(value=True, description='Grid')

        self.wireframe = widgets.Checkbox(value=False, description='Wireframe')

        self.colorbar = widgets.Checkbox(value=False, description='Colorbar')

        for w in (
            self.thr,
            self.resolution,
            self.trans,
            self.cmapw,
            self.grid,
            self.wireframe,
            self.colorbar,
        ):
            w.observe(self._refresh, names='value')

    # ---------- data prep ----------
    def _resize_vol(self, resolution: int) -> dict:
        original_z, original_y, original_x = np.shape(self.vol_full)
        max_size = max(original_z, original_y, original_x)

        # Compute uniform zoom factor to fit the target resolution
        zoom_factor = resolution / max_size

        # Resize the full volume
        vol_zoomed = ndimage.zoom(
            input=self.vol_full,
            zoom=zoom_factor,
            order=0,
            prefilter=False,
        )

        # Generate corresponding 3D grid
        x, y, z = vol_zoomed.shape
        xg, yg, zg = np.mgrid[0:x, 0:y, 0:z]

        # Cache the result with resolution as the key
        return (xg, yg, zg, vol_zoomed)

    # ---------- figure ----------
    def _init_figure(self) -> None:
        xg, yg, zg, v = self._resize_vol(resolution=self.resolution.value)
        isoval = self.thr.value  # * float(v.max())
        self.fig = go.FigureWidget(
            data=[
                go.Isosurface(
                    x=xg.flatten(),
                    y=yg.flatten(),
                    z=zg.flatten(),
                    cmin=v.min(),
                    cmax=v.max(),
                    value=v.flatten(),
                    isomin=isoval,
                    isomax=isoval,
                    opacity=1 - self.trans.value,
                    surface_count=1,
                    caps={'x_show': False, 'y_show': False, 'z_show': False},
                    showscale=self.colorbar.value,  # self.cbar.value,
                    colorscale=self.cmapw.value,
                )
            ]
        )
        self._layout_axes()

        self._last_resolution = self.resolution.value

    def _layout_axes(self) -> None:
        self.fig.update_layout(
            scene_aspectmode='data',
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
            scene={
                'xaxis': {'visible': self.grid.value},
                'yaxis': {'visible': self.grid.value},
                'zaxis': {'visible': self.grid.value},
            },
        )

    # ---------- redraw ----------
    def _refresh(self, *_) -> None:
        resolution = self.resolution.value
        tr = self.fig.data[0]

        surface_fill = 0.2 if self.wireframe.value else 1.0

        xg, yg, zg, v = self._resize_vol(resolution)
        isoval = self.thr.value  # * float(v.max())

        if self._last_resolution == resolution:
            # Only update visual parameters, not the volume data
            tr.update(
                isomin=isoval,
                isomax=isoval,
                surface={'fill': surface_fill},
            )
            tr.colorscale = self.cmapw.value
            tr.showscale = self.colorbar.value
            tr.opacity = 1 - self.trans.value
        else:
            # Update everything
            tr.update(
                x=xg.flatten(),
                y=yg.flatten(),
                z=zg.flatten(),
                value=v.flatten(),
                isomin=isoval,
                isomax=isoval,
                opacity=1 - self.trans.value,
                showscale=self.colorbar.value,
                colorscale=self.cmapw.value,
                surface={'fill': surface_fill},
            )

            self._last_resolution = resolution
        self._layout_axes()

    # ---------- UI ----------
    def _display_ui(self) -> None:
        title = widgets.HTML("<h3 style='margin-top:0;'>Iso-Surface Visualizer</h3>")

        controls = widgets.VBox(
            [
                title,
                self.thr,
                self.resolution,
                self.trans,
                self.cmapw,
                self.colorbar,
                self.grid,
                self.wireframe,
            ],
            layout=widgets.Layout(
                min_width='200px',
            ),
        )

        ui = widgets.HBox(
            [controls, self.fig], layout=widgets.Layout(width='100%', height='640px')
        )

        display(ui)


@coarseness('volume')
def iso_surface(volume: np.ndarray, colormap: str = 'Magma') -> None:
    """
    Creates an interactive iso-surface visualizer for a single surface level.

    Args:
        vol (np.ndarray): Volume to visualize an iso-surface of.
        colormap: (str, optional): Initial colormap for the iso-surface. This can be changed in the interface

    Example:
        ```python
        import qim3d

        vol = qim3d.generate.volume(noise_scale=0.020)
        qim3d.viz.iso_surface(vol)
        ```
        ![volume_comparison](../../assets/screenshots/iso_surface.gif)

    """
    IsoSurface(volume, colormap)


def _get_save_path(user_input: str, default_dir: str = '.') -> Path:
    input_path = Path(user_input)

    if input_path.is_absolute():
        return input_path
    else:
        return Path(default_dir) / input_path


def export_rotation(
    path: str,
    volume: np.ndarray,
    degrees: int = 360,
    n_frames: int = 180,
    fps: int = 30,
    image_size: tuple[int, int] | None = (256, 256),
    colormap: str = 'magma',
    camera_height: float = 2.0,
    camera_distance: float | str = 'auto',
    camera_focus: list | str = 'center',
    show: bool = False,
) -> None:
    """
    Export a rotation animation of volume.

    Args:
        path (str): The path to save the output. The path should end with .gif, .avi, .mp4 or .webm. If no file extension is specified, .gif is automatically added.
        volume (np.ndarray): Volume to create .gif of.
        degrees (int, optional): The amount of degrees for the volume to rotate. Defaults to 360.
        n_frames (int, optional): The amount of frames to generate. Defaults to 180.
        fps (int, optional): The amount of frames per second in the resulting animation. This determines the speed of the rotation of the volume. Defaults to 30.
        image_size (tuple of ints or None, optional): Pixel size (width, height) of each frame. If None, the plotter's default size is used. Defaults to (256, 256).
        colormap (str, optional): Determines color map of volume. Defaults to 'magma'.
        camera_height (float, optional): Determines the height of the camera rotating around the volume. The float value represents a multiple of the height of the z-axis. Defaults to 2.0.
        camera_distance (int or string, optional): Determines the distance of the camera from the center point. If 'auto' is used, it will be auto calculated. Otherwise a float value representing voxel distance is expected. Defaults to 'auto'.
        camera_focus (list or str, optional): Determines the voxel that the camera rotates around. Using 'center' will default to the center of the volume. Otherwise a list of three integers is expected. Defaults to 'center'.
        show (bool, optional): If True, the resulting animation will be shown in the Jupyter notebook. Defaults to False.

    Returns:
        None


    Raises:
        TypeError: If the camera focus argument is incorrectly used.
        TypeError: If the camera_distance argument is incorrectly used.
        ValueError: If the path contains an unrecognized file extension.

    Example:
        Creation of .gif file with default parameters of a generated volume.
        ```python
        import qim3d
        vol = qim3d.generate.volume()

        qim3d.viz.export_rotation('test.gif', vol, show=True)
        ```
        ![export_rotation_defaults](../../assets/screenshots/export_rotation_defaults.gif)

    Example:
        Creation of a .webm file with specified parameters of a generated volume in the shape of a tube.
        ```python
        import qim3d

        vol = qim3d.generate.volume(shape='tube')

        qim3d.viz.export_rotation('test.webm', vol,
                                  degrees = 360,
                                  n_frames = 120,
                                  fps = 30,
                                  image_size = (512,512),
                                  camera_height = 3.0,
                                  camera_distance = 'auto',
                                  camera_focus = 'center',
                                  show = True)
        ```
        ![export_rotation_video](../../assets/screenshots/export_rotation_video.gif)

    """
    if not (
        camera_focus == 'center'
        or (
            isinstance(camera_focus, list | np.ndarray)
            and not isinstance(camera_focus, str)
            and len(camera_focus) == 3
        )
    ):
        msg = f'Value "{camera_focus}" for camera focus is invalid. Use "center" or a list of three values.'
        raise TypeError(msg)
    if not (isinstance(camera_distance, float) or camera_distance == 'auto'):
        msg = f'Value "{camera_distance}" for camera distance is invalid. Use "auto" or a float value.'
        raise TypeError(msg)

    if Path(path).suffix == '':
        print(f'Input path: "{path}" does not have a filetype. Defaulting to .gif.')
        path += '.gif'

    # Handle img in (xyz) instead of (zyx) (due to rendering issues with the up-vector, ensure that z=y, such that we now have (x,z,y))
    vol = np.transpose(volume, (2, 0, 1))

    # Create a uniform grid
    grid = pv.ImageData()
    grid.dimensions = np.array(vol.shape) + 1  # PyVista dims are +1 from volume shape
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data['values'] = vol.flatten(order='F')  # Fortran order

    # Initialize plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(grid, opacity='linear', cmap=colormap)
    plotter.remove_scalar_bar()  # Remove colorbar

    frames = []
    camera_height = vol.shape[1] * camera_height

    if camera_distance == 'auto':
        bounds = np.array(plotter.bounds)  # (xmin, xmax, ymin, ymax, zmin, zmax)
        diag = np.linalg.norm(
            [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
        )
        camera_distance = diag * 2.0

    if camera_focus == 'center':
        _, center, _ = plotter.camera_position
    else:
        center = camera_focus

    center = np.array(center)

    angle_per_frame = degrees / n_frames
    radians_per_frame = np.radians(angle_per_frame)

    # Set up orbit radius and fixed up
    radius = camera_distance
    fixed_up = [0, 1, 0]
    for i in tqdm(range(n_frames), desc='Rendering'):
        theta = radians_per_frame * i
        x = radius * np.sin(theta)
        z = radius * np.cos(theta)
        y = camera_height  # fixed height

        eye = center + np.array([x, y, z])
        plotter.camera_position = [eye.tolist(), center.tolist(), fixed_up]

        plotter.render()
        img = plotter.screenshot(return_img=True, window_size=image_size)
        frames.append(img)

    if path[-4:] == '.gif':
        imageio.mimsave(path, frames, fps=fps, loop=0)

    elif path[-4:] == '.avi' or path[-4:] == '.mp4':
        writer = imageio.get_writer(path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    elif path[-5:] == '.webm':
        writer = imageio.get_writer(
            path, fps=fps, codec='vp9', ffmpeg_params=['-crf', '32']
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    else:
        msg = 'Invalid file extension. Please use .gif, .avi, .mp4 or .webm'
        raise ValueError(msg)

    path = _get_save_path(path)
    log.info('File saved to ' + str(path.resolve()))

    if show:
        if path.suffix == '.gif':
            display(Image(filename=path))
        elif path.suffix in ['.avi', '.mp4', '.webm']:
            display(Video(filename=path, html_attributes='controls autoplay loop'))


class VolumePlaneSlicer:
    @staticmethod
    def matplotlib_to_plotly_cmap(
        cmap: str | matplotlib.colors.Colormap,
    ) -> list[list[float, str]]:
        lin = np.linspace(0, 1, 256)
        rgbs = matplotlib.colormaps.get_cmap(cmap)(lin)[:, :3]
        colorscale = [
            [
                lin[i].item(),
                plotly.colors.label_rgb(plotly.colors.convert_to_RGB_255(rgbs[i])),
            ]
            for i in range(len(lin))
        ]
        return colorscale

    def __init__(
        self,
        volume: np.ndarray,
        colormap: str | matplotlib.colors.Colormap = 'magma',
        color_range: list[float | None, float | None] = None,
        showscale: bool = True,
        opacity: float = 1.0,
    ):
        self.volume = volume
        self.colormap = colormap
        self.initial_colorscale = self.matplotlib_to_plotly_cmap(colormap)
        self.showscale = showscale
        self.opacity = opacity
        self.continuous_update = self.volume.size <= 250**3  # only for small volumes

        self._last_indices = {}

        color_range = color_range if color_range else [None, None]
        vmin = color_range[0] or volume.min()
        vmax = color_range[1] or volume.max()
        self.color_range = [vmin, vmax]
        self.z_max, self.y_max, self.x_max = volume.shape

        self.x_axis = np.arange(self.x_max)
        self.y_axis = np.arange(self.y_max)
        self.z_axis = np.arange(self.z_max)

        # Precompute 2-D grids for Y and X planes to save time on each update
        self._x_grid_zx = np.repeat(self.x_axis[None, :], self.z_max, axis=0)  # (Z, X)
        self._z_grid_zx = np.repeat(self.z_axis[:, None], self.x_max, axis=1)  # (Z, X)
        # X plane (shape: z,y): z varies by row, y varies by col
        self._y_grid_zy = np.repeat(self.y_axis[None, :], self.z_max, axis=0)  # (Z, Y)
        self._z_grid_zy = np.repeat(self.z_axis[:, None], self.y_max, axis=1)  # (Z, Y)

        self.fig = go.FigureWidget()
        self._init_controls()
        self._init_surfaces()
        self._update_figure()
        self._set_observers()

    def _init_controls(self) -> None:
        # Slice controls
        slider_layout = widgets.Layout(width='400px')
        self.x_slider = widgets.IntSlider(
            value=self.x_max // 2,
            min=0,
            max=self.x_max - 1,
            description='X',
            layout=slider_layout,
            continuous_update=self.continuous_update,
        )
        self.y_slider = widgets.IntSlider(
            value=self.y_max // 2,
            min=0,
            max=self.y_max - 1,
            description='Y',
            layout=slider_layout,
            continuous_update=self.continuous_update,
        )
        self.z_slider = widgets.IntSlider(
            value=self.z_max // 2,
            min=0,
            max=self.z_max - 1,
            description='Z',
            layout=slider_layout,
            continuous_update=self.continuous_update,
        )

        checkbox_layout = widgets.Layout(width='20px')
        self.show_x = widgets.Checkbox(
            value=True, description='', indent=False, layout=checkbox_layout
        )
        self.show_y = widgets.Checkbox(
            value=True, description='', indent=False, layout=checkbox_layout
        )
        self.show_z = widgets.Checkbox(
            value=True, description='', indent=False, layout=checkbox_layout
        )

        hbox_layout = widgets.Layout(width='420px')
        z_controls = widgets.HBox([self.z_slider, self.show_z], layout=hbox_layout)
        y_controls = widgets.HBox([self.y_slider, self.show_y], layout=hbox_layout)
        x_controls = widgets.HBox([self.x_slider, self.show_x], layout=hbox_layout)
        slice_controls = widgets.VBox([z_controls, y_controls, x_controls])

        # Visual controls
        self.opacity_slider = widgets.FloatSlider(
            value=self.opacity,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Opacity',
            layout=widgets.Layout(width='350px'),
            continuous_update=self.continuous_update,
        )
        is_int = np.issubdtype(self.volume.dtype, np.integer)
        crange_slider_type = (
            widgets.IntRangeSlider if is_int else widgets.FloatRangeSlider
        )
        self.crange_slider = crange_slider_type(
            value=self.color_range,
            min=self.color_range[0],
            max=self.color_range[1],
            step=1 if is_int else (self.color_range[1] - self.color_range[0]) / 100,
            description='Color range',
            continuous_update=self.continuous_update,
            layout=widgets.Layout(width='400px'),
        )

        self.cmaps = [
            'Blues',
            'cividis',
            'cool',
            'gray',
            'Greys',
            'hot',
            'hsv',
            'inferno',
            'magma',
            'plasma',
            'spring',
            'viridis',
            'tab10',
            'turbo',
            'nipy_spectral',
        ]
        if isinstance(self.colormap, matplotlib.colors.Colormap):
            cmap_value = 'Custom'
            cmap_options = [cmap_value] + self.cmaps
        elif isinstance(self.colormap, str):
            if self.colormap in self.cmaps:
                cmap_value = self.colormap
                cmap_options = self.cmaps
            else:
                cmap_value = self.colormap
                cmap_options = [cmap_value] + self.cmaps

        self.cmap_dropdown = widgets.Dropdown(
            options=cmap_options,
            value=cmap_value,
            description='Colormap',
            layout=widgets.Layout(width='300px'),
        )

        visual_controls = widgets.VBox(
            [self.opacity_slider, self.crange_slider, self.cmap_dropdown],
            layout=widgets.Layout(width='450px'),
        )

        # Combined controls
        whitespace = widgets.Box(layout=widgets.Layout(width='100px'))
        self.controls = widgets.HBox([slice_controls, whitespace, visual_controls])

    def _init_surfaces(self) -> None:
        # --- Create three surfaces. Z uses 1-D x/y; X and Y use precomputed 2-D grids set once. ---
        # Z plane (shape y,x): use 1-D x/y axes, z is a 2-D constant plane updated per index
        z0 = self.z_slider.value
        z_plane = np.full((self.y_max, self.x_max), self.z_axis[z0], dtype=np.float32)

        # Y plane (shape z,x): x grid & z grid are static; y is a 2-D constant plane updated per index
        y0 = self.y_slider.value
        y_plane = np.full((self.z_max, self.x_max), self.y_axis[y0], dtype=np.float32)

        # X plane (shape z,y): y grid & z grid are static; x is a 2-D constant plane updated per index
        x0 = self.x_slider.value
        x_plane = np.full((self.z_max, self.y_max), self.x_axis[x0], dtype=np.float32)

        surfaces = [
            # Z plane
            go.Surface(
                name='Z',
                x=self.x_axis,  # 1-D
                y=self.y_axis,  # 1-D
                z=z_plane,  # 2-D constant (updated when Z changes)
                surfacecolor=np.zeros((self.y_max, self.x_max), dtype=np.float32),
                opacity=0,
                colorscale=self.initial_colorscale,
                showscale=False,
                cmin=self.color_range[0],
                cmax=self.color_range[1],
                showlegend=False,
            ),
            # Y plane (vertical): needs full 2-D x,y,z
            go.Surface(
                name='Y',
                x=self._x_grid_zx,  # 2-D static
                y=y_plane,  # 2-D constant (updated when Y changes)
                z=self._z_grid_zx,  # 2-D static
                surfacecolor=np.zeros((self.z_max, self.x_max), dtype=np.float32),
                opacity=0,
                colorscale=self.initial_colorscale,
                showscale=False,
                cmin=self.color_range[0],
                cmax=self.color_range[1],
                showlegend=False,
            ),
            # X plane (vertical): needs full 2-D x,y,z
            go.Surface(
                name='X',
                x=x_plane,  # 2-D constant (updated when X changes)
                y=self._y_grid_zy,  # 2-D static
                z=self._z_grid_zy,  # 2-D static
                surfacecolor=np.zeros((self.z_max, self.y_max), dtype=np.float32),
                opacity=0,
                colorscale=self.initial_colorscale,
                showscale=False,
                cmin=self.color_range[0],
                cmax=self.color_range[1],
                showlegend=False,
            ),
        ]
        self.fig.add_traces(surfaces)

        # dummy surface for the colorbar (unchanged)
        colorbar_surface = go.Surface(
            z=[[0, 0], [0, 0]],
            surfacecolor=[[0, 1], [0, 1]],
            opacity=0,
            colorscale=self.initial_colorscale,
            cmin=self.color_range[0],
            cmax=self.color_range[1],
            showscale=True,
            hoverinfo='skip',
            showlegend=False,
        )
        self.fig.add_trace(colorbar_surface)

        self.fig.update_layout(
            width=1000,
            height=500,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            scene={
                'xaxis': {'title': 'X', 'range': [0, self.x_max]},
                'yaxis': {'title': 'Y', 'range': [0, self.y_max]},
                'zaxis': {'title': 'Z', 'range': [0, self.z_max]},
                'aspectmode': 'manual',
                'aspectratio': {
                    axis: (size / max(self.volume.shape)) * 1.3
                    for axis, size in zip(
                        ['x', 'y', 'z'], [self.x_max, self.y_max, self.z_max]
                    )
                },
            },
            uirevision=True,  # preserve camera on updates
        )

        for tr in self.fig.data[:3]:
            tr.update(
                lighting={
                    'ambient': 1,
                    'diffuse': 0,
                    'specular': 0,
                    'roughness': 1,
                    'fresnel': 0,
                }
            )

        for tr in self.fig.data[:3]:
            tr.update(cmin=0, cmax=255)

    def _quantize_to_u8(self, arr: np.ndarray) -> np.ndarray:
        vmin, vmax = self.color_range
        # guard against degenerate range
        scale = 255.0 / max(float(vmax) - float(vmin), 1e-12)
        out = (arr.astype(np.float32, copy=False) - float(vmin)) * scale
        # clip and cast
        out = np.clip(out, 0.0, 255.0).astype(np.uint8, copy=False)
        return np.ascontiguousarray(out)

    def _update_plane(self, plane: Literal['X', 'Y', 'Z']) -> None:
        key = (
            ('Z', self.z_slider.value)
            if plane == 'Z'
            else ('Y', self.y_slider.value)
            if plane == 'Y'
            else ('X', self.x_slider.value)
        )
        if getattr(self, '_last_key', None) == key:
            return
        self._last_key = key

        with self.fig.batch_update():
            opacity = self.opacity_slider.value

            if plane == 'Z':
                k = self.z_slider.value
                data = np.take(self.volume, k, axis=0)
                if not data.flags['C_CONTIGUOUS']:  # ensure contiguous
                    data = np.ascontiguousarray(data)
                data = data.astype(np.float32, copy=False)
                s = self.fig.data[0]
                # Update constant Z-plane only if index changed
                if self._last_indices.get('k') != k:
                    s.z = np.full(
                        (self.y_max, self.x_max), self.z_axis[k], dtype=np.float32
                    )
                    self._last_indices['k'] = k
                q = self._quantize_to_u8(data)
                s.surfacecolor = q
                s.opacity = opacity

            elif plane == 'Y':
                j = self.y_slider.value
                data = np.take(self.volume, j, axis=1)
                if not data.flags['C_CONTIGUOUS']:  # ensure contiguous
                    data = np.ascontiguousarray(data)
                data = data.astype(np.float32, copy=False)
                s = self.fig.data[1]
                if self._last_indices.get('j') != j:
                    s.y = np.full(
                        (self.z_max, self.x_max), self.y_axis[j], dtype=np.float32
                    )
                    self._last_indices['j'] = j
                q = self._quantize_to_u8(data)
                s.surfacecolor = q
                s.opacity = opacity

            elif plane == 'X':
                i = self.x_slider.value
                data = np.take(self.volume, i, axis=2)
                if not data.flags['C_CONTIGUOUS']:  # ensure contiguous
                    data = np.ascontiguousarray(data)
                data = data.astype(np.float32, copy=False)
                s = self.fig.data[2]
                if self._last_indices.get('i') != i:
                    s.x = np.full(
                        (self.z_max, self.y_max), self.x_axis[i], dtype=np.float32
                    )
                    self._last_indices['i'] = i
                q = self._quantize_to_u8(data)
                s.surfacecolor = q
                s.opacity = opacity

            else:
                msg = f'Invalid plane: {plane}'
                raise ValueError(msg)

    def _toggle_visibility(self, plane: Literal['X', 'Y', 'Z']) -> None:
        if plane == 'X':
            self.fig.data[2].visible = self.show_x.value
        elif plane == 'Y':
            self.fig.data[1].visible = self.show_y.value
        elif plane == 'Z':
            self.fig.data[0].visible = self.show_z.value

    def _update_opacity(self) -> None:
        opacity = self.opacity_slider.value
        for surface in self.fig.data[:3]:
            surface.opacity = opacity

    def _update_crange(self) -> None:
        crange = self.crange_slider.value
        for surface in self.fig.data:
            surface.cmin = crange[0]
            surface.cmax = crange[1]

    def _update_cmap(self) -> None:
        cmap = self.cmap_dropdown.value
        if cmap == 'Custom':
            colorscale = self.initial_colorscale
        else:
            colorscale = self.matplotlib_to_plotly_cmap(cmap)

        for surface in self.fig.data:
            surface.colorscale = colorscale

    def _update_figure(self, change: dict = None) -> None:
        if change is None:
            for plane in ['X', 'Y', 'Z']:
                self._update_plane(plane)
            return

        owner = change['owner']
        owner_action_map = {
            self.x_slider: lambda: self._update_plane('X'),
            self.y_slider: lambda: self._update_plane('Y'),
            self.z_slider: lambda: self._update_plane('Z'),
            self.show_x: lambda: self._toggle_visibility('X'),
            self.show_y: lambda: self._toggle_visibility('Y'),
            self.show_z: lambda: self._toggle_visibility('Z'),
            self.opacity_slider: self._update_opacity,
            self.crange_slider: self._update_crange,
            self.cmap_dropdown: self._update_cmap,
        }

        try:
            owner_action_map[owner]()
        except KeyError as err:
            msg = f'Unhandled slider or control: {owner}'
            raise ValueError(msg) from err

    def _update_surface(
        self,
        surface: go.Surface,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        z_mesh: np.ndarray,
        surfacecolor: np.ndarray,
        opacity: float,
    ) -> None:
        surface.x = x_mesh
        surface.y = y_mesh
        surface.z = z_mesh
        surface.surfacecolor = surfacecolor
        surface.opacity = opacity

    def _set_observers(self) -> None:
        for control in [
            self.x_slider,
            self.y_slider,
            self.z_slider,
            self.show_x,
            self.show_y,
            self.show_z,
            self.opacity_slider,
            self.crange_slider,
            self.cmap_dropdown,
        ]:
            control.observe(self._update_figure, names='value')

    def show(self) -> None:
        display(self.controls, self.fig)


@coarseness('volume')
def planes(
    volume: np.ndarray,
    colormap: str | matplotlib.colors.Colormap = 'magma',
    min_value: float = None,
    max_value: float = None,
) -> None:
    """
    Displays an interactive 3D widget for viewing orthogonal cross-sections through a volume.

    Args:
        volume (np.ndarray): The 3D volume of interest.
        colormap (str or matplotlib.colors.Colormap, optional): Specifies the matplotlib color map.
        min_value (float, optional): Together with max_value define the data range the colormap covers. By default colormap covers the full range.
        max_value (float, optional): Together with min_value define the data range the colormap covers. By default colormap covers the full range.

    Returns:
        None

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.shell_225x128x128
        qim3d.viz.planes(vol)
        ```
        ![viz planes](../../assets/screenshots/viz-planes.gif)

    """
    VolumePlaneSlicer(
        volume=volume, colormap=colormap, color_range=[min_value, max_value]
    ).show()


class OverlaySlicer:
    _css_injected = False  # class-level flag

    def __init__(
        self,
        vol1: np.ndarray,
        vol2: np.ndarray,
        display_size: int = 512,
        cmaps: ColormapLike | tuple[ColormapLike, ColormapLike] = 'gray',
        volume1_values: tuple[float, float] = (None, None),
        volume2_values: tuple[float, float] = (None, None),
    ):
        self.vol1 = vol1
        self.vol2 = vol2
        self.display_size = display_size

        if isinstance(cmaps, str | matplotlib.colors.Colormap):
            cmaps = (cmaps, cmaps)
        self.cmaps = tuple(matplotlib.cm.get_cmap(c) for c in cmaps)
        self.img_format = 'png'

        self.slice_axis = 0
        self.slice_index = vol1.shape[self.slice_axis] // 2

        self.volume1_values = volume1_values
        self.volume2_values = volume2_values

        # inject CSS
        if not OverlaySlicer._css_injected:
            display(
                HTML("""
            <style>
            .pixelated {
                image-rendering: pixelated !important;
                image-rendering: crisp-edges !important;
            }
            </style>
            """)
            )
            OverlaySlicer._css_injected = True

        self._init_widgets()
        self._update_slice_axis(self.slice_axis)
        self._set_observers()
        self._update(
            slice_axis=self.slice_axis_widget.value,
            slice_index=self.slice_index_widget.value,
            fade=self.fade_slider.value,
        )

    def _init_widgets(self) -> None:
        self.fade_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            description='',
            continuous_update=True,
            readout=False,
        )
        self.fade_slider.layout.width = f'{self.display_size}px'
        self.fade_slider.style = {'handle_color': None, 'description_width': '0px'}

        self.slice_axis_widget = widgets.Dropdown(
            options=[0, 1, 2], value=self.slice_axis, description='Slice axis'
        )
        self.slice_axis_widget.layout.width = '250px'

        self.slice_index_widget = widgets.IntSlider(
            min=0,
            step=1,
            description='Slice index',
            layout=widgets.Layout(width='400px', height='auto'),
        )

        # image + centering container
        self.img_widget = widgets.Image(format=self.img_format)
        self.img_widget.add_class('pixelated')  # ensures nearest-neighbor
        self.img_box = widgets.Box(
            [self.img_widget],
            layout=widgets.Layout(
                width=f'{self.display_size}px',
                justify_content='center',
                align_items='center',
            ),
        )

    def _set_observers(self) -> None:
        self.fade_slider.observe(self._on_change, names='value')
        self.slice_axis_widget.observe(self._on_change, names='value')
        self.slice_index_widget.observe(self._on_change, names='value')

    def _on_change(self, _change: dict) -> None:
        self._update(
            slice_axis=self.slice_axis_widget.value,
            slice_index=self.slice_index_widget.value,
            fade=self.fade_slider.value,
        )

    def _update_slice_axis(self, slice_axis: int) -> None:
        self.slice_axis = slice_axis
        axis_len = self.vol1.shape[slice_axis]
        self.slice_index_widget.max = axis_len - 1
        self.slice_index_widget.value = axis_len // 2

    @staticmethod
    def _normalize(
        arr: np.ndarray, clim: tuple[float, float] = (None, None)
    ) -> np.ndarray:
        """Normalize to float in the interval [0,1]."""
        a = arr.astype(float)
        vmin = clim[0] if clim[0] else a.min()
        vmax = clim[1] if clim[1] else a.max()
        if vmax == vmin:
            vmax = vmin + 1.0
        return (a - vmin) / (vmax - vmin)

    def _blend(self, fraction: float) -> np.ndarray:
        """Blends slices by first converting to the colormaps' RGB space."""
        slice_axis = self.slice_axis_widget.value
        slice_index = self.slice_index_widget.value
        slice1 = self._normalize(
            np.take(self.vol1, slice_index, axis=slice_axis), clim=self.volume1_values
        )
        slice2 = self._normalize(
            np.take(self.vol2, slice_index, axis=slice_axis), clim=self.volume2_values
        )
        # cmap requires values in the interval [0,1] for its call method and returns RGBA in [0,1] as the last axis
        # drop alpha
        slice1_rgb = self.cmaps[0](slice1)[..., :3]
        slice2_rgb = self.cmaps[1](slice2)[..., :3]

        slice_blended = (1.0 - fraction) * slice1_rgb + fraction * slice2_rgb
        return slice_blended

    def _rgb_arr_to_bytes(self, arr: np.ndarray) -> bytes:
        arr = self._normalize(arr)
        arr = (arr * 255).astype(np.uint8)

        buf = io.BytesIO()
        PIL.Image.fromarray(arr, mode='RGB').save(buf, format=self.img_format.upper())
        return buf.getvalue()

    def _update(self, slice_axis: int, slice_index: int, fade: float) -> None:
        if slice_axis != self.slice_axis:
            self._update_slice_axis(slice_axis)
            slice_index = self.slice_index_widget.value

        blended = self._blend(fade)
        self.img_widget.value = self._rgb_arr_to_bytes(blended)

        # --- make display_size the maximum dimension ---
        h, w = blended.shape[:2]
        if w >= h:
            self.img_widget.layout.width = f'{self.display_size}px'
            self.img_widget.layout.height = 'auto'
        else:
            self.img_widget.layout.width = 'auto'
            self.img_widget.layout.height = f'{self.display_size}px'

        # ensure the centering box and slider share the same width
        self.img_box.layout.width = f'{self.display_size}px'
        self.fade_slider.layout.width = f'{self.display_size}px'

    # ---------- public builder ----------
    def build_interactive(self) -> widgets.VBox:
        left_label = widgets.Label('volume1', layout=widgets.Layout(width='auto'))
        right_label = widgets.Label('volume2', layout=widgets.Layout(width='auto'))
        labels_row = widgets.HBox(
            [
                left_label,
                widgets.Box(layout=widgets.Layout(flex='1 1 auto')),
                right_label,
            ],
            layout=widgets.Layout(width=f'{self.display_size}px'),
        )

        return widgets.VBox(
            [
                self.slice_axis_widget,
                self.slice_index_widget,
                self.img_box,
                self.fade_slider,
                labels_row,
            ]
        )


def overlay(
    volume1: np.ndarray,
    volume2: np.ndarray,
    volume1_values: tuple[float, float] = (None, None),
    volume2_values: tuple[float, float] = (None, None),
    colormaps: ColormapLike | tuple[ColormapLike, ColormapLike] = 'gray',
    display_size: int = 512,
) -> widgets.interactive:
    """
    Returns an interactive widget for comparing two volumes along slices in a fading overlay image.

    Args:
        volume1 (np.ndarray): The first volume.
        volume2 (np.ndarray): The second volume.
        volume1_values (tuple[float, float], optional): Set the color limits of volume1.
        volume2_values (tuple[float, float], optional): Set the color limits of volume2.
        colormaps (ColormapLike or tuple[ColormapLike, ColormapLike], optional): Specifies the colormaps used for each volume. A single value will be applied to both volumes.
        display_size (int, optional): Size in pixels of the image. If image is non-square, then the largest dimension will have display_size pixels.

    Returns:
        widget (widgets.widget_box.VBox): The interactive widget.


    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary = qim3d.filters.gaussian(vol, sigma=2) < 60
        labeled_volume, num_labels = qim3d.segmentation.watershed(binary)

        segm_cmap = qim3d.viz.colormaps.segmentation(num_labels, style = 'bright')

        qim3d.viz.overlay(vol, labeled_volume, colormaps=('grey', segm_cmap), volume2_values=(0, num_labels))
        ```
        ![viz overlay](../../assets/screenshots/viz-overlay.gif)

    """
    if volume1.ndim != 3:
        msg = 'Volume must be 3D.'
        raise ValueError(msg)
    if volume1.shape != volume2.shape:
        msg = 'Volumes must have the same shape.'
        raise ValueError(msg)

    interactive_widget = OverlaySlicer(
        vol1=volume1,
        vol2=volume2,
        cmaps=colormaps,
        display_size=display_size,
        volume1_values=volume1_values,
        volume2_values=volume2_values,
    ).build_interactive()
    return interactive_widget
