"""
Provides a collection of visualization functions.
"""

import math
import warnings

from typing import List, Optional, Union, Tuple

import dask.array as da
import ipywidgets as widgets
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import SVG, display, clear_output
import matplotlib
import numpy as np
import seaborn as sns
import skimage.measure

import qim3d
from qim3d.utils._logger import log


def slices_grid(
    volume: np.ndarray,
    slice_axis: int = 0,
    slice_positions: Optional[Union[str, int, List[int]]] = None,
    num_slices: int = 15,
    max_columns: int = 5,
    color_map: str = 'magma',
    value_min: float = None,
    value_max: float = None,
    image_size: int = None,
    image_height: int = 2,
    image_width: int = 2,
    display_figure: bool = False,
    display_positions: bool = True,
    interpolation: Optional[str] = None,
    color_bar: bool = False,
    color_bar_style: str = 'small',
    **matplotlib_imshow_kwargs,
) -> matplotlib.figure.Figure:
    """
    Displays one or several slices from a 3d volume.

    By default if `slice_positions` is None, slices_grid plots `num_slices` linearly spaced slices.
    If `slice_positions` is given as a string or integer, slices_grid will plot an overview with `num_slices` figures around that position.
    If `slice_positions` is given as a list, `num_slices` will be ignored and the slices from `slice_positions` will be plotted.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        slice_axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        slice_positions (int or list[int] or str or None, optional): One or several slicing levels. If None, linearly spaced slices will be displayed. Defaults to None.
        num_slices (int, optional): Defines how many slices the user wants to be displayed. Defaults to 15.
        max_columns (int, optional): The maximum number of columns to be plotted. Defaults to 5.
        color_map (str or matplotlib.colors.LinearSegmentedColormap, optional): Specifies the color map for the image. Defaults to "magma".
        value_min (float, optional): Together with value_max define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        value_max (float, optional): Together with value_min define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        image_height (int, optional): Height of the figure.
        image_width (int, optional): Width of the figure.
        display_figure (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.
        display_positions (bool, optional): If True, displays the position of the slices. Defaults to True.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.
        color_bar (bool, optional): Adds a colorbar positioned in the top-right for the corresponding colormap and data range. Defaults to False.
        color_bar_style (str, optional): Determines the style of the colorbar. Option 'small' is height of one image row. Option 'large' spans full height of image grid. Defaults to 'small'.

    Returns:
        fig (matplotlib.figure.Figure): The figure with the slices from the 3d array.

    Raises:
        ValueError: If the input is not a numpy.ndarray or da.core.Array.
        ValueError: If the slice_axis to slice along is not a valid choice, i.e. not an integer between 0 and the number of dimensions of the volume minus 1.
        ValueError: If the file or array is not a volume with at least 3 dimensions.
        ValueError: If the `position` keyword argument is not a integer, list of integers or one of the following strings: "start", "mid" or "end".
        ValueError: If the color_bar_style keyword argument is not one of the following strings: 'small' or 'large'.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.shell_225x128x128
        qim3d.viz.slices_grid(vol, num_slices=15)
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
    if not isinstance(volume, (np.ndarray, da.core.Array)):
        raise ValueError('Data type not supported')

    if volume.ndim < 3:
        raise ValueError(
            'The provided object is not a volume as it has less than 3 dimensions.'
        )

    color_bar_style_options = ['small', 'large']
    if color_bar_style not in color_bar_style_options:
        raise ValueError(
            f"Value '{color_bar_style}' is not valid for colorbar style. Please select from {color_bar_style_options}."
        )

    if isinstance(volume, da.core.Array):
        volume = volume.compute()

    # Ensure axis is a valid choice
    if not (0 <= slice_axis < volume.ndim):
        raise ValueError(
            f"Invalid value for 'slice_axis'. It should be an integer between 0 and {volume.ndim - 1}."
        )

    # Here we deal with the case that the user wants to use the objects colormap directly
    if (
        type(color_map) == matplotlib.colors.LinearSegmentedColormap
        or color_map == 'segmentation'
    ):
        num_labels = len(np.unique(volume))

        if color_map == 'segmentation':
            color_map = qim3d.viz.colormaps.segmentation(num_labels)
        # If value_min and value_max are not set like this, then in case the
        # number of objects changes on new slice, objects might change
        # colors. So when using a slider, the same object suddently
        # changes color (flickers), which is confusing and annoying.
        value_min = 0
        value_max = num_labels

    # Get total number of slices in the specified dimension
    n_total = volume.shape[slice_axis]

    # Position is not provided - will use linearly spaced slices
    if slice_positions is None:
        slice_idxs = np.linspace(0, n_total - 1, num_slices, dtype=int)
    # Position is a string
    elif isinstance(slice_positions, str) and slice_positions.lower() in [
        'start',
        'mid',
        'end',
    ]:
        if slice_positions.lower() == 'start':
            slice_idxs = _get_slice_range(0, num_slices, n_total)
        elif slice_positions.lower() == 'mid':
            slice_idxs = _get_slice_range(n_total // 2, num_slices, n_total)
        elif slice_positions.lower() == 'end':
            slice_idxs = _get_slice_range(n_total - 1, num_slices, n_total)
    #  Position is an integer
    elif isinstance(slice_positions, int):
        slice_idxs = _get_slice_range(slice_positions, num_slices, n_total)
    # Position is a list of integers
    elif isinstance(slice_positions, list) and all(
        isinstance(idx, int) for idx in slice_positions
    ):
        slice_idxs = slice_positions
    else:
        raise ValueError(
            'Position not recognized. Choose an integer, list of integers or one of the following strings: "start", "mid" or "end".'
        )

    # Make grid
    nrows = math.ceil(num_slices / max_columns)
    ncols = min(num_slices, max_columns)

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
    if isinstance(volume, da.core.Array):
        volume = volume.compute()

    if color_bar:
        # In this case, we want the vrange to be constant across the
        # slices, which makes them all comparable to a single color_bar.
        new_value_min = value_min if value_min is not None else np.min(volume)
        new_value_max = value_max if value_max is not None else np.max(volume)

    # Run through each ax of the grid
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(np.atleast_1d(ax_row)):
            slice_idx = i * max_columns + j
            try:
                slice_img = volume.take(slice_idxs[slice_idx], axis=slice_axis)

                if not color_bar:
                    # If value_min is higher than the highest value in the
                    # image ValueError is raised. We don't want to
                    # override the values because next slices might be okay
                    new_value_min = (
                        None
                        if (
                            isinstance(value_min, (float, int))
                            and value_min > np.max(slice_img)
                        )
                        else value_min
                    )
                    new_value_max = (
                        None
                        if (
                            isinstance(value_max, (float, int))
                            and value_max < np.min(slice_img)
                        )
                        else value_max
                    )

                ax.imshow(
                    slice_img,
                    cmap=color_map,
                    interpolation=interpolation,
                    vmin=new_value_min,
                    vmax=new_value_max,
                    **matplotlib_imshow_kwargs,
                )

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
                        bbox=dict(facecolor='#303030', linewidth=0, pad=0),
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
                        bbox=dict(facecolor='#303030', linewidth=0, pad=0),
                    )

            except IndexError:
                # Not a problem, because we simply do not have a slice to show
                pass

            # Hide the axis, so that we have a nice grid
            ax.axis('off')

    if color_bar:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            fig.tight_layout()

        norm = matplotlib.colors.Normalize(
            vmin=new_value_min, vmax=new_value_max, clip=True
        )
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=color_map)

        if color_bar_style == 'small':
            # Figure coordinates of top-right axis
            tr_pos = np.atleast_1d(axs[0])[-1].get_position()
            # The width is divided by ncols to make it the same relative size to the images
            color_bar_ax = fig.add_axes(
                [tr_pos.x1 + 0.05 / ncols, tr_pos.y0, 0.05 / ncols, tr_pos.height]
            )
            fig.colorbar(mappable=mappable, cax=color_bar_ax, orientation='vertical')
        elif color_bar_style == 'large':
            # Figure coordinates of bottom- and top-right axis
            br_pos = np.atleast_1d(axs[-1])[-1].get_position()
            tr_pos = np.atleast_1d(axs[0])[-1].get_position()
            # The width is divided by ncols to make it the same relative size to the images
            color_bar_ax = fig.add_axes(
                [
                    br_pos.xmax + 0.05 / ncols,
                    br_pos.y0 + 0.0015,
                    0.05 / ncols,
                    (tr_pos.y1 - br_pos.y0) - 0.0015,
                ]
            )
            fig.colorbar(mappable=mappable, cax=color_bar_ax, orientation='vertical')

    if display_figure:
        plt.show()

    plt.close()

    return fig


def _get_slice_range(position: int, num_slices: int, n_total: int) -> np.ndarray:
    """Helper function for `slices`. Returns the range of slices to be displayed around the given position."""
    start_idx = position - num_slices // 2
    end_idx = (
        position + num_slices // 2
        if num_slices % 2 == 0
        else position + num_slices // 2 + 1
    )
    slice_idxs = np.arange(start_idx, end_idx)

    if slice_idxs[0] < 0:
        slice_idxs = np.arange(0, num_slices)
    elif slice_idxs[-1] > n_total:
        slice_idxs = np.arange(n_total - num_slices, n_total)

    return slice_idxs


def slicer(
    volume: np.ndarray,
    slice_axis: int = 0,
    color_map: str = 'magma',
    value_min: float = None,
    value_max: float = None,
    image_height: int = 3,
    image_width: int = 3,
    display_positions: bool = False,
    interpolation: Optional[str] = None,
    image_size: int = None,
    color_bar: str = None,
    **matplotlib_imshow_kwargs,
) -> widgets.interactive:
    """
    Interactive widget for visualizing slices of a 3D volume.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        slice_axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        color_map (str or matplotlib.colors.LinearSegmentedColormap, optional): Specifies the color map for the image. Defaults to 'magma'.
        value_min (float, optional): Together with value_max define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        value_max (float, optional): Together with value_min define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        image_height (int, optional): Height of the figure. Defaults to 3.
        image_width (int, optional): Width of the figure. Defaults to 3.
        display_positions (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.
        color_bar (str, optional): Controls the options for color bar. If None, no color bar is included. If 'volume', the color map range is constant for each slice. If 'slices', the color map range changes dynamically according to the slice. Defaults to None.

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

    color_bar_options = [None, 'slices', 'volume']
    if color_bar not in color_bar_options:
        raise ValueError(
            f"Unrecognized value '{color_bar}' for parameter color_bar. "
            f'Expected one of {color_bar_options}.'
        )
    show_color_bar = color_bar is not None
    if color_bar == 'slices':
        # Precompute the minimum and maximum along each slice for faster widget sliding.
        non_slice_axes = tuple(i for i in range(volume.ndim) if i != slice_axis)
        slice_mins = np.min(volume, axis=non_slice_axes)
        slice_maxs = np.max(volume, axis=non_slice_axes)

    # Create the interactive widget
    def _slicer(slice_positions):
        if color_bar == 'slices':
            dynamic_min = slice_mins[slice_positions]
            dynamic_max = slice_maxs[slice_positions]
        else:
            dynamic_min = value_min
            dynamic_max = value_max

        fig = slices_grid(
            volume,
            slice_axis=slice_axis,
            color_map=color_map,
            value_min=dynamic_min,
            value_max=dynamic_max,
            image_height=image_height,
            image_width=image_width,
            display_positions=display_positions,
            interpolation=interpolation,
            slice_positions=slice_positions,
            num_slices=1,
            display_figure=True,
            color_bar=show_color_bar,
            **matplotlib_imshow_kwargs,
        )
        return fig

    position_slider = widgets.IntSlider(
        value=volume.shape[slice_axis] // 2,
        min=0,
        max=volume.shape[slice_axis] - 1,
        description='Slice',
        continuous_update=True,
    )
    slicer_obj = widgets.interactive(_slicer, slice_positions=position_slider)
    slicer_obj.layout = widgets.Layout(align_items='flex-start')

    return slicer_obj


def slicer_orthogonal(
    volume: np.ndarray,
    color_map: str = 'magma',
    value_min: float = None,
    value_max: float = None,
    image_height: int = 3,
    image_width: int = 3,
    display_positions: bool = False,
    interpolation: Optional[str] = None,
    image_size: int = None,
) -> widgets.interactive:
    """
    Interactive widget for visualizing orthogonal slices of a 3D volume.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        color_map (str or matplotlib.colors.LinearSegmentedColormap, optional): Specifies the color map for the image. Defaults to "magma".
        value_min (float, optional): Together with value_max define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        value_max (float, optional): Together with value_min define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        image_height (int, optional): Height of the figure.
        image_width (int, optional): Width of the figure.
        display_positions (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        slicer_orthogonal_obj (widgets.HBox): The interactive widget for visualizing orthogonal slices of a 3D volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fly_150x256x256
        qim3d.viz.slicer_orthogonal(vol, color_map="magma")
        ```
        ![viz slicer_orthogonal](../../assets/screenshots/viz-orthogonal.gif)

    """

    if image_size:
        image_height = image_size
        image_width = image_size

    get_slicer_for_axis = lambda slice_axis: slicer(
        volume,
        slice_axis=slice_axis,
        color_map=color_map,
        value_min=value_min,
        value_max=value_max,
        image_height=image_height,
        image_width=image_width,
        display_positions=display_positions,
        interpolation=interpolation,
    )

    z_slicer = get_slicer_for_axis(slice_axis=0)
    y_slicer = get_slicer_for_axis(slice_axis=1)
    x_slicer = get_slicer_for_axis(slice_axis=2)

    z_slicer.children[0].description = 'Z'
    y_slicer.children[0].description = 'Y'
    x_slicer.children[0].description = 'X'

    return widgets.HBox([z_slicer, y_slicer, x_slicer])


def fade_mask(
    volume: np.ndarray,
    axis: int = 0,
    color_map: str = 'magma',
    value_min: float = None,
    value_max: float = None,
) -> widgets.interactive:
    """
    Interactive widget for visualizing the effect of edge fading on a 3D volume.

    This can be used to select the best parameters before applying the mask.

    Args:
        volume (np.ndarray): The volume to apply edge fading to.
        axis (int, optional): The axis along which to apply the fading. Defaults to 0.
        color_map (str, optional): Specifies the color map for the image. Defaults to "viridis".
        value_min (float or None, optional): Together with value_max define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        value_max (float or None, optional): Together with value_min define the data range the colormap covers. By default colormap covers the full range. Defaults to None

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
    def _slicer(position, decay_rate, ratio, geometry, invert):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        slice_img = volume[position, :, :]
        # If value_min is higher than the highest value in the image ValueError is raised
        # We don't want to override the values because next slices might be okay
        new_value_min = (
            None
            if (isinstance(value_min, (float, int)) and value_min > np.max(slice_img))
            else value_min
        )
        new_value_max = (
            None
            if (isinstance(value_max, (float, int)) and value_max < np.min(slice_img))
            else value_max
        )

        axes[0].imshow(
            slice_img, cmap=color_map, vmin=new_value_min, vmax=new_value_max
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
        axes[1].imshow(mask[position, :, :], cmap=color_map)
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
        # If value_min is higher than the highest value in the image ValueError is raised
        # We don't want to override the values because next slices might be okay
        slice_img = masked_volume[position, :, :]
        new_value_min = (
            None
            if (isinstance(value_min, (float, int)) and value_min > np.max(slice_img))
            else value_min
        )
        new_value_max = (
            None
            if (isinstance(value_max, (float, int)) and value_max < np.min(slice_img))
            else value_max
        )
        axes[2].imshow(
            slice_img, cmap=color_map, vmin=new_value_min, vmax=new_value_max
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


def chunks(zarr_path: str, **kwargs) -> widgets.interactive:
    """
    Function to visualize chunks of a Zarr dataset using the specified visualization method.

    Args:
        zarr_path (str or os.PathLike): Path to the Zarr dataset.
        **kwargs (Any): Additional keyword arguments to pass to the visualization method.

    Example:
        ```python
        import qim3d

        # Download dataset
        downloader = qim3d.io.Downloader()
        data = downloader.Snail.Escargot(load_file=True)

        # Export as OME-Zarr
        qim3d.io.export_ome_zarr("Escargot.zarr", data, chunk_size=100, downsample_rate=2, replace=True)

        # Explore chunks
        qim3d.viz.chunks("Escargot.zarr")
        ```
        ![chunks-visualization](../../assets/screenshots/chunks_visualization.gif)

    """

    # Load the Zarr dataset
    zarr_data = zarr.open(zarr_path, mode='r')

    # Save arguments for later use
    # visualization_method = visualization_method
    # preserved_kwargs = kwargs

    # Create label to display the chunk coordinates
    widget_title = widgets.HTML('<h2>Chunk Explorer</h2>')
    chunk_info_label = widgets.HTML(value='Chunk info will be displayed here')

    def load_and_visualize(
        scale, z_coord, y_coord, x_coord, visualization_method, **kwargs
    ):
        # Get chunk shape for the selected scale
        chunk_shape = zarr_data[scale].chunks

        # Calculate slice indices for the selected chunk
        slices = (
            slice(
                z_coord * chunk_shape[0],
                min((z_coord + 1) * chunk_shape[0], zarr_data[scale].shape[0]),
            ),
            slice(
                y_coord * chunk_shape[1],
                min((y_coord + 1) * chunk_shape[1], zarr_data[scale].shape[1]),
            ),
            slice(
                x_coord * chunk_shape[2],
                min((x_coord + 1) * chunk_shape[2], zarr_data[scale].shape[2]),
            ),
        )

        # Extract start and stop values from each slice object
        z_start, z_stop = slices[0].start, slices[0].stop
        y_start, y_stop = slices[1].start, slices[1].stop
        x_start, x_stop = slices[2].start, slices[2].stop

        # Extract the chunk
        chunk = zarr_data[scale][slices]

        # Update the chunk info label with the chunk coordinates
        info_string = (
            f'<b>shape:</b> {chunk_shape}\n'
            + f'<b>coordinates:</b> ({z_coord}, {y_coord}, {x_coord})\n'
            + f'<b>ranges: </b>Z({z_start}-{z_stop})   Y({y_start}-{y_stop})   X({x_start}-{x_stop})\n'
            + f'<b>dtype:</b> {chunk.dtype}\n'
            + f'<b>min value:</b> {np.min(chunk)}\n'
            + f'<b>max value:</b> {np.max(chunk)}\n'
            + f'<b>mean value:</b> {np.mean(chunk)}\n'
        )

        chunk_info_label.value = f"""
            <div style="font-size: 14px; text-align: left; margin-left:32px">
                <h3 style="margin: 0px">Chunk Info</h3>
                    <div style="font-size: 14px; text-align: left;">
                    <pre>{info_string}</pre>
                    </div>
            </div>

            """

        # Prepare chunk visualization based on the selected method
        if visualization_method == 'slicer':  # return a widget
            viz_widget = qim3d.viz.slicer(chunk, **kwargs)
        elif visualization_method == 'slices':  # return a plt.Figure
            viz_widget = widgets.Output()
            with viz_widget:
                viz_widget.clear_output(wait=True)
                fig = qim3d.viz.slices_grid(chunk, **kwargs)
                display(fig)
        elif visualization_method == 'volume':
            viz_widget = widgets.Output()
            with viz_widget:
                viz_widget.clear_output(wait=True)
                out = qim3d.viz.volumetric(chunk, show=False, **kwargs)
                display(out)
        else:
            log.info(f'Invalid visualization method: {visualization_method}')

        return viz_widget

    # Function to calculate the number of chunks for each dimension, including partial chunks
    def get_num_chunks(shape, chunk_size):
        return [(s + chunk_size[i] - 1) // chunk_size[i] for i, s in enumerate(shape)]

    scale_options = {
        f'{i} {zarr_data[i].shape}': i for i in range(len(zarr_data))
    }  # len(zarr_data) gives number of scales

    description_width = '128px'
    # Create dropdown for scale
    scale_dropdown = widgets.Dropdown(
        options=scale_options,
        value=0,  # Default to first scale
        description='OME-Zarr scale',
        style={'description_width': description_width, 'text_align': 'left'},
    )

    # Initialize the options for x, y, and z based on the first scale by default
    multiscale_shape = zarr_data[0].shape
    chunk_shape = zarr_data[0].chunks
    num_chunks = get_num_chunks(multiscale_shape, chunk_shape)

    z_dropdown = widgets.Dropdown(
        options=list(range(num_chunks[0])),
        value=0,
        description='First dimension (Z)',
        style={'description_width': description_width, 'text_align': 'left'},
    )

    y_dropdown = widgets.Dropdown(
        options=list(range(num_chunks[1])),
        value=0,
        description='Second dimension (Y)',
        style={'description_width': description_width, 'text_align': 'left'},
    )

    x_dropdown = widgets.Dropdown(
        options=list(range(num_chunks[2])),
        value=0,
        description='Third dimension (X)',
        style={'description_width': description_width, 'text_align': 'left'},
    )

    method_dropdown = widgets.Dropdown(
        options=['slicer', 'slices', 'volume'],
        value='slicer',
        description='Visualization',
        style={'description_width': description_width, 'text_align': 'left'},
    )

    # Funtion to temporarily disable observers
    def disable_observers():
        x_dropdown.unobserve(update_visualization, names='value')
        y_dropdown.unobserve(update_visualization, names='value')
        z_dropdown.unobserve(update_visualization, names='value')
        method_dropdown.unobserve(update_visualization, names='value')

    # Funtion to enable observers
    def enable_observers():
        x_dropdown.observe(update_visualization, names='value')
        y_dropdown.observe(update_visualization, names='value')
        z_dropdown.observe(update_visualization, names='value')
        method_dropdown.observe(update_visualization, names='value')

    # Function to update the x, y, z dropdowns when the scale changes and reset the coordinates to 0
    def update_coordinate_dropdowns(scale):
        disable_observers()  # to avoid multiple reload of the visualization when updating the dropdowns

        multiscale_shape = zarr_data[scale].shape
        chunk_shape = zarr_data[scale].chunks
        num_chunks = get_num_chunks(
            multiscale_shape, chunk_shape
        )  # Calculate  new chunk options

        # Reset X, Y, Z dropdowns to 0
        z_dropdown.options = list(range(num_chunks[0]))
        z_dropdown.value = 0  # Reset to 0
        z_dropdown.disabled = (
            len(z_dropdown.options) == 1
        )  # Disable if only one option (0) is available

        y_dropdown.options = list(range(num_chunks[1]))
        y_dropdown.value = 0  # Reset to 0
        y_dropdown.disabled = (
            len(y_dropdown.options) == 1
        )  # Disable if only one option (0) is available

        x_dropdown.options = list(range(num_chunks[2]))
        x_dropdown.value = 0  # Reset to 0
        x_dropdown.disabled = (
            len(x_dropdown.options) == 1
        )  # Disable if only one option (0) is available

        enable_observers()

        update_visualization()

    # Function to update the visualization when any dropdown value changes
    def update_visualization(*args):
        scale = scale_dropdown.value
        x_coord = x_dropdown.value
        y_coord = y_dropdown.value
        z_coord = z_dropdown.value
        visualization_method = method_dropdown.value

        # Clear and update the chunk visualization
        slicer_widget = load_and_visualize(
            scale, z_coord, y_coord, x_coord, visualization_method, **kwargs
        )

        # Recreate the layout and display the new visualization
        final_layout.children = [widget_title, hbox_layout, slicer_widget]

    # Attach an observer to scale dropdown to update x, y, z dropdowns when the scale changes
    scale_dropdown.observe(
        lambda change: update_coordinate_dropdowns(scale_dropdown.value), names='value'
    )

    enable_observers()

    # Create first visualization
    slicer_widget = load_and_visualize(
        scale_dropdown.value,
        z_dropdown.value,
        y_dropdown.value,
        x_dropdown.value,
        method_dropdown.value,
        **kwargs,
    )

    # Create the layout
    vbox_dropbox = widgets.VBox(
        [scale_dropdown, z_dropdown, y_dropdown, x_dropdown, method_dropdown]
    )
    hbox_layout = widgets.HBox([vbox_dropbox, chunk_info_label])
    final_layout = widgets.VBox([widget_title, hbox_layout, slicer_widget])

    # Display the VBox
    display(final_layout)


def histogram(
    volume: np.ndarray,
    bins: Union[int, str] = 'auto',
    slice_idx: Union[int, str] = None,
    axis: int = 0,
    kde: bool = True,
    log_scale: bool = False,
    despine: bool = True,
    show_title: bool = True,
    color: str = 'qim3d',
    edgecolor: str | None = None,
    figsize: tuple[float, float] = (8, 4.5),
    element: str = 'step',
    return_fig: bool = False,
    show: bool = True,
    **sns_kwargs,
) -> None | matplotlib.figure.Figure:
    """
    Plots a histogram of voxel intensities from a 3D volume, with options to show a specific slice or the entire volume.

    Utilizes [seaborn.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html) for visualization.

    Args:
        volume (np.ndarray): A 3D NumPy array representing the volume to be visualized.
        bins (int or str, optional): Number of histogram bins or a binning strategy (e.g., "auto"). Default is "auto".
        axis (int, optional): Axis along which to take a slice. Default is 0.
        slice_idx (int or str or None, optional): Specifies the slice to visualize. If an integer, it represents the slice index along the selected axis.
                                               If "middle", the function uses the middle slice. If None, the entire volume is visualized. Default is None.
        kde (bool, optional): Whether to overlay a kernel density estimate. Default is True.
        log_scale (bool, optional): Whether to use a logarithmic scale on the y-axis. Default is False.
        despine (bool, optional): If True, removes the top and right spines from the plot for cleaner appearance. Default is True.
        show_title (bool, optional): If True, displays a title with slice information. Default is True.
        color (str, optional): Color for the histogram bars. If "qim3d", defaults to the qim3d color. Default is "qim3d".
        edgecolor (str, optional): Color for the edges of the histogram bars. Default is None.
        figsize (tuple of floats, optional): Size of the figure (width, height). Default is (8, 4.5).
        element (str, optional): Type of histogram to draw ('bars', 'step', or 'poly'). Default is "step".
        return_fig (bool, optional): If True, returns the figure object instead of showing it directly. Default is False.
        show (bool, optional): If True, displays the plot. If False, suppresses display. Default is True.
        **sns_kwargs (Any): Additional keyword arguments for `seaborn.histplot`.

    Returns:
        fig (Optional[matplotlib.figure.Figure]): If `return_fig` is True, returns the generated figure object. Otherwise, returns None.

    Raises:
        ValueError: If `axis` is not a valid axis index (0, 1, or 2).
        ValueError: If `slice_idx` is an integer and is out of range for the specified axis.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.histogram(vol)
        ```
        ![viz histogram](../../assets/screenshots/viz-histogram-vol.png)

        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.histogram(vol, bins=32, slice_idx="middle", axis=1, kde=False, log_scale=True)
        ```
        ![viz histogram](../../assets/screenshots/viz-histogram-slice.png)

    """

    if not (0 <= axis < volume.ndim):
        raise ValueError(f'Axis must be an integer between 0 and {volume.ndim - 1}.')

    if slice_idx == 'middle':
        slice_idx = volume.shape[axis] // 2

    if slice_idx:
        if 0 <= slice_idx < volume.shape[axis]:
            img_slice = np.take(volume, indices=slice_idx, axis=axis)
            data = img_slice.ravel()
            title = f'Intensity histogram of slice #{slice_idx} {img_slice.shape} along axis {axis}'
        else:
            raise ValueError(
                f'Slice index out of range. Must be between 0 and {volume.shape[axis] - 1}.'
            )
    else:
        data = volume.ravel()
        title = f'Intensity histogram for whole volume {volume.shape}'

    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        plt.yscale('log')

    if color == 'qim3d':
        color = qim3d.viz.colormaps.qim(1.0)

    sns.histplot(
        data,
        bins=bins,
        kde=kde,
        color=color,
        element=element,
        edgecolor=edgecolor,
        **sns_kwargs,
    )

    if despine:
        sns.despine(
            fig=None,
            ax=None,
            top=True,
            right=True,
            left=False,
            bottom=False,
            offset={'left': 0, 'bottom': 18},
            trim=True,
        )

    plt.xlabel('Voxel Intensity')
    plt.ylabel('Frequency')

    if show_title:
        plt.title(title, fontsize=10)

    # Handle show and return
    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig

class _LineProfile:
    def __init__(self, volume, slice_axis, slice_index, vertical_position, horizontal_position, angle, fraction_range):
        self.volume = volume
        self.slice_axis = slice_axis

        self.dims = np.array(volume.shape)
        self.pad = 1 # Padding on pivot point to avoid border issues
        self.cmap = [matplotlib.cm.plasma, matplotlib.cm.spring][1]

        self.initialize_widgets()
        self.update_slice_axis(slice_axis)
        self.slice_index_widget.value = slice_index
        self.x_widget.value = horizontal_position
        self.y_widget.value = vertical_position
        self.angle_widget.value = angle
        self.line_fraction_widget.value = [fraction_range[0], fraction_range[1]]
    
    def update_slice_axis(self, slice_axis):
        self.slice_axis = slice_axis
        self.slice_index_widget.max = self.volume.shape[slice_axis] - 1
        self.slice_index_widget.value = self.volume.shape[slice_axis] // 2

        self.x_max, self.y_max = np.delete(self.dims, self.slice_axis) - 1
        self.x_widget.max = self.x_max - self.pad
        self.x_widget.value = self.x_max // 2
        self.y_widget.max = self.y_max - self.pad
        self.y_widget.value = self.y_max // 2

    def initialize_widgets(self):
        layout = widgets.Layout(width='300px', height='auto')
        self.x_widget = widgets.IntSlider(min=self.pad, step=1, description="", layout=layout)
        self.y_widget = widgets.IntSlider(min=self.pad, step=1, description="", layout=layout)
        self.angle_widget = widgets.IntSlider(min=0, max=360, step=1, value=0, description="", layout=layout)
        self.line_fraction_widget = widgets.FloatRangeSlider(
            min=0, max=1, step=0.01, value=[0, 1], 
            description="", layout=layout
        )

        self.slice_axis_widget = widgets.Dropdown(options=[0,1,2], value=self.slice_axis, description='Slice axis')
        self.slice_axis_widget.layout.width = '250px'

        self.slice_index_widget = widgets.IntSlider(min=0, step=1, description="Slice index", layout=layout)
        self.slice_index_widget.layout.width = '400px'
    
    def calculate_line_endpoints(self, x, y, angle):
        """
        Line is parameterized as: [x + t*np.cos(angle), y + t*np.sin(angle)]
        """
        if np.isclose(angle, 0):
            return [0, y], [self.x_max, y]
        elif np.isclose(angle, np.pi/2):
            return [x, 0], [x, self.y_max]
        elif np.isclose(angle, np.pi):
            return [self.x_max, y], [0, y]
        elif np.isclose(angle, 3*np.pi/2):
            return [x, self.y_max], [x, 0]
        elif np.isclose(angle, 2*np.pi):
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
    
    def update(self, slice_axis, slice_index, x, y, angle_deg, fraction_range):
        if slice_axis != self.slice_axis:
            self.update_slice_axis(slice_axis)
            x = self.x_widget.value
            y = self.y_widget.value
            slice_index = self.slice_index_widget.value
        
        clear_output(wait=True)
        
        image = np.take(self.volume, slice_index, slice_axis)
        angle = np.radians(angle_deg)
        src, dst = [np.array(point, dtype='float32') for point in self.calculate_line_endpoints(x, y, angle)]

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
        segments = np.stack([np.column_stack([y_seg[:-2], x_seg[:-2]]), 
                             np.column_stack([y_seg[2:], x_seg[2:]])], axis=1)
        norm = plt.Normalize(vmin=0, vmax=num_segments-1)
        colors = self.cmap(norm(np.arange(num_segments - 1)))
        lc = matplotlib.collections.LineCollection(segments, colors=colors, linewidth=2)

        ax[0].imshow(image,cmap='gray')
        ax[0].add_collection(lc)
        # pivot point
        ax[0].plot(y,x,marker='s', linestyle='', color='cyan', markersize=4)
        ax[0].set_xlabel(f'axis {np.delete(np.arange(3), self.slice_axis)[1]}')
        ax[0].set_ylabel(f'axis {np.delete(np.arange(3), self.slice_axis)[0]}')
        
        # Profile intensity plot
        norm = plt.Normalize(0, vmax=len(y_pline) - 1)
        x_pline = np.arange(len(y_pline))
        points = np.column_stack((x_pline, y_pline))[:, np.newaxis, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = matplotlib.collections.LineCollection(segments, cmap=self.cmap, norm=norm, array=x_pline[:-1], linewidth=2)

        ax[1].add_collection(lc)
        ax[1].autoscale()
        ax[1].set_xlabel('Distance along line')
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()
    
    def build_interactive(self):
        # Group widgets into two columns
        title_style = "text-align:center; font-size:16px; font-weight:bold; margin-bottom:5px;"
        title_column1 = widgets.HTML(f"<div style='{title_style}'>Line parameterization</div>")
        title_column2 = widgets.HTML(f"<div style='{title_style}'>Slice selection</div>")

        # Make label widgets instead of descriptions which have different lengths.
        label_layout = widgets.Layout(width='120px')
        label_x = widgets.Label("Vertical position", layout=label_layout)
        label_y = widgets.Label("Horizontal position", layout=label_layout)
        label_angle = widgets.Label("Angle (°)", layout=label_layout)
        label_fraction = widgets.Label("Fraction range", layout=label_layout)

        row_x = widgets.HBox([label_x, self.x_widget])
        row_y = widgets.HBox([label_y, self.y_widget])
        row_angle = widgets.HBox([label_angle, self.angle_widget])
        row_fraction = widgets.HBox([label_fraction, self.line_fraction_widget])

        controls_column1 = widgets.VBox([title_column1, row_x, row_y, row_angle, row_fraction])
        controls_column2 = widgets.VBox([title_column2, self.slice_axis_widget, self.slice_index_widget])
        controls = widgets.HBox([controls_column1, controls_column2])

        interactive_plot = widgets.interactive_output(
            self.update, 
            {'slice_axis': self.slice_axis_widget, 'slice_index': self.slice_index_widget, 
            'x': self.x_widget, 'y': self.y_widget, 'angle_deg': self.angle_widget,
            'fraction_range': self.line_fraction_widget}
        )

        return widgets.VBox([controls, interactive_plot])

def line_profile(
        volume: np.ndarray,
        slice_axis: int=0,
        slice_index: int | str='middle',
        vertical_position: int | str='middle',
        horizontal_position: int | str='middle',
        angle: int=0,
        fraction_range: Tuple[float,float]=(0.00, 1.00)
    ) -> widgets.interactive:
    """Returns an interactive widget for visualizing the intensity profiles of lines on slices.

    Args:
        volume (np.ndarray): The 3D volume of interest.
        slice_axis (int, optional): Specifies the initial axis along which to slice.
        slice_index (int or str, optional): Specifies the initial slice index along slice_axis.
        vertical_position (int or str, optional): Specifies the initial vertical position of the line's pivot point.
        horizontal_position (int or str, optional): Specifies the initial horizontal position of the line's pivot point.
        angle (int or float, optional): Specifies the initial angle (°) of the line around the pivot point. A float will be converted to an int. A value outside the range will be wrapped modulo. 
        fraction_range (tuple or list, optional): Specifies the fraction of the line segment to use from border to border. Both the start and the end should be in the range [0.0, 1.0].

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
    def parse_position(pos, pos_range, name):
        if isinstance(pos, int):
            if not pos_range[0] <= pos < pos_range[1]:
                raise ValueError(f'Value for {name} must be inside [{pos_range[0]}, {pos_range[1]}]')
            return pos
        elif isinstance(pos, str):
            pos = pos.lower()
            if pos == 'start': return pos_range[0]
            elif pos == 'middle': return pos_range[0] + (pos_range[1] - pos_range[0]) // 2
            elif pos == 'end': return pos_range[1]
            else:
                raise ValueError(
                    f"Invalid string '{pos}' for {name}. "
                    "Must be 'start', 'middle', or 'end'."
                )
        else:
            raise TypeError(f'Axis position must be of type int or str.')
    
    if not isinstance(volume, (np.ndarray, da.core.Array)):
        raise ValueError("Data type for volume not supported.")
    if volume.ndim != 3:
        raise ValueError("Volume must be 3D.")
    
    dims = volume.shape
    slice_index = parse_position(slice_index, (0, dims[slice_axis] - 1), 'slice_index')
    # the omission of the ends for the pivot point is due to border issues.
    vertical_position = parse_position(vertical_position, (1, np.delete(dims, slice_axis)[0] - 2), 'vertical_position')
    horizontal_position = parse_position(horizontal_position, (1, np.delete(dims, slice_axis)[1] - 2), 'horizontal_position')
    
    if not isinstance(angle, int | float):
        raise ValueError("Invalid type for angle.")
    angle = round(angle) % 360

    if not (0.0 <= fraction_range[0] <= 1.0 and 0.0 <= fraction_range[1] <= 1.0 and fraction_range[0] <= fraction_range[1]):
        raise ValueError("Invalid values for fraction_range.")

    lp = _LineProfile(volume, slice_axis, slice_index, vertical_position, horizontal_position, angle, fraction_range)
    return lp.build_interactive()
