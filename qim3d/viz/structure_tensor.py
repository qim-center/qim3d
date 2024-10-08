import numpy as np
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
import logging
from qim3d.utils.logger import log


previous_logging_level = logging.getLogger().getEffectiveLevel()
logging.getLogger().setLevel(logging.CRITICAL)
import structure_tensor as st

logging.getLogger().setLevel(previous_logging_level)


def vectors(
    volume: np.ndarray,
    vec: np.ndarray,
    axis: int = 0,
    volume_cmap:str = 'grey',
    vmin:float = None,
    vmax:float = None,
    slice_idx: Optional[Union[int, float]] = None,
    grid_size: int = 10,
    interactive: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = False,
) -> Union[plt.Figure, widgets.interactive]:
    """
    Visualizes the orientation of the structures in a 3D volume using the eigenvectors of the structure tensor.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        vec (np.ndarray): The eigenvectors of the structure tensor.
        axis (int, optional): The axis along which to visualize the orientation. Defaults to 0.
        volume_cmap (str, optional): Defines colormap for display of the volume
        vmin (float, optional): Together with vmax define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        vmax (float, optional): Together with vmin define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        slice_idx (int or float, optional): The initial slice to be visualized. The slice index
            can afterwards be changed. If value is an integer, it will be the index of the slice
            to be visualized. If value is a float between 0 and 1, it will be multiplied by the
            number of slices and rounded to the nearest integer. If None, the middle slice will
            be used. Defaults to None.
        grid_size (int, optional): The size of the grid. Defaults to 10.
        interactive (bool, optional): If True, returns an interactive widget. Defaults to True.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (15, 5).
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.

    Raises:
        ValueError: If the axis to slice along is not 0, 1, or 2.
        ValueError: If the slice index is not an integer or a float between 0 and 1.

    Returns:
        fig (Union[plt.Figure, widgets.interactive]): If `interactive` is True, returns an interactive widget. Otherwise, returns a matplotlib figure.

    Note:
        The orientation of the vectors is visualized using an HSV color map, where the saturation corresponds to the vector component
        of the slicing direction (i.e. z-component when choosing visualization along `axis = 0`). Hence, if an orientation in the volume
        is orthogonal to the slicing direction, the corresponding color of the visualization will be gray.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.NT_128x128x128
        val, vec = qim3d.processing.structure_tensor(vol)

        # Visualize the structure tensor
        qim3d.viz.vectors(vol, vec, axis = 2, interactive = True)
        ```
        ![structure tensor](assets/screenshots/structure_tensor_visualization.gif)

    """

    # Ensure volume is a float
    if volume.dtype != np.float32 and volume.dtype != np.float64:
        volume = volume.astype(np.float32)

    # Normalize the volume if needed (i.e. if values are in [0, 255])
    if volume.max() > 1.0:
        volume = volume / 255.0

    # Define grid size limits
    min_grid_size = max(1, volume.shape[axis] // 50)
    max_grid_size = max(1, volume.shape[axis] // 10)
    if max_grid_size <= min_grid_size:
        max_grid_size = min_grid_size * 5

    if not grid_size:
        grid_size = (min_grid_size + max_grid_size) // 2

    # Testing
    if grid_size < min_grid_size or grid_size > max_grid_size:
        # Adjust grid size as little as possible to be within the limits
        grid_size = min(max(min_grid_size, grid_size), max_grid_size)
        log.warning(f"Adjusting grid size to {grid_size} as it is out of bounds.")

    def _structure_tensor(volume, vec, axis, slice_idx, grid_size, figsize, show):

        # Choose the appropriate slice based on the specified dimension
        if axis == 0:
            data_slice = volume[slice_idx, :, :]
            vectors_slice_x = vec[0, slice_idx, :, :]
            vectors_slice_y = vec[1, slice_idx, :, :]
            vectors_slice_z = vec[2, slice_idx, :, :]

        elif axis == 1:
            data_slice = volume[:, slice_idx, :]
            vectors_slice_x = vec[0, :, slice_idx, :]
            vectors_slice_y = vec[2, :, slice_idx, :]
            vectors_slice_z = vec[1, :, slice_idx, :]

        elif axis == 2:
            data_slice = volume[:, :, slice_idx]
            vectors_slice_x = vec[1, :, :, slice_idx]
            vectors_slice_y = vec[2, :, :, slice_idx]
            vectors_slice_z = vec[0, :, :, slice_idx]

        else:
            raise ValueError("Invalid dimension. Use 0 for Z, 1 for Y, or 2 for X.")

        # Create three subplots
        fig, ax = plt.subplots(1, 3, figsize=figsize, layout="constrained")

        blend_hue_saturation = (
            lambda hue, sat: hue * (1 - sat) + 0.5 * sat
        )  # Function for blending hue and saturation
        blend_slice_colors = lambda slice, colors: 0.5 * (
            slice + colors
        )  # Function for blending image slice with orientation colors

        # ----- Subplot 1: Image slice with orientation vectors ----- #
        # Create meshgrid with the correct dimensions
        xmesh, ymesh = np.mgrid[0 : data_slice.shape[0], 0 : data_slice.shape[1]]

        # Create a slice object for selecting the grid points
        g = slice(grid_size // 2, None, grid_size)

        # Angles from 0 to pi
        angles_quiver = np.mod(
            np.arctan2(vectors_slice_y[g, g], vectors_slice_x[g, g]), np.pi
        )

        # Calculate z-component (saturation)
        saturation_quiver = (vectors_slice_z[g, g] ** 2)[:, :, np.newaxis]

        # Calculate hue
        hue_quiver = plt.cm.hsv(angles_quiver / np.pi)

        # Blend hue and saturation
        rgba_quiver = blend_hue_saturation(hue_quiver, saturation_quiver)
        rgba_quiver = np.clip(
            rgba_quiver, 0, 1
        )  # Ensure rgba values are values within [0, 1]
        rgba_quiver_flat = rgba_quiver.reshape(
            (rgba_quiver.shape[0] * rgba_quiver.shape[1], 4)
        )  # Flatten array for quiver plot

        # Plot vectors
        ax[0].quiver(
            ymesh[g, g],
            xmesh[g, g],
            vectors_slice_x[g, g],
            vectors_slice_y[g, g],
            color=rgba_quiver_flat,
            angles="xy",
        )
        ax[0].quiver(
            ymesh[g, g],
            xmesh[g, g],
            -vectors_slice_x[g, g],
            -vectors_slice_y[g, g],
            color=rgba_quiver_flat,
            angles="xy",
        )

        ax[0].imshow(data_slice, cmap = volume_cmap, vmin = vmin, vmax = vmax)
        ax[0].set_title(
            f"Orientation vectors (slice {slice_idx})"
            if not interactive
            else "Orientation vectors"
        )
        ax[0].set_axis_off()

        # ----- Subplot 2: Orientation histogram ----- #
        nbins = 36

        # Angles from 0 to pi
        angles = np.mod(np.arctan2(vectors_slice_y, vectors_slice_x), np.pi)

        # Orientation histogram over angles
        distribution, bin_edges = np.histogram(angles, bins=nbins, range=(0.0, np.pi))

        # Half circle (180 deg)
        bin_centers = (np.arange(nbins) + 0.5) * np.pi / nbins

        # Calculate z-component (saturation) for each bin
        bins = np.digitize(angles.ravel(), bin_edges)
        saturation_bin = np.array(
            [
                (
                    np.mean((vectors_slice_z**2).ravel()[bins == i])
                    if np.sum(bins == i) > 0
                    else 0
                )
                for i in range(1, len(bin_edges))
            ]
        )

        # Calculate hue for each bin
        hue_bin = plt.cm.hsv(bin_centers / np.pi)

        # Blend hue and saturation
        rgba_bin = hue_bin.copy()
        rgba_bin[:, :3] = blend_hue_saturation(
            hue_bin[:, :3], saturation_bin[:, np.newaxis]
        )

        ax[1].bar(bin_centers, distribution, width=np.pi / nbins, color=rgba_bin)
        ax[1].set_xlabel("Angle [radians]")
        ax[1].set_xlim([0, np.pi])
        ax[1].set_aspect(np.pi / ax[1].get_ylim()[1])
        ax[1].set_xticks([0, np.pi / 2, np.pi])
        ax[1].set_xticklabels(["0", "$\\frac{\\pi}{2}$", "$\\pi$"])
        ax[1].set_yticks([])
        ax[1].set_ylabel("Frequency")
        ax[1].set_title(f"Histogram over orientation angles")

        # ----- Subplot 3: Image slice colored according to orientation ----- #
        # Calculate z-component (saturation)
        saturation = (vectors_slice_z**2)[:, :, np.newaxis]

        # Calculate hue
        hue = plt.cm.hsv(angles / np.pi)

        # Blend hue and saturation
        rgba = blend_hue_saturation(hue, saturation)

        # Grayscale image slice blended with orientation colors
        data_slice_orientation_colored = (
            blend_slice_colors(plt.cm.gray(data_slice), rgba) * 255
        ).astype("uint8")

        ax[2].imshow(data_slice_orientation_colored)
        ax[2].set_title(
            f"Colored orientations (slice {slice_idx})"
            if not interactive
            else "Colored orientations"
        )
        ax[2].set_axis_off()

        if show:
            plt.show()

        plt.close()

        return fig

    if vec.ndim == 5:
        vec = vec[0, ...]
        log.warning(
            "Eigenvector array is full. Only the eigenvectors corresponding to the first eigenvalue will be used."
        )

    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2

    elif isinstance(slice_idx, float):
        if slice_idx < 0 or slice_idx > 1:
            raise ValueError(
                "Values of slice_idx of float type must be between 0 and 1."
            )
        slice_idx = int(slice_idx * volume.shape[0]) - 1

    if interactive:
        slide_idx_slider = widgets.IntSlider(
            min=0,
            max=volume.shape[axis] - 1,
            step=1,
            value=slice_idx,
            description="Slice index",
            layout=widgets.Layout(width="450px"),
        )

        grid_size_slider = widgets.IntSlider(
            min=min_grid_size,
            max=max_grid_size,
            step=1,
            value=grid_size,
            description="Grid size",
            layout=widgets.Layout(width="450px"),
        )

        widget_obj = widgets.interactive(
            _structure_tensor,
            volume=widgets.fixed(volume),
            vec=widgets.fixed(vec),
            axis=widgets.fixed(axis),
            slice_idx=slide_idx_slider,
            grid_size=grid_size_slider,
            figsize=widgets.fixed(figsize),
            show=widgets.fixed(True),
        )
        # Arrange sliders horizontally
        sliders_box = widgets.HBox([slide_idx_slider, grid_size_slider])
        widget_obj = widgets.VBox([sliders_box, widget_obj.children[-1]])
        widget_obj.layout.align_items = "center"

        if show:
            display(widget_obj)

        return widget_obj

    else:
        return _structure_tensor(volume, vec, axis, slice_idx, grid_size, figsize, show)
