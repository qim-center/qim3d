import numpy as np
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
import logging as log

def vectors(
    volume: np.ndarray,
    vec: np.ndarray,
    axis: int = 0,
    slice_idx: Optional[Union[int, float]] = None,
    interactive: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = False,
) -> Union[plt.Figure, widgets.interactive]:
    """
    Displays a grid of eigenvectors from the structure tensor to visualize the orientation of the structures in the volume.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        vec (np.ndarray): The eigenvectors of the structure tensor.
        axis (int, optional): The axis along which to visualize the local thickness. Defaults to 0.
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

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.NT_128x128x128
        val, vec = qim3d.processing.structure_tensor(vol, visualize=True, axis=2)

        # Visualize the structure tensor
        qim3d.viz.vectors(vol, vec, axis=2, slice_idx=0.5, interactive=True)
        ```
        ![structure tensor](assets/screenshots/structure_tensor.gif)  

    """

    # Define Grid size limits
    min_grid_size = max(1, volume.shape[axis] // 50)
    max_grid_size = max(1, volume.shape[axis] // 10)
    if max_grid_size <= min_grid_size:
        max_grid_size = min_grid_size * 5

    # Testing
    grid_size = (min_grid_size + max_grid_size) // 2

    if grid_size < min_grid_size or grid_size > max_grid_size:
        # Adjust grid size as little as possible to be within the limits
        grid_size = min(max(min_grid_size, grid_size), max_grid_size)
        log.warning(f"Adjusting grid size to {grid_size} as it is out of bounds.")

    def _structure_tensor(volume, vec, axis, slice_idx, grid_size, figsize, show):

        # Create subplots
        fig, ax = plt.subplots(1, 2, figsize=figsize, layout="constrained")

        # Choose the appropriate slice based on the specified dimension
        if axis == 0:
            data_slice = volume[slice_idx, :, :]
            vectors_slice_x = vec[0, slice_idx, :, :]
            vectors_slice_y = vec[1, slice_idx, :, :]
        elif axis == 1:
            data_slice = volume[:, slice_idx, :]
            vectors_slice_x = vec[0, :, slice_idx, :]
            vectors_slice_y = vec[2, :, slice_idx, :]
        elif axis == 2:
            data_slice = volume[:, :, slice_idx]
            vectors_slice_x = vec[1, :, :, slice_idx]
            vectors_slice_y = vec[2, :, :, slice_idx]
        else:
            raise ValueError("Invalid dimension. Use 0 for Z, 1 for Y, or 2 for X.")

        ax[0].imshow(data_slice, cmap=plt.cm.gray)

        # Create meshgrid with the correct dimensions
        xmesh, ymesh = np.mgrid[0 : data_slice.shape[0], 0 : data_slice.shape[1]]

        # Create a slice object for selecting the grid points
        g = slice(grid_size // 2, None, grid_size)

        # Plot vectors
        ax[0].quiver(
            ymesh[g, g],
            xmesh[g, g],
            vectors_slice_x[g, g],
            vectors_slice_y[g, g],
            color="orange",
            angles="xy",
        )
        ax[0].quiver(
            ymesh[g, g],
            xmesh[g, g],
            -vectors_slice_x[g, g],
            -vectors_slice_y[g, g],
            color="orange",
            angles="xy",
        )

        # Set title and turn off axis
        ax[0].set_title(f"Slice {slice_idx}" if not interactive else None)
        ax[0].set_axis_off()

        # Orientations histogram
        nbins = 36
        angles = np.arctan2(vectors_slice_y, vectors_slice_x)  # angles from 0 to pi
        distribution, bin_edges = np.histogram(angles, bins=nbins, range=(0.0, np.pi))

        # Find the bin with the maximum count
        peak_bin_idx = np.argmax(distribution)
        # Calculate the center of the peak bin
        peak_angle_rad = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2
        # Convert the peak angle to degrees
        peak_angle_deg = np.degrees(peak_angle_rad)
        bin_centers = (np.arange(nbins) + 0.5) * np.pi / nbins  # half circle (180 deg)
        colors = plt.cm.hsv(bin_centers / np.pi)
        ax[1].bar(bin_centers, distribution, width=np.pi / nbins, color=colors)
        ax[1].set_xlabel("Angle [radians]")
        ax[1].set_xlim([0, np.pi])
        ax[1].set_aspect(np.pi / ax[1].get_ylim()[1])
        ax[1].set_xticks([0, np.pi / 2, np.pi])
        ax[1].set_xticklabels(["0", "$\\frac{\\pi}{2}$", "$\\pi$"])
        ax[1].set_ylabel("Count")
        ax[1].set_title(f"Histogram over angles (peak at {round(peak_angle_deg)}°)")

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
        return _structure_tensor(volume, vec, axis, slice_idx, figsize, show)
