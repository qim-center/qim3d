import logging
from typing import Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from qim3d.utils._logger import log

previous_logging_level = logging.getLogger().getEffectiveLevel()
logging.getLogger().setLevel(logging.CRITICAL)

logging.getLogger().setLevel(previous_logging_level)


def vectors(
    volume: np.ndarray,
    vectors: np.ndarray,
    axis: int = 0,
    volume_colormap: str = 'grey',
    min_value: float | None = None,
    max_value: float | None = None,
    slice_index: int | float | None = None,
    grid_size: int = 10,
    interactive: bool = True,
    figsize: tuple[int, int] = (10, 5),
    show: bool = False,
) -> plt.Figure | widgets.interactive:
    """
    Visualizes the orientation of the structures in a 3D volume using the eigenvectors of the structure tensor.

    Args:
        volume (np.ndarray): The 3D volume to be sliced.
        vectors (np.ndarray): The eigenvectors of the structure tensor.
        axis (int, optional): The axis along which to visualize the orientation. Defaults to 0.
        volume_colormap (str, optional): Defines colormap for display of the volume
        min_value (float, optional): Together with max_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        max_value (float, optional): Together with min_value define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        slice_index (int or float or None, optional): The initial slice to be visualized. The slice index
            can afterwards be changed. If value is an integer, it will be the index of the slice
            to be visualized. If value is a float between 0 and 1, it will be multiplied by the
            number of slices and rounded to the nearest integer. If None, the middle slice will
            be used. Defaults to None.
        grid_size (int, optional): The size of the grid. Defaults to 10.
        interactive (bool, optional): If True, returns an interactive widget. Defaults to True.
        figsize (tuple, optional): The size of the figure. Defaults to (15, 5).
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.

    Raises:
        ValueError: If the axis to slice along is not 0, 1, or 2.
        ValueError: If the slice index is not an integer or a float between 0 and 1.

    Returns:
        fig (widgets.interactive or plt.Figure): If `interactive` is True, returns an interactive widget. Otherwise, returns a matplotlib figure.

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
        ![structure tensor](../../assets/screenshots/structure_tensor_visualization.gif)

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
        log.warning(f'Adjusting grid size to {grid_size} as it is out of bounds.')

    def _structure_tensor(volume, vectors, axis, slice_index, grid_size, figsize, show):
        # Choose the appropriate slice based on the specified dimension
        if axis == 0:
            data_slice = volume[slice_index, :, :]
            vectors_slice_x = vectors[0, slice_index, :, :]
            vectors_slice_y = vectors[1, slice_index, :, :]
            vectors_slice_z = vectors[2, slice_index, :, :]

        elif axis == 1:
            data_slice = volume[:, slice_index, :]
            vectors_slice_x = vectors[0, :, slice_index, :]
            vectors_slice_y = vectors[2, :, slice_index, :]
            vectors_slice_z = vectors[1, :, slice_index, :]

        elif axis == 2:
            data_slice = volume[:, :, slice_index]
            vectors_slice_x = vectors[1, :, :, slice_index]
            vectors_slice_y = vectors[2, :, :, slice_index]
            vectors_slice_z = vectors[0, :, :, slice_index]

        else:
            msg = 'Invalid dimension. Use 0 for Z, 1 for Y, or 2 for X.'
            raise ValueError(msg)

        # Create three subplots
        fig, ax = plt.subplots(1, 3, figsize=figsize, layout='constrained')

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
        g = slice(grid_size // 2, None, grid_size)  # noqa: A002

        # Angles from 0 to pi
        angles_quiver = np.mod(
            np.arctan2(
                vectors_slice_y[g, g],
                vectors_slice_x[g, g],
            ),
            np.pi,
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
            angles='xy',
        )
        ax[0].quiver(
            ymesh[g, g],
            xmesh[g, g],
            -vectors_slice_x[g, g],
            -vectors_slice_y[g, g],
            color=rgba_quiver_flat,
            angles='xy',
        )

        ax[0].imshow(data_slice, cmap=volume_colormap, vmin=min_value, vmax=max_value)
        ax[0].set_title(
            f'Orientation vectors (slice {slice_index})'
            if not interactive
            else 'Orientation vectors'
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
        ax[1].set_xlabel('Angle [radians]')
        ax[1].set_xlim([0, np.pi])
        ax[1].set_aspect(np.pi / ax[1].get_ylim()[1])
        ax[1].set_xticks([0, np.pi / 2, np.pi])
        ax[1].set_xticklabels(['0', '$\\frac{\\pi}{2}$', '$\\pi$'])
        ax[1].set_yticks([])
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Histogram over orientation angles')

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
        ).astype('uint8')

        ax[2].imshow(data_slice_orientation_colored)
        ax[2].set_title(
            f'Colored orientations (slice {slice_index})'
            if not interactive
            else 'Colored orientations'
        )
        ax[2].set_axis_off()

        if show:
            plt.show()

        plt.close()

        return fig

    if vectors.ndim == 5:
        vectors = vectors[0, ...]
        log.warning(
            'Eigenvector array is full. Only the eigenvectors corresponding to the first eigenvalue will be used.'
        )

    if slice_index is None:
        slice_index = volume.shape[axis] // 2

    elif isinstance(slice_index, float):
        if slice_index < 0 or slice_index > 1:
            raise ValueError(
                'Values of slice_index of float type must be between 0 and 1.'
            )
        slice_index = int(slice_index * volume.shape[0]) - 1

    if interactive:
        slice_index_slider = widgets.IntSlider(
            min=0,
            max=volume.shape[axis] - 1,
            step=1,
            value=slice_index,
            description='Slice index',
            layout=widgets.Layout(width='450px'),
        )

        grid_size_slider = widgets.IntSlider(
            min=min_grid_size,
            max=max_grid_size,
            step=1,
            value=grid_size,
            description='Grid size',
            layout=widgets.Layout(width='450px'),
        )

        widget_obj = widgets.interactive(
            _structure_tensor,
            volume=widgets.fixed(volume),
            vectors=widgets.fixed(vectors),
            axis=widgets.fixed(axis),
            slice_index=slice_index_slider,
            grid_size=grid_size_slider,
            figsize=widgets.fixed(figsize),
            show=widgets.fixed(True),
        )
        # Arrange sliders horizontally
        sliders_box = widgets.HBox([slice_index_slider, grid_size_slider])
        widget_obj = widgets.VBox([sliders_box, widget_obj.children[-1]])
        widget_obj.layout.align_items = 'center'

        if show:
            display(widget_obj)

        return widget_obj

    else:
        return _structure_tensor(
            volume, vectors, axis, slice_index, grid_size, figsize, show
        )


def vector_field_3d(
    vec: np.ndarray,
    val: np.ndarray,
    select_eigen: Literal['smallest', 'largest', 'middle'] = 'smallest',
    sampling_step: int = 4,
    max_cones: int = 50000,
    cone_size: float = 1,
    verbose: bool = True,
    colormap: str = 'Portland',
    cmin: float = None,
    cmax: float = None,
    **kwargs,
) -> go.Figure:
    """
    Visualize 3D structure tensor eigenvectors as cones in Plotly.

    Each cone represents an eigenvector whose direction and size indicate the dominant local orientation and the magnitude of the corresponding eigenvalue.
    If `sampling_step` is greater than 1, each cone represents the average orientation and magnitude within that sampled region.

    The function is designed to work directly with the outputs of `qim3d.processing.structure_tensor()` which is in ascendic order by default.

    Args:
        val (np.ndarray): Eigenvalues from the structure tensor, with shape `(3, *vol.shape)`.
        vec (np.ndarray): Eigenvectors from the structure tensor.
            Shape depends on the `smallest` parameter from qim3d.processing.structure_tensor():

            - `(3, nx, ny, nz)` if `smallest=True`
            - `(3, 3, nx, ny, nz)` if `smallest=False`

        select_eigen (Literal["smallest","largest","middle"], optional):
            If vec has shape `(3, 3, nx, ny, nz)`, specifies which eigenvector to visualize.
        sampling_step (int, optional):
            Grid spacing for sampling points.
            Default is `4`.
        max_cones (int, optional):
            Maximum number of cones to display. If more points are sampled,
            only the locations with the highest eigenvalues are kept.
            Default is `50000`.
        cone_size (float, optional):
            Scaling factor for cone length, proportional to vector magnitude.
            Default is `1`.
        verbose (bool, optional):
            Whether to print progress and info messages.
            Default is `True`.
        colormap (str, optional):
            Name of the Plotly colorscale used for cones.
            Default is `"Portland"`.
        cmin (float, optional):
            Minimum value for color scale. If `None`, uses the minimum vector magnitude.
        cmax (float, optional):
            Maximum value for color scale. If `None`, uses the maximum vector magnitude.
        **kwargs:
            Additional keyword arguments passed to `plotly.graph_objects.Cone`.
            See the [Plotly Cone documentation](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Cone.html)
            for full customization options.

    Raises:
        ValueError: If an invalid combination of `smallest` and `sort` is provided.

    Returns:
        fig (plotly.graph_objects.Figure):
            Interactive 3D Plotly figure showing cone representations of local orientation vectors.

    Example:
        ```python
        vol = qim3d.examples.fibers_150x150x150
        val, vec = qim3d.processing.structure_tensor(vol, smallest = False)
        qim3d.viz.vector_field_3d(vec, val, sampling_step=12, max_cones=5000, cone_size = 2, select_eigen="smallest")
        ```
        ![vector field](../../assets/screenshots/viz-vector_field.png)

    Notes:
        **Understanding the Structure Tensor**

        Each voxel is associated with three **eigenvalues** (λ₁, λ₂, λ₃) and corresponding
        **eigenvectors**, which describe how much intensity varies along each direction.

        The relative magnitudes of the eigenvalues determine the type of local structure:

        | Structure type              | Eigenvalue pattern               | Dominant orientation vector                          |
        |-----------------------------|----------------------------------|-------------------------------------------------------|
        | **Planar structure (surface)** | Two large, one small (λ₁, λ₂ ≫ λ₃) | Eigenvector with **smallest** eigenvalue → surface normal |
        | **Linear structure (fiber)**   | One large, two small (λ₁ ≫ λ₂, λ₃) | Eigenvector with **largest** eigenvalue → line direction  |
        | **Isotropic region (flat)**    | All similar (λ₁ ≈ λ₂ ≈ λ₃)        | No dominant direction                                  |

        So based on what you are interested in visualizing, you may want to select different eigenvectors using the `select_eigen` parameter.

    """
    if vec.ndim == 4:
        val = val[0]
    elif vec.ndim == 5:
        if select_eigen == 'largest':
            val = val[2]
            vec = vec[2, :, ...]
        elif select_eigen == 'smallest':
            val = val[0]
            vec = vec[0, :, ...]
        elif select_eigen == 'middle':
            val = val[1]
            vec = vec[1, :, ...]
        else:
            msg = f'Invalid select_eigen value: {select_eigen}. Choose between "smallest", "largest", or "middle".'
            raise ValueError(msg)
    vec = np.transpose(vec, (1, 2, 3, 0))

    nx, ny, nz, _ = vec.shape
    if verbose:
        log.info(f'Original number of grid points: {nx * ny * nz}')
    half = sampling_step // 2

    # Sampling grid
    grid_x = np.arange(0, nx, sampling_step)
    grid_y = np.arange(0, ny, sampling_step)
    grid_z = np.arange(0, nz, sampling_step)

    points, vectors, values = [], [], []

    # Average vectors and eigenvalues in each sampling cube
    for px in grid_x:
        for py in grid_y:
            for pz in grid_z:
                x0, x1 = max(px - half, 0), min(px + half + 1, nx)
                y0, y1 = max(py - half, 0), min(py + half + 1, ny)
                z0, z1 = max(pz - half, 0), min(pz + half + 1, nz)

                region_vecs = vec[x0:x1, y0:y1, z0:z1, :]
                region_vals = val[x0:x1, y0:y1, z0:z1]

                avg_vec = region_vecs.mean(axis=(0, 1, 2))
                avg_val = region_vals.mean()

                points.append((px, py, pz))
                vectors.append(avg_vec)
                values.append(avg_val)

    points = np.array(points)
    vectors = np.array(vectors)
    values = np.array(values)

    # Select top N highest eigenvalue locations
    idx_top = np.argsort(values)[::-1][:max_cones]
    points_top = points[idx_top]
    vectors_top = vectors[idx_top]
    values_top = values[idx_top]

    if verbose:
        log.info(f'Number of grid points sampled: {len(values)}')
        log.info(f'Number of cones actually plotted: {len(points_top)}')

    # Normalize vectors and scale by eigenvalue magnitude
    norms = np.linalg.norm(vectors_top, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_vecs = vectors_top / norms

    # Apply decay to downscale weak directions
    # scaled_strength = values_top * np.exp(
    #     -decay_rate * (1 - values_top / values_top.max())
    # )
    # scaled_strength = np.log(values_top - values_top.min() + 1)

    scaled_strength = np.log1p(values_top)

    u = unit_vecs[:, 0] * scaled_strength
    v = unit_vecs[:, 1] * scaled_strength
    w = unit_vecs[:, 2] * scaled_strength

    # Compute magnitude for color scaling if needed
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    min_mag = magnitude.min()
    max_mag = magnitude.max()
    if verbose:
        log.info(f'Min magnitude: {min_mag:.4f}, Max magnitude: {max_mag:.4f}')

    if cmin is None:
        cmin = min_mag
    if cmax is None:
        cmax = max_mag

    # Use 'raw' to display sizes in actual vector length.
    fig = go.Figure(
        data=go.Cone(
            x=points_top[:, 0],
            y=points_top[:, 1],
            z=points_top[:, 2],
            u=u,
            v=v,
            w=w,
            sizemode='scaled',
            sizeref=cone_size,
            colorscale=colormap,
            colorbar_title='Orientation strength',
            cmin=cmin,
            cmax=cmax,
            **kwargs,
        ),
        layout={'width': 900, 'height': 700},
    )

    return fig
