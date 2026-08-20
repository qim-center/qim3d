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
    background: int | None = None,
    show: bool = False,
) -> plt.Figure | widgets.interactive:
    """
        Visualizes the local orientation of structures using structure tensor eigenvectors
        overlaid on a 2D slice of the volume.

        Generates a three-panel visualization:

        1. **Quiver Plot:** Arrows showing the dominant orientation direction in the slice plane.
        2. **Orientation Histogram:** Distribution of orientation angles in the current slice,
           colored by the same HSV scheme used in the color map.
        3. **Color Map:** The slice colored by orientation using an HSV scheme where hue encodes
           the in-plane angle and saturation encodes how much the fiber points out of the plane
           (fibers pointing out of the screen appear desaturated toward gray).

        The color scheme follows the fan-based approach described in Dahl 2026:
        https://people.compute.dtu.dk/vand/notes/ST_intro.pdf

    Args:
            volume (np.ndarray): The 3D input volume with shape (Z, Y, X).
            vectors (np.ndarray): The eigenvectors of the structure tensor with shape (3, Z, Y, X).
                If the full eigenvector array of shape (3, 3, Z, Y, X) is provided, only the
                first eigenvector (corresponding to the smallest eigenvalue) will be used.
            axis (int, optional): The axis along which to slice the volume.
                0 slices along Z, 1 along Y, 2 along X. Defaults to 0.
            volume_colormap (str, optional): The colormap used to display the background volume
                slice in the quiver plot. Defaults to 'grey'.
            min_value (float, optional): Minimum intensity value for display contrast adjustment.
                Useful for highlighting specific intensity ranges. Defaults to None.
            max_value (float, optional): Maximum intensity value for display contrast adjustment.
                Useful for highlighting specific intensity ranges. Defaults to None.
            slice_index (int or float, optional): Which slice to display initially.
                Provide an integer for the exact slice index, a float between 0.0 and 1.0
                for a relative position, or None to default to the middle slice.
            grid_size (int, optional): Spacing between arrows in the quiver plot in pixels.
                Lower values produce denser arrow fields. Defaults to 10.
            interactive (bool, optional): If True, returns a widget with sliders to scroll
                through slices and adjust arrow density. If False, returns a static figure.
                Defaults to True.
            figsize (tuple[int, int], optional): Width and height of the figure in inches.
                Defaults to (10, 5).
            background (float, optional): Intensity threshold below which orientation vectors
                are hidden. Useful for suppressing arrows in background regions. Set to 0 to
                hide all zero-intensity regions. Defaults to None (no filtering).
            show (bool, optional): If True, immediately displays the plot by calling
                plt.show(). Defaults to False.

    Returns:
            object (widgets.interactive or matplotlib.figure.Figure):
                A widget with interactive sliders if interactive is True,
                or a static matplotlib figure if interactive is False.

    Raises:
            ValueError: If axis is not 0, 1, or 2, or if slice_index is out of bounds.

    Example:
    ```python
            import qim3d

            vol = qim3d.examples.NT_128x128x128
            val, vec = qim3d.processing.structure_tensor(vol)

            qim3d.viz.vectors(vol, vec, axis=2, interactive=True)
    ```
            ![structure tensor](../../assets/screenshots/structure_tensor_visualization.gif)

    """

    # Ensure volume is a float array for correct normalization
    if volume.dtype != np.float32 and volume.dtype != np.float64:
        volume = volume.astype(np.float32)

    # Normalize volume to [0, 1] if values are in [0, 255]
    if volume.max() > 1.0:
        volume = volume / 255.0

    # Compute valid grid size range based on volume dimensions
    min_grid_size = max(1, volume.shape[axis] // 50)
    max_grid_size = max(1, volume.shape[axis] // 10)
    if max_grid_size <= min_grid_size:
        max_grid_size = min_grid_size * 5

    if not grid_size:
        grid_size = (min_grid_size + max_grid_size) // 2

    if grid_size < min_grid_size or grid_size > max_grid_size:
        grid_size = min(max(min_grid_size, grid_size), max_grid_size)
        log.warning(f'Adjusting grid size to {grid_size} as it is out of bounds.')

    def _structure_tensor(volume, vectors, axis, slice_index, grid_size, figsize, show):
        # Extract the 2D slice and corresponding vector components based on the chosen axis
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

        fig, ax = plt.subplots(1, 3, figsize=figsize, layout='constrained')

        # Blending function: mixes pure hue color toward gray (0.5) based on saturation
        # When saturation is 0 the color is pure hue, when 1 it becomes gray
        blend_hue_saturation = lambda hue, sat: hue * (1 - sat) + 0.5 * sat

        # ===================== PANEL 1: QUIVER PLOT =====================

        xmesh, ymesh = np.mgrid[0 : data_slice.shape[0], 0 : data_slice.shape[1]]

        # Sample grid points at regular intervals defined by grid_size
        g = slice(grid_size // 2, None, grid_size)  # noqa: A002

        # Only show arrows where the volume intensity is above zero
        intensity_mask = data_slice[g, g] > 0

        x_valid = xmesh[g, g][intensity_mask]
        y_valid = ymesh[g, g][intensity_mask]
        vx_valid = vectors_slice_x[g, g][intensity_mask]
        vy_valid = vectors_slice_y[g, g][intensity_mask]

        # Compute in-plane angle in [0, π] using mod π to handle sign ambiguity
        # (v and -v represent the same orientation, mod π maps them to the same angle)
        angles_quiver = np.mod(
            np.arctan2(vectors_slice_y[g, g], vectors_slice_x[g, g]),
            np.pi,
        )

        # The z-component squared gives how much the fiber points out of the slice plane
        # This is used as saturation: high vz means the fiber is out of plane → gray color
        saturation_quiver = (vectors_slice_z[g, g] ** 2)[:, :, np.newaxis]

        hue_quiver = plt.cm.hsv(angles_quiver / np.pi)
        rgba_quiver = blend_hue_saturation(hue_quiver, saturation_quiver)
        rgba_quiver = np.clip(rgba_quiver, 0, 1)
        rgba_quiver_flat = rgba_quiver.reshape(
            (rgba_quiver.shape[0] * rgba_quiver.shape[1], 4)
        )

        # Plot bidirectional arrows (both +v and -v) since eigenvectors have no preferred sense
        ax[0].quiver(
            y_valid, x_valid, vx_valid, vy_valid, color=rgba_quiver_flat, angles='xy'
        )
        ax[0].quiver(
            y_valid, x_valid, -vx_valid, -vy_valid, color=rgba_quiver_flat, angles='xy'
        )
        ax[0].imshow(data_slice, cmap=volume_colormap, vmin=min_value, vmax=max_value)
        ax[0].set_title(
            f'Orientation vectors (slice {slice_index})'
            if not interactive
            else 'Orientation vectors'
        )
        ax[0].set_axis_off()

        # ===================== PANEL 2: ORIENTATION HISTOGRAM =====================

        nbins = 36

        # Compute angles for the full slice, optionally filtered by background threshold
        angles = np.mod(np.arctan2(vectors_slice_y, vectors_slice_x), np.pi)
        intensity_mask_full = (
            data_slice > background
            if background is not None
            else np.ones_like(data_slice, dtype=bool)
        )
        angles_filtered = angles[intensity_mask_full]

        distribution, bin_edges = np.histogram(
            angles_filtered, bins=nbins, range=(0.0, np.pi)
        )
        bin_centers = (np.arange(nbins) + 0.5) * np.pi / nbins

        # Compute mean out-of-plane component per bin for saturation coloring
        bins = np.digitize(angles.ravel(), bin_edges)
        saturation_bin = np.array(
            [
                np.mean((vectors_slice_z**2).ravel()[bins == i])
                if np.sum(bins == i) > 0
                else 0
                for i in range(1, len(bin_edges))
            ]
        )

        hue_bin = plt.cm.hsv(bin_centers / np.pi)
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

        # ===================== PANEL 3: COLOR MAP =====================

        saturation = (vectors_slice_z**2)[:, :, np.newaxis]
        hue = plt.cm.hsv(angles / np.pi)
        rgba = blend_hue_saturation(hue, saturation)

        # Only color pixels above the background threshold, keep the rest as grayscale
        intensity_mask_2d = (
            data_slice > background
            if background is not None
            else np.ones_like(data_slice, dtype=bool)
        )

        gray_slice = plt.cm.gray(data_slice)[:, :, :3]
        data_slice_orientation_colored = gray_slice.copy()
        data_slice_orientation_colored[intensity_mask_2d] = 0.5 * (
            gray_slice[intensity_mask_2d] + rgba[:, :, :3][intensity_mask_2d]
        )
        data_slice_orientation_colored = (data_slice_orientation_colored * 255).astype(
            'uint8'
        )

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

    # If the full eigenvector array is provided, use only the first eigenvector
    if vectors.ndim == 5:
        vectors = vectors[0, ...]
        log.warning(
            'Eigenvector array is full. Only the eigenvectors corresponding to the first eigenvalue will be used.'
        )

    # Determine the initial slice index
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
    volume: np.ndarray,
    select_eigen: Literal['smallest', 'largest', 'middle'] = 'smallest',
    sampling_step: int = 4,
    cone_size: float = 1,
    verbose: bool = True,
    cmin: float = None,
    cmax: float = None,
    **kwargs,
) -> go.Figure:
    """
        Visualizes the 3D eigenvector field of a structure tensor as a bidirectional cone plot.

        Each sampled location in the volume is represented by two mirrored cones pointing in
        opposite directions along the eigenvector axis. Two cones are used because eigenvectors
        are direction-less: the structure tensor tells us the orientation axis but not which end
        points where.

        Color and cone size both encode the eigenvalue corresponding to the selected eigenvector,
        inverted and normalized to [0, 1]. This means cones are larger and brighter where the
        local structure is most coherent (low eigenvalue) and smaller and darker in noisy or
        ambiguous regions (high eigenvalue).

        Note: Plotly's go.Cone colors each cone by the magnitude of its vector, which means
        color and size cannot be decoupled. Any scalar encoded as color will automatically also
        control size. This rules out direction-based coloring schemes. For full RGB coloring
        based on fiber orientation, use the streamlines function instead.

        Background voxels where the volume is zero are masked out before sampling.

        For background on structure tensors and eigenvalue interpretation, see:
        https://people.compute.dtu.dk/vand/notes/ST_intro.pdf

    Args:
            vec (np.ndarray): Eigenvectors of the structure tensor with shape (3, 3, Z, Y, X).
                The first dimension indexes the eigenvector (0 = smallest, 1 = middle, 2 = largest).
            val (np.ndarray): Eigenvalues of the structure tensor with shape (3, Z, Y, X).
                The first dimension indexes the eigenvalue (0 = smallest, 1 = middle, 2 = largest).
            volume (np.ndarray): The original 3D volume with shape (Z, Y, X). Used to mask out
                background regions where the volume intensity is zero.
            select_eigen (str, optional): Which eigenvector to visualize. Use 'smallest' for the
                direction of minimum intensity change (fiber direction in structure tensors),
                'largest' for the direction of maximum change, or 'middle' for the intermediate
                direction. Defaults to 'smallest'.
            sampling_step (int, optional): Spacing in voxels between sampled locations. Higher
                values produce fewer but faster cones. Lower values produce denser visualizations
                but are slower to compute. Defaults to 4.
            cone_size (float, optional): Global scale factor controlling the size of all cones.
                Increase to make cones larger, decrease to make them smaller. Defaults to 1.
            verbose (bool, optional): If True, prints information about the number of cones
                plotted and the eigenvalue range. Defaults to True.
            cmin (float, optional): Minimum value for the colorscale. If None, uses the minimum
                value in the data. Defaults to None.
            cmax (float, optional): Maximum value for the colorscale. If None, uses the maximum
                value in the data. Defaults to None.
            **kwargs: Additional keyword arguments passed directly to Plotly's go.Cone.

    Returns:
            go.Figure: An interactive Plotly 3D figure showing the cone plot.

    Example:
    ```python
            import qim3d

            vol = qim3d.examples.fiber_150x256x256
            val, vec = qim3d.processing.structure_tensor(vol, sigma=2.0, rho=6)

            fig = qim3d.viz.vector_field_3d(vec, val, vol, select_eigen='smallest')
            fig.show()
    ```

    """

    eps = 1e-12

    # Select the eigenvector and its corresponding eigenvalue based on user choice
    if vec.ndim == 5:
        if select_eigen == 'largest':
            vec = vec[2, :, ...]
            eigen_val = val[2]
        elif select_eigen == 'smallest':
            vec = vec[0, :, ...]
            eigen_val = val[0]
        elif select_eigen == 'middle':
            vec = vec[1, :, ...]
            eigen_val = val[1]
        else:
            raise ValueError(
                f'Invalid select_eigen: {select_eigen}. '
                'Choose "smallest", "largest", or "middle".'
            )

    # Rearrange axes from (3, Z, Y, X) to (Z, Y, X, 3) for easier spatial indexing
    vec = np.transpose(vec, (1, 2, 3, 0))

    # Zero out eigenvectors in background voxels to avoid plotting cones in empty space
    vec[volume == 0] = 0

    # Normalize the eigenvalue to [0, 1] using the 99th percentile to avoid outlier spikes,
    # then invert so that low eigenvalues (strong coherent structure) produce large bright cones
    ev_flat = eigen_val.ravel()
    ev_max = np.percentile(ev_flat[ev_flat > 0], 99)
    eigen_val_norm = np.clip(eigen_val / (ev_max + eps), 0, 1)
    eigen_val_norm = 1 - eigen_val_norm

    if verbose:
        log.info(
            f'Eigenvalue range: {eigen_val[eigen_val > 0].min():.4f} to {eigen_val.max():.4f}'
        )
        log.info(
            f'Normalized eigenvalue range: {eigen_val_norm.min():.4f} to {eigen_val_norm.max():.4f}'
        )

    nx, ny, nz, _ = vec.shape
    half = sampling_step // 2

    if verbose:
        log.info(f'Original number of grid points: {nx * ny * nz}')

    points, vectors, val_values = [], [], []

    # Average vectors and eigenvalues within each sampling cube across the volume
    for px in np.arange(0, nx, sampling_step):
        for py in np.arange(0, ny, sampling_step):
            for pz in np.arange(0, nz, sampling_step):
                x0, x1 = max(px - half, 0), min(px + half + 1, nx)
                y0, y1 = max(py - half, 0), min(py + half + 1, ny)
                z0, z1 = max(pz - half, 0), min(pz + half + 1, nz)

                avg_vec = vec[x0:x1, y0:y1, z0:z1, :].mean(axis=(0, 1, 2))
                avg_val = eigen_val_norm[x0:x1, y0:y1, z0:z1].mean()

                # Skip cubes that are entirely background or have no valid data
                if (
                    not np.isfinite(avg_vec).all()
                    or np.all(avg_vec == 0)
                    or avg_val <= 0
                ):
                    continue

                points.append((px, py, pz))
                vectors.append(avg_vec)
                val_values.append(avg_val)

    if not points:
        raise ValueError('No valid cones to plot. Try lowering sampling_step.')

    points = np.array(points)
    vectors = np.array(vectors)
    val_values = np.array(val_values)

    if verbose:
        log.info(f'Cones plotted: {len(points)}')
        log.info(
            f'Eigenvalue (normalized) range: {val_values.min():.4f} to {val_values.max():.4f}'
        )

    # Normalize each vector to unit length for consistent cone direction
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_vecs = vectors / norms

    # Resolve sign ambiguity by forcing vz >= 0
    # Eigenvectors have no preferred sense (v and -v are equivalent), so we
    # standardize the direction to ensure consistent orientation across the field
    flip = unit_vecs[:, 2] < 0
    unit_vecs[flip] *= -1

    # Scale the unit vectors by the normalized eigenvalue so that the cone magnitude
    # encodes structural coherence — go.Cone uses magnitude for both color and size
    u = unit_vecs[:, 0] * val_values
    v = unit_vecs[:, 1] * val_values
    w = unit_vecs[:, 2] * val_values

    if cmin is None:
        cmin = val_values.min()
    if cmax is None:
        cmax = val_values.max()

    shared = dict(
        x=points[:, 2],
        y=points[:, 1],
        z=points[:, 0],
        sizemode='scaled',
        sizeref=cone_size,
        colorscale='Hot',
        cmin=cmin,
        cmax=cmax,
        anchor='tail',
        **kwargs,
    )

    # Plot two mirrored cones per point to represent the bidirectional nature of eigenvectors
    fig = go.Figure(
        data=[
            go.Cone(u=u, v=v, w=w, colorbar_title=f'λ ({select_eigen})', **shared),
            go.Cone(u=-u, v=-v, w=-w, showscale=False, **shared),
        ],
        layout={'width': 900, 'height': 700},
    )

    return fig


def streamlines(
    volume,
    eigenvectors,
    eigenvalues,
    background_threshold=None,
    fiber_spacing=20,
    initial_step_size=0.5,
    max_step_size=2.5,
    max_fiber_length=300,
    terminal_speed=1e-10,
    show_volume=False,
    show_starting_points=False,
    camera_position='iso',
):
    """
        Visualizes fiber orientations as 3D streamlines by tracing paths through the
        eigenvector field of the structure tensor.

        Starting from a uniform grid of seed points placed inside the foreground of the volume,
        each fiber is traced in both directions by following the local eigenvector orientation.
        Tracing stops when the fiber reaches the maximum allowed length or enters a region
        where the vector magnitude drops below the terminal speed threshold, which happens
        naturally in background regions and areas with low structural coherence.

        Fibers are colored using the fan-based color scheme from Dahl 2026, designed for
        planar fiber distributions. The in-plane azimuthal angle of the eigenvector is mapped
        to hue via the HSV color wheel, while the out-of-plane component desaturates the color
        toward gray. Fibers lying flat in the XY plane appear as fully saturated colors, while
        fibers pointing out of plane appear gray. Sign ambiguity is resolved by using mod π
        when computing the angle, so that v and -v always map to the same color.

        For background on structure tensors, eigenvalues, and orientation analysis, see:
        https://people.compute.dtu.dk/vand/notes/ST_intro.pdf

    Parameters:
            volume (np.ndarray): The 3D input volume with shape (Z, Y, X). Used for background
                detection and optional volume rendering.
            eigenvectors (np.ndarray): Structure tensor eigenvectors with shape (3, 3, Z, Y, X)
                or (3, Z, Y, X). The first eigenvector, corresponding to the direction of minimum
                intensity change, is used as the fiber direction.
            eigenvalues (np.ndarray): Structure tensor eigenvalues with shape (3, Z, Y, X).
                The smallest eigenvalue λ1 is used to scale the eigenvectors so that tracing
                stops naturally in noisy or incoherent regions.
            background_threshold (float, optional): Intensity value below which seed points are
                rejected as background. If not provided, it is computed automatically using
                Otsu thresholding. Set to 0 to use all non-zero regions. Defaults to None.
            fiber_spacing (int, optional): Distance in voxels between fiber seed points. Lower
                values produce denser fiber visualizations but increase computation time.
                Defaults to 20.
            initial_step_size (float, optional): Starting step length for fiber integration in
                voxels. Smaller values follow fiber paths more accurately but are slower.
                Defaults to 0.5.
            max_step_size (float, optional): Maximum allowed step length in voxels. Larger values
                produce smoother fibers but may skip fine details. Defaults to 2.5.
            max_fiber_length (int, optional): Maximum number of integration steps per fiber.
                Higher values allow longer fibers but increase computation time. Defaults to 300.
            terminal_speed (float, optional): Minimum vector magnitude below which fiber tracing
                stops. Since eigenvectors are scaled by the inverse of λ1, this threshold
                naturally stops fibers in background and incoherent regions. Lower values allow
                fibers to continue longer into uncertain areas. Defaults to 1e-10.
            show_volume (bool, optional): If True, renders the original volume as a semi-transparent
                gray background behind the fibers for spatial context. Defaults to False.
            show_starting_points (bool, optional): If True, shows the fiber seed points as red
                spheres. Useful for understanding the seeding distribution. Defaults to False.
            camera_position (str, optional): Initial camera viewpoint for the 3D render.
                Options are 'iso' for isometric, 'xy', 'xz', or 'yz'. Defaults to 'iso'.

    Returns:
            None: Displays the visualization directly in a PyVista window.

        Example:
    ```python
            import qim3d

            val, vec = qim3d.processing.structure_tensor(volume, sigma=2.0, rho=6)

            qim3d.viz.streamlines(volume, vec, val)

            qim3d.viz.streamlines(volume, vec, val, fiber_spacing=8, initial_step_size=0.3)

            qim3d.viz.streamlines(volume, vec, val, fiber_spacing=25, show_volume=True)

            qim3d.viz.streamlines(volume, vec, val, max_fiber_length=600, show_starting_points=True)
    ```

    """
    import numpy as np
    import pyvista as pv

    # Compute the background threshold automatically using Otsu thresholding if not provided
    if background_threshold is None:
        nonzero = volume[volume > 0]
        if len(nonzero) > 0:
            vmin, vmax = int(volume.min()), int(volume.max())
            hist, bin_edges = np.histogram(volume, bins=4096, range=(vmin, vmax))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            w0 = np.cumsum(hist)
            w1 = hist.sum() - w0
            mu0 = np.cumsum(hist * bin_centers) / np.maximum(w0, 1)
            mu1 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(w1[::-1], 1))[
                ::-1
            ]
            sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
            background_threshold = bin_centers[np.argmax(sigma_b2)] * 1.05
        else:
            background_threshold = 0

    print(f'Background threshold: {background_threshold:.2f}')

    # Use the first eigenvector which corresponds to the direction of minimum intensity
    # change, representing the dominant fiber orientation in structure tensor analysis
    if eigenvectors.ndim == 5:
        vec_fiber = eigenvectors[0].copy()
    else:
        vec_fiber = eigenvectors.copy()

    # Scale the eigenvectors by the inverse of the smallest eigenvalue λ1.
    # λ1 is small where the local structure is coherent and the fiber direction is reliable.
    # Inverting it produces large magnitudes in well-defined structural regions and small
    # magnitudes in noisy or background regions, causing fiber tracing to stop there naturally.
    epsilon = 1e-12
    l1 = eigenvalues[0]

    l1_valid = l1.ravel()[l1.ravel() > 0]
    inv_l1 = np.clip(1.0 / (l1 + epsilon), 0, 1.0 / epsilon)
    inv_l1_max = np.percentile(inv_l1[l1 > 0], 99) if len(l1_valid) > 0 else 1.0
    inv_l1_norm = np.clip(inv_l1 / (inv_l1_max + epsilon), 0, 1)

    # Explicitly zero out background voxels so fibers stop at the volume boundary
    inv_l1_norm[volume == 0] = 0

    vec_scaled = vec_fiber * inv_l1_norm[np.newaxis, ...]

    print(
        f'Inverse λ1 normalized range: {inv_l1_norm.min():.4f} to {inv_l1_norm.max():.4f}'
    )
    print(
        f'  Mean (foreground): {inv_l1_norm[volume > background_threshold].mean():.4f}'
    )

    # Set up the PyVista structured grid with the volume dimensions and unit spacing
    grid = pv.ImageData()
    grid.dimensions = (volume.shape[2], volume.shape[1], volume.shape[0])
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)

    # Reorder arrays from (Z, Y, X) to (X, Y, Z) as required by PyVista
    vectors_reordered = vec_scaled.transpose(3, 2, 1, 0)
    grid.point_data['vectors'] = vectors_reordered.reshape(-1, 3, order='F')

    intensity_reordered = volume.transpose(2, 1, 0)
    grid.point_data['intensity'] = intensity_reordered.flatten(order='F')

    # Compute the fan-based RGB color for each voxel following Dahl 2026, Fig. 6 col. 3:
    # (r, g, b) = (1 - vz²) · hsv2rgb(arctan(vy/vx), 1, 1) + 0.5·vz²
    # Hue encodes the in-plane azimuthal angle, vz² desaturates out-of-plane fibers toward gray
    vx = vec_fiber[0]
    vy = vec_fiber[1]
    vz = vec_fiber[2]

    # Use mod π to resolve sign ambiguity: arctan2(-vy, -vx) = arctan2(vy, vx) + π,
    # and mod π makes them equal, so v and -v always get the same hue
    hue = (np.arctan2(vy, vx) % np.pi) / np.pi
    h6 = hue * 6.0
    i = h6.astype(int) % 6
    f = h6 - np.floor(h6)

    # Vectorized HSV to RGB conversion with saturation=1 and value=1 (pure hue colors)
    hsv_r = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [1, 1 - f, 0, 0, f, 1]
    ).astype(np.float32)
    hsv_g = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [f, 1, 1, 1 - f, 0, 0]
    ).astype(np.float32)
    hsv_b = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [0, 0, f, 1, 1, 1 - f]
    ).astype(np.float32)

    vz2 = vz**2
    fan_r = (1 - vz2) * hsv_r + 0.5 * vz2
    fan_g = (1 - vz2) * hsv_g + 0.5 * vz2
    fan_b = (1 - vz2) * hsv_b + 0.5 * vz2

    fan_rgb = np.stack([fan_r, fan_g, fan_b], axis=-1)
    fan_rgb_reordered = fan_rgb.transpose(2, 1, 0, 3)
    fan_rgb_flat = (
        np.clip(fan_rgb_reordered, 0, 1).reshape(-1, 3, order='F') * 255
    ).astype(np.uint8)
    grid.point_data['fan_rgb'] = fan_rgb_flat

    # Place seed points on a uniform grid within the bounding box of the foreground
    threshold = (
        np.percentile(intensity_reordered[intensity_reordered > 0], 10)
        if (intensity_reordered > 0).any()
        else 0
    )
    nonzero_coords = np.argwhere(intensity_reordered > threshold)

    if len(nonzero_coords) == 0:
        print('WARNING: No foreground voxels found!')
        return

    x_min, y_min, z_min = nonzero_coords.min(axis=0)
    x_max, y_max, z_max = nonzero_coords.max(axis=0)

    x_seeds = np.arange(x_min, x_max, fiber_spacing)
    y_seeds = np.arange(y_min, y_max, fiber_spacing)
    z_seeds = np.arange(z_min, z_max, fiber_spacing)
    seed_grid = np.array(
        np.meshgrid(x_seeds, y_seeds, z_seeds, indexing='ij')
    ).T.reshape(-1, 3)

    # Keep only seed points where the volume intensity is above the background threshold
    seed_indices = seed_grid.astype(int)
    intensity_3d_grid = grid.point_data['intensity'].reshape(grid.dimensions, order='F')
    seed_intensities = intensity_3d_grid[
        seed_indices[:, 0], seed_indices[:, 1], seed_indices[:, 2]
    ]
    valid_seeds = seed_grid[seed_intensities > background_threshold]

    print(f'Seeds: {len(seed_grid)} → {len(valid_seeds)} after filtering')

    if len(valid_seeds) == 0:
        print('WARNING: No valid seeds after filtering!')
        return

    seed_points = pv.PolyData(valid_seeds)

    print('Generating streamlines...')
    print(f'  Max steps: {max_fiber_length}')
    print(f'  Step size: {initial_step_size} to {max_step_size}')

    streamlines_mesh = grid.streamlines_from_source(
        seed_points,
        vectors='vectors',
        max_steps=max_fiber_length,
        initial_step_length=initial_step_size,
        max_step_length=max_step_size,
        integration_direction='both',
        terminal_speed=terminal_speed,
        surface_streamlines=False,
        interpolator_type='cell',  # cell locator is more robust than point locator
        compute_vorticity=False,  # not needed for line visualization, saves computation
        progress_bar=True,
    )

    print(
        f'Generated {streamlines_mesh.n_lines} fibers with {streamlines_mesh.n_points} total points'
    )

    plotter = pv.Plotter()

    if show_volume:
        plotter.add_volume(
            grid,
            scalars='intensity',
            opacity='linear',
            cmap='gray',
            opacity_unit_distance=20,
        )

    plotter.add_mesh(
        streamlines_mesh,
        scalars='fan_rgb',
        rgb=True,
        line_width=2,
        render_lines_as_tubes=False,
        show_scalar_bar=False,
    )

    if show_starting_points:
        plotter.add_mesh(
            seed_points, color='red', point_size=8, render_points_as_spheres=True
        )

    plotter.camera_position = camera_position
    plotter.add_text(
        f'Fiber Visualization ({streamlines_mesh.n_lines} fibers)', font_size=12
    )
    plotter.show()
