"""
Volumetric visualization using K3D.

!!! quote "Reference"
    Volumetric visualization uses K3D:
    [Github page](https://github.com/K3D-tools/K3D-jupyter)

"""

import k3d
import matplotlib.pyplot as plt
import numpy as np
import pygel3d
import plotly.graph_objects as go
from matplotlib.colors import Colormap

from qim3d.utils._decorators import coarseness
from qim3d.utils._logger import log
from qim3d.utils._misc import downscale_img, scale_to_float16


@coarseness('volume')
def volumetric(
    volume: np.ndarray,
    aspectmode: str = 'data',
    show: bool = True,
    save: bool = False,
    grid_visible: bool = False,
    colormap: str = 'magma',
    constant_opacity: bool = False,
    opacity_function: str | list = None,
    min_value: float | None = None,
    max_value: float | None = None,
    samples: int | str = 'auto',
    max_voxels: int = 256**3,
    data_type: str = 'scaled_float16',
    camera_mode: str = 'orbit',
    **kwargs,
) -> k3d.Plot | None:
    """
    Renders a 3D volume using high-performance hardware-accelerated ray-casting.

    Creates an interactive 3D visualization in the browser using K3D. This function is ideal for inspecting complex voxel data, understanding 3D spatial relationships, or creating exportable HTML representations of a stack. It handles large datasets by automatically downsampling if the size exceeds a set threshold.

    **Key Features:**

    * **Browser-Based:** Renders directly in Jupyter notebooks or exports to standalone HTML.
    * **Performance:** Automatically manages sampling rates and data types (`float16`) for smooth interaction.
    * **Customization:** Supports custom colormaps, opacity transfer functions, and camera modes.

    Args:
        volume (numpy.ndarray): The 3D input data to be rendered.
        aspectmode (str, optional): Controls the proportions of the scene axes.

            * `'data'`: Axes are drawn in proportion to the physical data range.
            * `'cube'`: Axes are constrained to a cube regardless of data range.

        show (bool, optional): If `True`, displays the plot immediately.
        save (bool or str, optional): Controls saving the output.

            * **str**: Saves the visualization to the specified HTML file path.
            * **True**: Saves as 'snapshot.html' (default behavior of save).
            * **False**: Does not save the file.

        grid_visible (bool, optional): If `True`, displays a grid around the volume.
        colormap (str, matplotlib.colors.Colormap, or list, optional): Colormap for the rendering. Can be a Matplotlib name (e.g., 'magma') or object.
        constant_opacity (bool, optional): **Deprecated**. Use `opacity_function='constant'` instead.
        opacity_function (str or list, optional): Defines the transparency transfer function.

            * `'constant'`: Sets a uniform opacity for segmentation masks.
            * **list**: A specific list defining the opacity curve.

        min_value (float, optional): Minimum value for color scaling. If `None`, inferred from data.
        max_value (float, optional): Maximum value for color scaling. If `None`, inferred from data.
        samples (int or str, optional): Number of ray-marching samples.

            * `'auto'`: Calculates optimal samples based on volume size.
            * **int**: Specific number of samples (lower is faster, higher is better quality).

        max_voxels (int, optional): Maximum number of voxels allowed before downsampling occurs (defaults to approx. 16 million).
        data_type (str, optional): Internal data type for rendering. `'scaled_float16'` reduces memory usage.
        camera_mode (str, optional): Interaction mode for the camera (`'orbit'`, `'trackball'`, or `'fly'`).
        **kwargs: Additional keyword arguments passed to `k3d.plot`.

    Returns:
        plot (k3d.Plot):
            The K3D plot object. Returned if `show=False`.

    Raises:
        ValueError: If `aspectmode` is not 'data' or 'cube'.
        ValueError: If `camera_mode` is not 'orbit', 'trackball', or 'fly'.

    Tip:
        The function can be used for object label visualization using a `colormap` created with `qim3d.viz.colormaps.objects` along with setting `objects=True`. The latter ensures appropriate rendering.

    Example:
        Display a volume inline:
        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/fima-bone_128x128x128-20240221113459.html" width="100%" height="500" frameborder="0"></iframe>

        Save the rendering to an HTML file without displaying it:
        ```python
        plot = qim3d.viz.volumetric(vol, show=False, save="my_render.html")
        ```
    """

    pixel_count = volume.shape[0] * volume.shape[1] * volume.shape[2]
    # target is 60fps on m1 macbook pro, using test volume: https://data.qim.dk/pages/foam.html
    if samples == 'auto':
        y1, x1 = 256, 16777216  # 256 samples at res 256*256*256=16.777.216
        y2, x2 = 32, 134217728  # 32 samples at res 512*512*512=134.217.728

        # we fit linear function to the two points
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1

        samples = int(min(max(a * pixel_count + b, 64), 512))
    else:
        samples = int(samples)  # make sure it's an integer

    if aspectmode.lower() not in ['data', 'cube']:
        msg = "aspectmode should be either 'data' or 'cube'"
        raise ValueError(msg)

    if camera_mode not in ['orbit', 'trackball', 'fly']:
        msg = "camera_mode should be either 'orbit', 'trackbal' or 'fly'"
        raise ValueError(msg)

    # check if image should be downsampled for visualization
    original_shape = volume.shape
    volume = downscale_img(volume, max_voxels=max_voxels)

    new_shape = volume.shape

    if original_shape != new_shape:
        log.warning(
            f'Downsampled image for visualization, from {original_shape} to {new_shape}'
        )

    # Scale the image to float16 if needed
    if save:
        # When saving, we need float64
        volume = volume.astype(np.float64)
    else:
        if data_type == 'scaled_float16':
            volume = scale_to_float16(volume)
        else:
            volume = volume.astype(data_type)

    # Set color ranges
    color_range = [np.min(volume), np.max(volume)]
    if min_value:
        color_range[0] = min_value
    if max_value:
        color_range[1] = max_value

    # Handle the different formats that colormap can take
    if colormap:
        if isinstance(colormap, str):
            colormap = plt.get_cmap(colormap)  # Convert to Colormap object
        if isinstance(colormap, Colormap):
            # Convert to the format of colormap required by k3d.volume
            attr_vals = np.linspace(0.0, 1.0, num=colormap.N)
            rgb_vals = colormap(np.arange(0, colormap.N))[:, :3]
            colormap = np.column_stack((attr_vals, rgb_vals)).tolist()

    # Default k3d.volume settings
    interpolation = True

    if constant_opacity:
        log.warning(
            'Deprecation warning: Keyword argument "constant_opacity" is deprecated and will be removed next release. Instead use opacity_function="constant".'
        )
        # without these settings, the plot will look bad when colormap is created with qim3d.viz.colormaps.objects
        opacity_function = [0.0, float(constant_opacity), 1.0, float(constant_opacity)]
        interpolation = False
    else:
        if opacity_function == 'constant':
            # without these settings, the plot will look bad when colormap is created with qim3d.viz.colormaps.objects
            opacity_function = [0.0, float(True), 1.0, float(True)]
            interpolation = False
        elif opacity_function is None:
            opacity_function = []

    # Create the volume plot
    plt_volume = k3d.volume(
        volume,
        bounds=(
            [0, volume.shape[2], 0, volume.shape[1], 0, volume.shape[0]]
            if aspectmode.lower() == 'data'
            else None
        ),
        colormap=colormap,
        samples=samples,
        color_range=color_range,
        opacity_function=opacity_function,
        interpolation=interpolation,
    )
    plot = k3d.plot(grid_visible=grid_visible, **kwargs)
    plot += plt_volume
    plot.camera_mode = camera_mode
    if save:
        # Save html to disk
        with open(str(save), 'w', encoding='utf-8') as fp:
            fp.write(plot.get_snapshot())

    if show:
        plot.display()
    else:
        return plot


def mesh(
    mesh,
    backend: str = 'pygel3d',
    wireframe: bool = True,
    flat_shading: bool = True,
    grid_visible: bool = False,
    show: bool = True,
    save: bool = False,
    **kwargs,
) -> k3d.Plot | go.FigureWidget | None:
    """
    Visualizes a 3D surface mesh using either high-fidelity or fast interactive backends.

    Renders a polygonal surface model (Manifold) to inspect geometry, topology, or segmentation results. This function supports two rendering engines: 'pygel3d' for high-quality, smoothed aesthetic visualization, and 'k3d' for fast, hardware-accelerated performance suitable for large tessellations.

    **Key Features:**

    * **Dual Backends:** Choose between presentation-quality rendering (`pygel3d`) or performance-focused exploration (`k3d`).
    * **Geometry Inspection:** Toggle wireframes to inspect edge loops and vertex placement.
    * **Export:** Save K3D visualizations directly to HTML for sharing.

    Args:
        mesh (pygel3d.hmesh.Manifold): The input polygonal mesh object.
        backend (str, optional): The visualization engine to use.

            * `'pygel3d'`: High-quality rendering (default).
            * `'k3d'`: Fast, WebGL-based rendering.

        wireframe (bool, optional): If `True`, overlays the mesh edges (wireframe) on the surface.
        flat_shading (bool, optional): If `True`, renders faces with flat colors instead of smooth interpolation. Works only with `backend='k3d'`.
        grid_visible (bool, optional): If `True`, displays a background grid. Works only with `backend='k3d'`.
        show (bool, optional): If `True`, displays the plot inline immediately. Works only with `backend='k3d'`.
        save (bool or str, optional): Controls saving the output (Works only with `backend='k3d'`).

            * **str**: Saves the visualization to the specified HTML file path.
            * **True**: Saves as 'snapshot.html'.
            * **False**: Does not save the file.

        **kwargs: Additional backend-specific arguments.

            * **k3d**: Arguments passed to `k3d.plot`.
            * **pygel3d**: Arguments passed to `pygel3d.display`.

    Returns:
        plot (k3d.Plot or go.FigureWidget or None):
            The visualization object.

            * **k3d.Plot**: Returned if `backend='k3d'` and `show=False`.
            * **go.FigureWidget**: Returned if `backend='pygel3d'`.
            * **None**: Returned if `backend='k3d'` and `show=True`.

    Raises:
        ValueError: If `backend` is not 'pygel3d' or 'k3d'.
        ValueError: If the mesh data is malformed (vertices or faces have incorrect shapes).

    Example:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.volume()

        # Convert the 3D numpy array to a Pygel3D mesh object
        mesh = qim3d.mesh.from_volume(synthetic_blob, mesh_precision=0.5)

        # Visualize the generated mesh
        qim3d.viz.mesh(mesh)
        ```
        ![pygel3d_visualization](../../assets/screenshots/viz-pygel_mesh.png)

        ```python
        qim3d.viz.mesh(mesh, backend='k3d', wireframe=False, flat_shading=False)
        ```
        ![k3d_visualization](../../assets/screenshots/viz-k3d_mesh.png)
    """

    if len(mesh.vertices()) > 100000:
        msg = f'The mesh has {len(mesh.vertices())} vertices, visualization may be slow. Consider using a smaller <mesh_precision> when computing the mesh.'
        log.info(msg)

    if backend not in ['k3d', 'pygel3d']:
        msg = "Invalid backend. Choose 'pygel3d' or 'k3d'."
        raise ValueError(msg)

    # Extract vertex positions and face indices
    face_indices = list(mesh.faces())
    vertices_array = np.array(mesh.positions())

    # Extract face vertex indices
    face_vertices = [
        list(mesh.circulate_face(int(fid), mode='v'))[:3] for fid in face_indices
    ]
    face_vertices = np.array(face_vertices, dtype=np.uint32)

    # Validate the mesh structure
    if vertices_array.shape[1] != 3 or face_vertices.shape[1] != 3:
        msg = 'Vertices must have shape (N, 3) and faces (M, 3)'
        raise ValueError(msg)

    # Separate valid kwargs for each backend
    valid_k3d_kwargs = {k: v for k, v in kwargs.items() if k not in ['smooth', 'data']}
    valid_pygel_kwargs = {k: v for k, v in kwargs.items() if k in ['smooth', 'data']}

    if backend == 'k3d':
        vertices_array = np.ascontiguousarray(vertices_array.astype(np.float32))
        face_vertices = np.ascontiguousarray(face_vertices)

        mesh_plot = k3d.mesh(
            vertices=vertices_array,
            indices=face_vertices,
            wireframe=wireframe,
            flat_shading=flat_shading,
        )

        # Create plot
        plot = k3d.plot(grid_visible=grid_visible, **valid_k3d_kwargs)
        plot += mesh_plot

        if save:
            # Save html to disk
            with open(str(save), 'w', encoding='utf-8') as fp:
                fp.write(plot.get_snapshot())

        if show:
            plot.display()
        else:
            return plot

    elif backend == 'pygel3d':
        jd.set_export_mode(True)
        return jd.display(mesh, wireframe=wireframe, **valid_pygel_kwargs)
