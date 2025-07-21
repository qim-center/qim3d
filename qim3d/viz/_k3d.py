"""
Volumetric visualization using K3D.

!!! quote "Reference"
    Volumetric visualization uses K3D:
    [Github page](https://github.com/K3D-tools/K3D-jupyter)

"""

import k3d
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pygel3d
from matplotlib.colors import Colormap
from pygel3d import jupyter_display as jd

from qim3d.utils._logger import log
from qim3d.utils._misc import downscale_img, scale_to_float16


def volumetric(
    img: np.ndarray,
    aspectmode: str = 'data',
    show: bool = True,
    save: bool = False,
    grid_visible: bool = False,
    color_map: str = 'magma',
    constant_opacity: bool = False,
    opacity_function: str | list = None,
    vmin: float | None = None,
    vmax: float | None = None,
    samples: int | str = 'auto',
    max_voxels: int = 512**3,
    data_type: str = 'scaled_float16',
    **kwargs,
) -> k3d.Plot | None:
    """
    Visualizes a 3D volume using volumetric rendering.

    Args:
        img (numpy.ndarray): The input 3D image data. It should be a 3D numpy array.
        aspectmode (str, optional): Determines the proportions of the scene's axes. Defaults to "data".
            If `'data'`, the axes are drawn in proportion with the axes' ranges.
            If `'cube'`, the axes are drawn as a cube, regardless of the axes' ranges.
        show (bool, optional): If True, displays the visualization inline. Defaults to True.
        save (bool or str, optional): If True, saves the visualization as an HTML file.
            If a string is provided, it's interpreted as the file path where the HTML
            file will be saved. Defaults to False.
        grid_visible (bool, optional): If True, the grid is visible in the plot. Defaults to False.
        color_map (str or matplotlib.colors.Colormap or list, optional): The color map to be used for the volume rendering. If a string is passed, it should be a matplotlib colormap name. Defaults to 'magma'.
        constant_opacity (bool): Set to True if doing an object label visualization with a corresponding color_map; otherwise, the plot may appear poorly. Defaults to False.
        opacity_function (str or list, optional): Applies an opacity function to the plot, enabling custom values for opaqueness. Set to True if doing an object label visualization with a corresponding color_map; otherwise, the plot may appear poorly. Defaults to [].
        vmin (float or None, optional): Together with vmax defines the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        vmax (float or None, optional): Together with vmin defines the data range the colormap covers. By default colormap covers the full range. Defaults to None
        samples (int or 'auto', optional): The number of samples to be used for the volume rendering in k3d. Input 'auto' for auto selection. Defaults to 'auto'.
            Lower values will render faster but with lower quality.
        max_voxels (int, optional): Defaults to 512^3.
        data_type (str, optional): Default to 'scaled_float16'.
        **kwargs (Any): Additional keyword arguments to be passed to the `k3d.plot` function.

    Returns:
        plot (k3d.plot): If `show=False`, returns the K3D plot object.

    Raises:
        ValueError: If `aspectmode` is not `'data'` or `'cube'`.

    Tip:
        The function can be used for object label visualization using a `color_map` created with `qim3d.viz.colormaps.objects` along with setting `objects=True`. The latter ensures appropriate rendering.

    Example:
        Display a volume inline:

        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/fima-bone_128x128x128-20240221113459.html" width="100%" height="500" frameborder="0"></iframe>

        Save a plot to an HTML file:

        ```python
        import qim3d
        vol = qim3d.examples.bone_128x128x128
        plot = qim3d.viz.volumetric(vol, show=False, save="plot.html")
        ```

    """

    pixel_count = img.shape[0] * img.shape[1] * img.shape[2]
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
    # check if image should be downsampled for visualization
    original_shape = img.shape
    img = downscale_img(img, max_voxels=max_voxels)

    new_shape = img.shape

    if original_shape != new_shape:
        log.warning(
            f'Downsampled image for visualization, from {original_shape} to {new_shape}'
        )

    # Scale the image to float16 if needed
    if save:
        # When saving, we need float64
        img = img.astype(np.float64)
    else:
        if data_type == 'scaled_float16':
            img = scale_to_float16(img)
        else:
            img = img.astype(data_type)

    # Set color ranges
    color_range = [np.min(img), np.max(img)]
    if vmin:
        color_range[0] = vmin
    if vmax:
        color_range[1] = vmax

    # Handle the different formats that color_map can take
    if color_map:
        if isinstance(color_map, str):
            color_map = plt.get_cmap(color_map)  # Convert to Colormap object
        if isinstance(color_map, Colormap):
            # Convert to the format of color_map required by k3d.volume
            attr_vals = np.linspace(0.0, 1.0, num=color_map.N)
            rgb_vals = color_map(np.arange(0, color_map.N))[:, :3]
            color_map = np.column_stack((attr_vals, rgb_vals)).tolist()

    # Default k3d.volume settings
    interpolation = True

    if constant_opacity:
        log.warning(
            'Deprecation warning: Keyword argument "constant_opacity" is deprecated and will be removed next release. Instead use opacity_function="constant".'
        )
        # without these settings, the plot will look bad when color_map is created with qim3d.viz.colormaps.objects
        opacity_function = [0.0, float(constant_opacity), 1.0, float(constant_opacity)]
        interpolation = False
    else:
        if opacity_function == 'constant':
            # without these settings, the plot will look bad when color_map is created with qim3d.viz.colormaps.objects
            opacity_function = [0.0, float(True), 1.0, float(True)]
            interpolation = False
        elif opacity_function is None:
            opacity_function = []

    # Create the volume plot
    plt_volume = k3d.volume(
        img,
        bounds=(
            [0, img.shape[2], 0, img.shape[1], 0, img.shape[0]]
            if aspectmode.lower() == 'data'
            else None
        ),
        color_map=color_map,
        samples=samples,
        color_range=color_range,
        opacity_function=opacity_function,
        interpolation=interpolation,
    )
    plot = k3d.plot(grid_visible=grid_visible, **kwargs)
    plot += plt_volume
    if save:
        # Save html to disk
        with open(str(save), 'w', encoding='utf-8') as fp:
            fp.write(plot.get_snapshot())

    if show:
        plot.display()
    else:
        return plot


def mesh(
    mesh: pygel3d.hmesh.Manifold,
    backend: str = 'pygel3d',
    wireframe: bool = True,
    flat_shading: bool = True,
    grid_visible: bool = False,
    show: bool = True,
    save: bool = False,
    **kwargs,
) -> k3d.Plot | go.FigureWidget | None:
    """
    Visualize a 3D mesh using `pygel3d` or `k3d`. The visualization with the pygel3d backend provides higher-quality rendering, but it may take more time compared to using the k3d backend.

    Args:
        mesh (pygel3d.hmesh.Manifold): The input mesh object.
        backend (str, optional): The visualization backend to use.
            Choose between `pygel3d` (default) and `k3d`.
        wireframe (bool, optional): If True, displays the mesh as a wireframe.
            Works both with backend `pygel3d` and `k3d`. Defaults to True.
        flat_shading (bool, optional): If True, applies flat shading to the mesh.
            Works only with backend `k3d`. Defaults to True.
        grid_visible (bool, optional): If True, shows a grid in the visualization.
            Works only with backend `k3d`. Defaults to False.
        show (bool, optional): If True, displays the visualization inline, useful for multiple plots.
            Works only with backend `k3d`. Defaults to True.
        save (bool or str, optional): If True, saves the visualization as an HTML file.
            If a string is provided, it's interpreted as the file path where the HTML
            file will be saved. Works only with the backend `k3d`. Defaults to False.
        **kwargs (Any): Additional keyword arguments specific to the chosen backend:

            - `k3d.plot` kwargs: Arguments that customize the [`k3d.plot`](https://k3d-jupyter.org/reference/factory.plot.html) visualization.

            - `pygel3d.display` kwargs: Arguments that customize the [`pygel3d.display`](https://www2.compute.dtu.dk/projects/GEL/PyGEL/pygel3d/jupyter_display.html#display) visualization.

    Returns:
        k3d.Plot or None:

            - If `backend="k3d"`, returns a `k3d.Plot` object.
            - If `backend="pygel3d"`, the function displays the mesh but does not return a plot object.

    Raises:
        ValueError: If `backend` is not `pygel3d` or `k3d`.

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
