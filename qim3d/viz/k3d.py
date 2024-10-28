"""
Volumetric visualization using K3D

!!! quote "Reference"
    Volumetric visualization uses K3D:
    [Github page](https://github.com/K3D-tools/K3D-jupyter)

"""

import numpy as np
from qim3d.utils.logger import log
from qim3d.utils.misc import downscale_img, scale_to_float16


def vol(
    img,
    aspectmode="data",
    show=True,
    save=False,
    grid_visible=False,
    cmap=None,
    vmin=None,
    vmax=None,
    samples="auto",
    max_voxels=512**3,
    data_type="scaled_float16",
    **kwargs,
):
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
        cmap (list, optional): The color map to be used for the volume rendering. Defaults to None.
        vmin (float, optional): Together with vmax defines the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        vmax (float, optional): Together with vmin defines the data range the colormap covers. By default colormap covers the full range. Defaults to None
        samples (int, optional): The number of samples to be used for the volume rendering in k3d. Defaults to 512.
            Lower values will render faster but with lower quality.
        max_voxels (int, optional): Defaults to 512^3.
        data_type (str, optional): Default to 'scaled_float16'.
        **kwargs: Additional keyword arguments to be passed to the `k3d.plot` function.

    Returns:
        plot (k3d.plot): If `show=False`, returns the K3D plot object.

    Raises:
        ValueError: If `aspectmode` is not `'data'` or `'cube'`.

    Example:
        Display a volume inline:

        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.vol(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/fima-bone_128x128x128-20240221113459.html" width="100%" height="500" frameborder="0"></iframe>

        Save a plot to an HTML file:

        ```python
        import qim3d
        vol = qim3d.examples.bone_128x128x128
        plot = qim3d.viz.vol(vol, show=False, save="plot.html")
        ```

    """
    import k3d

    pixel_count = img.shape[0] * img.shape[1] * img.shape[2]
    # target is 60fps on m1 macbook pro, using test volume: https://data.qim.dk/pages/foam.html
    if samples == "auto":
        y1, x1 = 256, 16777216  # 256 samples at res 256*256*256=16.777.216
        y2, x2 = 32, 134217728  # 32 samples at res 512*512*512=134.217.728

        # we fit linear function to the two points
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1

        samples = int(min(max(a * pixel_count + b, 64), 512))
    else:
        samples = int(samples)  # make sure it's an integer

    if aspectmode.lower() not in ["data", "cube"]:
        raise ValueError("aspectmode should be either 'data' or 'cube'")
    # check if image should be downsampled for visualization
    original_shape = img.shape
    img = downscale_img(img, max_voxels=max_voxels)

    new_shape = img.shape

    if original_shape != new_shape:
        log.warning(
            f"Downsampled image for visualization, from {original_shape} to {new_shape}"
        )

    # Scale the image to float16 if needed
    if save:
        # When saving, we need float64
        img = img.astype(np.float64)
    else:

        if data_type == "scaled_float16":
            img = scale_to_float16(img)
        else:
            img = img.astype(data_type)

    # Set color ranges
    color_range = [np.min(img), np.max(img)]
    if vmin:
        color_range[0] = vmin
    if vmax:
        color_range[1] = vmax

    # Create the volume plot
    plt_volume = k3d.volume(
        img,
        bounds=(
            [0, img.shape[2], 0, img.shape[1], 0, img.shape[0]]
            if aspectmode.lower() == "data"
            else None
        ),
        color_map=cmap,
        samples=samples,
        color_range=color_range,
    )
    plot = k3d.plot(grid_visible=grid_visible, **kwargs)
    plot += plt_volume
    if save:
        # Save html to disk
        with open(str(save), "w", encoding="utf-8") as fp:
            fp.write(plot.get_snapshot())

    if show:
        plot.display()
    else:
        return plot  


def mesh(
    verts,
    faces,
    wireframe=True,
    flat_shading=True,
    grid_visible=False,
    show=True,
    save=False,
    **kwargs,
):
    """
    Visualizes a 3D mesh using K3D.

    Args:
        verts (numpy.ndarray): A 2D array (Nx3) containing the vertices of the mesh.
        faces (numpy.ndarray): A 2D array (Mx3) containing the indices of the mesh faces.
        wireframe (bool, optional): If True, the mesh is rendered as a wireframe. Defaults to True.
        flat_shading (bool, optional): If True, flat shading is applied to the mesh. Defaults to True.
        grid_visible (bool, optional): If True, the grid is visible in the plot. Defaults to False.
        show (bool, optional): If True, displays the visualization inline. Defaults to True.
        save (bool or str, optional): If True, saves the visualization as an HTML file.
            If a string is provided, it's interpreted as the file path where the HTML
            file will be saved. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the `k3d.plot` function.

    Returns:
        plot (k3d.plot): If `show=False`, returns the K3D plot object.

    Example:
        ```python
        import qim3d

        vol = qim3d.generate.blob(base_shape=(128,128,128),
                                  final_shape=(128,128,128),
                                  noise_scale=0.03,
                                  order=1,
                                  gamma=1,
                                  max_value=255,
                                  threshold=0.5,
                                  dtype='uint8'
                                  )
        mesh = qim3d.processing.create_mesh(vol, step_size=3)
        qim3d.viz.mesh(mesh.vertices, mesh.faces)
        ```
        <iframe src="https://platform.qim.dk/k3d/mesh_visualization.html" width="100%" height="500" frameborder="0"></iframe>
    """
    import k3d

    # Validate the inputs
    if verts.shape[1] != 3:
        raise ValueError("Vertices array must have shape (N, 3)")
    if faces.shape[1] != 3:
        raise ValueError("Faces array must have shape (M, 3)")
    
    # Ensure the correct data types and memory layout
    verts = np.ascontiguousarray(verts.astype(np.float32))  # Cast and ensure C-contiguous layout
    faces = np.ascontiguousarray(faces.astype(np.uint32))    # Cast and ensure C-contiguous layout


    # Create the mesh plot
    plt_mesh = k3d.mesh(
        vertices=verts,
        indices=faces,
        wireframe=wireframe,
        flat_shading=flat_shading,
    )

    # Create plot
    plot = k3d.plot(grid_visible=grid_visible, **kwargs)
    plot += plt_mesh

    if save:
        # Save html to disk
        with open(str(save), "w", encoding="utf-8") as fp:
            fp.write(plot.get_snapshot())

    if show:
        plot.display()
    else:
        return plot
