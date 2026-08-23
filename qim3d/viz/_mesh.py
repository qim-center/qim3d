
from pygel3d import jupyter_display as jd
import pygel3d
import pyvista as pv

from qim3d.mesh._common_mesh_methods import SurfaceMesh, VolumeMesh
from qim3d.utils._logger import log



def mesh(
    mesh: pygel3d.hmesh.Manifold | SurfaceMesh | VolumeMesh,
    wireframe: bool = False,
    show_edges: bool = True,
    show: bool = True,
    save_screenshot: str = '',
    export_html: str = '',
    explode: int = 0,
    smooth_shading: bool = False,
    face_color: str = '#cccccc',
    edge_color: str = '#993333',
    **kwargs,
) -> None:
    """
    Visualize a 3D mesh using `pygel3d` or `pyvista`. If you need more advanced tools, use pyvista directly.

    Args:
        mesh (pygel3d.hmesh.Manifold | SurfaceMesh | VolumeMesh): The input mesh object.
        wireframe (bool, optional): If True, displays the mesh as a wireframe. Defaults to False.
        show_edges (bool, optional): If True, shows edges of the mesh. Defaults to True.
        show (bool, optional): If True, displays the visualization inline, useful for multiple plots.
            Works only with backend `pyvista`. Defaults to True.
        save_screenshot (str, optional): If True, saves the visualization as a `png` file.
            The string is interpreted as the file path where the screenshot will 
            be saved. Works only with the backend `pyvista`. Defaults to ''.
        export_html (str, optional): If True, saves the visualization as an `html` file.
            The string is interpreted as the file path where the scene will 
            be saved. Works only with the backend `pyvista`. Defaults to ''.
        explode (int, optional): Only works when mesh is qim3d.mesh.VolumeMesh.
            Defines how spread are the tetrahedrons. If 0, the volume is intact.
            Defaults to 1.
        smooth_shading (bool, optional): Smooths out edges. Only works with `pyvista`.
            Defaults to False.
        face_color (str, optional): Face color of the mesh. Only works with `pyvista`.
            Doesn't work with `wireframe = True`. Defaults to '#cccccc'.
        edge_color (str, optional): Edge color of the mesh. Only works with `pyvista`.
            Defaults to '#993333'.
        **kwargs (Any): Additional keyword arguments specific to the chosen backend:
            - `pyvista` kwargs: Arguments that customize the [`pyvista`](https://docs.pyvista.org/api/plotting) visualization.
            - `pygel3d.display` kwargs: Arguments that customize the [`pygel3d.display`](https://www2.compute.dtu.dk/projects/GEL/PyGEL/pygel3d/jupyter_display.html#display) visualization.

    Returns:
        None: The function displays the mesh but does not return a plot object.


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
    """

    if isinstance(mesh, (VolumeMesh, SurfaceMesh)):
        plotter = pv.Plotter()

        if isinstance(mesh, VolumeMesh):
            mesh = mesh.explode(explode)

        if wireframe:
            kwargs['style'] =  'wireframe'
        plotter.add_mesh(mesh, 
                         show_edges = show_edges, 
                         smooth_shading = smooth_shading,
                         show_scalar_bar=False,
                         color = face_color,
                         edge_color=edge_color, 
                         **kwargs)

        if show:
            plotter.show()

        if save_screenshot:
            if not save_screenshot.endswith('png'):
                save_screenshot = save_screenshot + '.png'
            plotter.screenshot(save_screenshot)

        if export_html:
            if not export_html.endswith('.html'):
                export_html = export_html + '.html'
            plotter.export_html(export_html)

        return

    if isinstance(mesh, pygel3d.hmesh.Manifold):
        if len(mesh.vertices()) > 100000:
            msg = f'The mesh has {len(mesh.vertices())} vertices, visualization may be slow. Consider using a smaller <mesh_precision> when computing the mesh.'
            log.info(msg)

        jd.set_export_mode(True)
        valid_pygel_kwargs = {k: v for k, v in kwargs.items() if k in ['smooth', 'data']}
        return jd.display(mesh, wireframe=show_edges, **valid_pygel_kwargs)