import numpy as np
import scipy
import scipy.ndimage
from pygel3d import hmesh

from qim3d.utils import log


def from_volume(
    volume: np.ndarray, mesh_precision: float = 1.0, **kwargs: any
) -> hmesh.Manifold:
    """
    Converts a 3D binary or grayscale volume into a polygon mesh (isosurface extraction).

    This function transforms voxel-based data into a vector-based surface representation (triangular mesh). This process, often called polygonization or tessellation, is a necessary step for 3D printing (exporting to STL), finite element analysis (FEA), or surface-based geometric measurements. It utilizes the [`volumetric_isocontour`](https://www2.compute.dtu.dk/projects/GEL/PyGEL/pygel3d/hmesh.html#volumetric_isocontour) function from PyGEL3D to generate a high-quality manifold.

    Args:
        volume (np.ndarray): The 3D input array representing the voxel grid.
        mesh_precision (float, optional): A scaling factor between 0.0 and 1.0 to adjust the resolution of the volume before meshing.
            
            * **1.0**: Uses the original resolution (most detailed, highest polygon count).
            * **< 1.0**: Downsamples the volume (e.g., 0.5 reduces size by half) to create a coarser, lighter mesh with fewer triangles.
        
        **kwargs: Additional keyword arguments passed to `pygel3d.hmesh.volumetric_isocontour`.

    Raises:
        ValueError: If the input is not 3D, is empty, or if `mesh_precision` is outside the (0, 1] range.

    Returns:
        mesh (hmesh.Manifold):
            The generated mesh object containing vertices, edges, and faces.

    Example:
        Convert a 3D numpy array to a Pygel3D mesh object:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.volume()

        # Visualize the generated blob
        qim3d.viz.volumetric(synthetic_blob)
        ```
        ![pygel3d_visualization_vol](../../assets/screenshots/viz-pygel_mesh_vol.png){width='300', length='200'}

        ```python
        # Convert the 3D numpy array to a Pygel3D mesh object
        mesh = qim3d.mesh.from_volume(synthetic_blob, mesh_precision=0.5)

        # Visualize the generated mesh
        qim3d.viz.mesh(mesh)
        ```
        ![pygel3d_visualization_mesh](../../assets/screenshots/viz-pygel_mesh.png){width='300', length='200'}
    """

    if volume.ndim != 3:
        msg = 'The input volume must be a 3D numpy array.'
        raise ValueError(msg)

    if volume.size == 0:
        msg = 'The input volume must not be empty.'
        raise ValueError(msg)

    if not (0 < mesh_precision <= 1):
        msg = 'The mesh precision must be between 0 and 1.'
        raise ValueError(msg)

    # Apply scaling to adjust mesh resolution
    volume = scipy.ndimage.zoom(volume, zoom=mesh_precision, order=0)

    mesh = hmesh.volumetric_isocontour(volume, **kwargs)

    return mesh
