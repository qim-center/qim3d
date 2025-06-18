import numpy as np
import scipy
import scipy.ndimage
from pygel3d import hmesh

from qim3d.utils import log


def from_volume(
    volume: np.ndarray, mesh_precision: float = 1.0, **kwargs: any
) -> hmesh.Manifold:
    """
    Convert a 3D numpy array to a mesh object using the [volumetric_isocontour](https://www2.compute.dtu.dk/projects/GEL/PyGEL/pygel3d/hmesh.html#volumetric_isocontour)
    function from Pygel3D.

    Args:
        volume (np.ndarray): A 3D numpy array representing a volume.
        mesh_precision (float, optional): Scaling factor for adjusting the resolution of the mesh.
                                          Default is 1.0 (no scaling).
        **kwargs: Additional arguments to pass to the Pygel3D volumetric_isocontour function.

    Raises:
        ValueError: If the input volume is not a 3D numpy array or if the input volume is empty.

    Returns:
        hmesh.Manifold: A Pygel3D mesh object representing the input volume.

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
