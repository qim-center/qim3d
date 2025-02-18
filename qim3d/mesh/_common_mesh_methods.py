import numpy as np
from skimage import measure, filters
from pygel3d import hmesh
from typing import Tuple, Any
from qim3d.utils._logger import log


def from_volume(
    volume: np.ndarray,
    **kwargs: any
) -> hmesh.Manifold:
    """ Convert a 3D numpy array to a mesh object using the [volumetric_isocontour](https://www2.compute.dtu.dk/projects/GEL/PyGEL/pygel3d/hmesh.html#volumetric_isocontour) function from Pygel3D.

    Args:
        volume (np.ndarray): A 3D numpy array representing a volume.
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
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)

        # Convert the 3D numpy array to a Pygel3D mesh object
        mesh = qim3d.mesh.from_volume(synthetic_blob)
        ```
    """
    
    if volume.ndim != 3:
        raise ValueError("The input volume must be a 3D numpy array.")
    
    if volume.size == 0:
        raise ValueError("The input volume must not be empty.")

    mesh = hmesh.volumetric_isocontour(volume, **kwargs)
    return mesh