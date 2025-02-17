import numpy as np
from skimage import measure, filters
from pygel3d import hmesh
from typing import Tuple, Any
from qim3d.utils._logger import log


def from_volume(
    volume: np.ndarray,
    **Kwargs
) -> hmesh.Manifold:
    
    if volume.ndim != 3:
        raise ValueError("The input volume must be a 3D numpy array.")
    
    mesh = hmesh.volumetric_isocontour(volume)
    return mesh