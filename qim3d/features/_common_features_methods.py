import numpy as np
import qim3d.processing
from qim3d.utils._logger import log
import qim3d
from pygel3d import hmesh

def volume(obj: np.ndarray|hmesh.Manifold) -> float:
    """
    Compute the volume of a 3D mesh using the Pygel3D library.

    Args:
        obj: Either a np.ndarray volume or a mesh object of type hmesh.Manifold.

    Returns:
        volume (float): The volume of the object. 
    """

    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj)

    return hmesh.volume(obj)

def area(obj: np.ndarray|hmesh.Manifold) -> float:
    """
    Compute the surface area of a 3D mesh using the Pygel3D library.

    Args:
        obj: Either a np.ndarray volume or a mesh object of type hmesh.Manifold.

    Returns:
        area (float): The surface area of the object. 
    """

    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj)

    return hmesh.area(obj)

def sphericity_pygel3d(obj: np.ndarray|hmesh.Manifold) -> float:
    """
    Compute the sphericity of a 3D mesh using the Pygel3D library.

    Args:
        obj: Either a np.ndarray volume or a mesh object of type hmesh.Manifold.

    Returns:
        sphericity (float): The sphericity of the object. 
    """

    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume_pygel3d(obj)

    volume = volume(obj)
    area = area(obj)

    if area == 0:
        log.warning("Surface area is zero, sphericity is undefined.")
        return np.nan

    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area
    log.info(f"Sphericity: {sphericity}")
    return sphericity