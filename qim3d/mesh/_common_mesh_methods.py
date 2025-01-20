import numpy as np
from skimage import measure, filters
import trimesh
from typing import Tuple, Any
from qim3d.utils._logger import log


def from_volume(
    volume: np.ndarray,
    level: float = None,
    step_size: int = 1,
    allow_degenerate: bool = False,
    padding: Tuple[int, int, int] = (2, 2, 2),
    **kwargs: Any,
) -> trimesh.Trimesh:
    """
    Convert a volume to a mesh using the Marching Cubes algorithm, with optional thresholding and padding.

    Args:
        volume (np.ndarray): The 3D numpy array representing the volume.
        level (float, optional): The threshold value for Marching Cubes. If None, Otsu's method is used.
        step_size (int, optional): The step size for the Marching Cubes algorithm.
        allow_degenerate (bool, optional): Whether to allow degenerate (i.e. zero-area) triangles in the end-result. If False, degenerate triangles are removed, at the cost of making the algorithm slower. Default False.
        padding (tuple of ints, optional): Padding to add around the volume.
        **kwargs: Additional keyword arguments to pass to `skimage.measure.marching_cubes`.

    Returns:
        mesh (trimesh.Trimesh): The generated mesh.

    Example:
        ```python
        import qim3d
        vol = qim3d.generate.noise_object(base_shape=(128,128,128),
                                  final_shape=(128,128,128),
                                  noise_scale=0.03,
                                  order=1,
                                  gamma=1,
                                  max_value=255,
                                  threshold=0.5,
                                  dtype='uint8'
                                  )
        mesh = qim3d.mesh.from_volume(vol, step_size=3)
        qim3d.viz.mesh(mesh.vertices, mesh.faces)
        ```
        <iframe src="https://platform.qim.dk/k3d/mesh_visualization.html" width="100%" height="500" frameborder="0"></iframe>
    """
    if volume.ndim != 3:
        raise ValueError("The input volume must be a 3D numpy array.")

    # Compute the threshold level if not provided
    if level is None:
        level = filters.threshold_otsu(volume)
        log.info(f"Computed level using Otsu's method: {level}")

    # Apply padding to the volume
    if padding is not None:
        pad_z, pad_y, pad_x = padding
        padding_value = np.min(volume)
        volume = np.pad(
            volume,
            ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)),
            mode="constant",
            constant_values=padding_value,
        )
        log.info(f"Padded volume with {padding} to shape: {volume.shape}")

    # Call skimage.measure.marching_cubes with user-provided kwargs
    verts, faces, normals, values = measure.marching_cubes(
        volume, level=level, step_size=step_size, allow_degenerate=allow_degenerate, **kwargs
    )

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Fix face orientation to ensure normals point outwards
    trimesh.repair.fix_inversion(mesh, multibody=True) 

    return mesh
