import numpy as np
from skimage import measure, filters
import trimesh
from typing import Tuple, Any


def create_mesh(
    volume: np.ndarray,
    level: float = None,
    step_size=1,
    padding: Tuple[int, int, int] = (2, 2, 2),
    **kwargs: Any,
) -> trimesh.Trimesh:
    """
    Convert a volume to a mesh using the Marching Cubes algorithm, with optional thresholding and padding.

    Args:
        volume (np.ndarray): The 3D numpy array representing the volume.
        level (float, optional): The threshold value for Marching Cubes. If None, Otsu's method is used.
        padding (tuple of int, optional): Padding to add around the volume.
        **kwargs: Additional keyword arguments to pass to `skimage.measure.marching_cubes`.

    Returns:
        trimesh: The generated mesh.

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
        mesh = qim3d.processing.create_mesh(vol step_size=3)
        qim3d.viz.mesh(mesh.vertices, mesh.faces)
        ```

    """
    if volume.ndim != 3:
        raise ValueError("The input volume must be a 3D numpy array.")

    # Compute the threshold level if not provided
    if level is None:
        level = filters.threshold_otsu(volume)
        print(f"Computed level using Otsu's method: {level}")

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
        print(f"Padded volume with {padding} to shape: {volume.shape}")

    # Call skimage.measure.marching_cubes with user-provided kwargs
    verts, faces, normals, values = measure.marching_cubes(
        volume, level=level, step_size=step_size, **kwargs
    )

    print(len(verts))

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    return mesh
