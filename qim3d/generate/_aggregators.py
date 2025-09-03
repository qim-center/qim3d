import numpy as np
import scipy.ndimage
from tqdm.notebook import tqdm

import qim3d.generate
from qim3d.utils._logger import log

__all__ = ['volume_collection']


def random_placement(
    collection: np.ndarray,
    blob: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """
    Place blob at random available position in collection.

    Args:
        collection (numpy.ndarray): 3D volume of the collection.
        blob (numpy.ndarray): 3D volume of the blob.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        collection (numpy.ndarray): 3D volume of the collection with the blob placed.
        placed (bool): Flag for placement success.

    """
    # Find available (zero) elements in collection
    available_z, available_y, available_x = np.where(collection == 0)

    # Flag for placement success
    placed = False

    # Default mid position
    mid = np.array(blob.shape) // 2

    # Attempt counter
    j = 1

    while (not placed) and (j < 200_000):
        # Select a random available position in collection
        idx = rng.choice(len(available_z))
        z, y, x = available_z[idx], available_y[idx], available_x[idx]

        start = np.array([z, y, x])  # Start position of blob placement
        end = start + np.array(blob.shape)  # End position of blob placement

        # Check if the blob fits in the selected region (without overlap)
        if np.all(
            collection[start[0] : end[0], start[1] : end[1], start[2] : end[2]] == 0
        ):
            # Check if placement is within bounds (bool)
            within_bounds = np.all(start >= 0) and np.all(
                end <= np.array(collection.shape)
            )

            if within_bounds:
                # Place blob
                collection[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = (
                    blob
                )
                placed = True
                mid = start + np.array(blob.shape) // 2

        # Increment attempt counter
        j += 1

    return collection, placed, mid


def specific_placement(
    collection: np.ndarray,
    blob: np.ndarray,
    positions: list[tuple],
) -> tuple[np.ndarray, bool]:
    """
    Place blob at one of the specified positions in the collection.

    Args:
        collection (numpy.ndarray): 3D volume of the collection.
        blob (numpy.ndarray): 3D volume of the blob.
        positions (list[tuple]): List of specified positions as (z, y, x) coordinates for the blobs.

    Returns:
        collection (numpy.ndarray): 3D volume of the collection with the blob placed.
        placed (bool): Flag for placement success.
        positions (list[tuple]): List of remaining positions to place blobs.

    """
    # Flag for placement success
    placed = False

    for position in positions:
        # Get coordinates of next position
        z, y, x = position

        # Place blob with center at specified position
        start = (
            np.array([z, y, x]) - np.array(blob.shape) // 2
        )  # Start position of blob placement
        end = start + np.array(blob.shape)  # End position of blob placement

        # Check if the blob fits in the selected region (without overlap)
        if np.all(
            collection[start[0] : end[0], start[1] : end[1], start[2] : end[2]] == 0
        ):
            # Check if placement is within bounds (bool)
            within_bounds = np.all(start >= 0) and np.all(
                end <= np.array(collection.shape)
            )

            if within_bounds:
                # Place blob
                collection[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = (
                    blob
                )
                placed = True

                # Remove position from list
                positions.remove(position)
                break

    return collection, placed, positions


def _volume_collection(
    collection_shape: tuple = (200, 200, 200),
    num_volumes: int = 15,
    data: np.ndarray | list[np.ndarray] | None = None,
    positions: list[tuple] = None,
    min_shape: tuple = (40, 40, 40),
    max_shape: tuple = (60, 60, 60),
    volume_shape_zoom: tuple = (1.0, 1.0, 1.0),
    min_volume_noise: float = 0.02,
    max_volume_noise: float = 0.05,
    min_rotation_degrees: int = 0,
    max_rotation_degrees: int = 360,
    rotation_axes: list[tuple] = None,
    min_gamma: float = 0.8,
    max_gamma: float = 1.2,
    min_high_value: int = 128,
    max_high_value: int = 255,
    min_threshold: float = 0.5,
    max_threshold: float = 0.6,
    smooth_borders: bool = False,
    volume_shape: str = None,
    seed: int = 0,
    verbose: bool = False,
    return_positions: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3D volume of multiple synthetic volumes using Perlin noise.

    Args:
        collection_shape (tuple of ints, optional): Shape of the final collection volume to generate. Defaults to (200, 200, 200).
        num_volumes (int, optional): Number of synthetic volumes to include in the collection. Defaults to 15.
        data (numpy.ndarray or list[numpy.ndarray], optional): Predefined volume(s) to use for the collection. If provided, the function will use these volumes instead of generating new ones. Defaults to None.
        positions (list[tuple], optional): List of specific positions as (z, y, x) coordinates for the volumes. If not provided, they are placed randomly into the collection. Defaults to None.
        min_shape (tuple of ints, optional): Minimum shape of the volumes. Defaults to (40, 40, 40).
        max_shape (tuple of ints, optional): Maximum shape of the volumes. Defaults to (60, 60, 60).
        volume_shape_zoom (tuple of floats, optional): Scaling factors for each dimension of each volume. Defaults to (1.0, 1.0, 1.0).
        min_volume_noise (float, optional): Minimum scale factor for Perlin noise. Defaults to 0.02.
        max_volume_noise (float, optional): Maximum scale factor for Perlin noise. Defaults to 0.05.
        min_rotation_degrees (int, optional): Minimum rotation angle in degrees. Defaults to 0.
        max_rotation_degrees (int, optional): Maximum rotation angle in degrees. Defaults to 360.
        rotation_axes (list[tuple], optional): List of axis pairs that will be randomly chosen to rotate around. Defaults to [(0, 1), (0, 2), (1, 2)].
        min_gamma (float, optional): Minimum gamma correction factor. Defaults to 0.8.
        max_gamma (float, optional): Maximum gamma correction factor. Defaults to 1.2.
        min_high_value (int, optional): Minimum maximum value for the volume intensity. Defaults to 128.
        max_high_value (int, optional): Maximum maximum value for the volume intensity. Defaults to 255.
        min_threshold (float, optional): Minimum threshold value for clipping low intensity values. Defaults to 0.5.
        max_threshold (float, optional): Maximum threshold value for clipping low intensity values. Defaults to 0.6.
        smooth_borders (bool, optional): Flag for smoothing volume borders to avoid straight edges in the volumes. If True, the `min_threshold` and `max_threshold` parameters are ignored. Defaults to False.
        volume_shape (str or None, optional): Shape of the volume to generate, either "cylinder", or "tube". Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to 0.
        verbose (bool, optional): Flag to enable verbose logging. Defaults to False.
        return_positions (bool, optional): Flag to return position of randomly placed blobs.

    Returns:
        volume_collection (numpy.ndarray): 3D volume of the generated collection of synthetic volumes with specified parameters.
        labels (numpy.ndarray): Array with labels for each voxel, same shape as volume_collection.

    Raises:
        TypeError: If `data` is not a numpy array or list of numpy arrays.
        ValueError: If `data` is provided but not a 3D numpy array.
        TypeError: If `collection_shape` is not 3D.
        ValueError: If volume parameters are invalid.

    Note:
        - The function places volumes without overlap.
        - The function can either place volumes at random positions in the collection (if `positions = None`) or at specific positions provided in the `positions` argument. If specific positions are provided, the number of blobs must match the number of positions (e.g. `num_volumes = 2` with `positions = [(12, 8, 10), (24, 20, 18)]`).
        - If not all `num_volumes` can be placed, the function returns the `volume_collection` volume with as many volumes as possible in it, and logs an error.
        - Labels for all volumes are returned, even if they are not a single connected component.

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        num_volumes = 15
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = num_volumes)

        # Visualize the collection
        qim3d.viz.volumetric(volume_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_default.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        qim3d.viz.slicer(volume_collection)
        ```
        ![synthetic_collection](../../assets/screenshots/synthetic_collection_default.gif)

        ```python
        # Visualize labels
        cmap = qim3d.viz.colormaps.segmentation(num_labels=num_volumes)
        qim3d.viz.slicer(labels, color_map=cmap, value_max=num_volumes)
        ```
        ![synthetic_collection](../../assets/screenshots/synthetic_collection_default_labels.gif)

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of dense objects
        volume_collection, labels = qim3d.generate.volume_collection(
            min_high_value = 255,
            max_high_value = 255,
            min_volume_noise = 0.05,
            max_volume_noise = 0.05,
            min_threshold = 0.99,
            max_threshold = 0.99,
            min_gamma = 0.02,
            max_gamma = 0.02
            )

        # Visualize the collection
        qim3d.viz.volumetric(volume_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_dense.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of cylindrical structures
        volume_collection, labels = qim3d.generate.volume_collection(
            num_volumes = 40,
            collection_shape = (300, 150, 150),
            min_shape = (280, 10, 10),
            max_shape = (290, 15, 15),
            min_volume_noise = 0.08,
            max_volume_noise = 0.09,
            max_rotation_degrees = 5,
            min_threshold = 0.7,
            max_threshold = 0.9,
            min_gamma = 0.10,
            max_gamma = 0.11,
            volume_shape = "cylinder"
            )

        # Visualize the collection
        qim3d.viz.volumetric(volume_collection)

        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_cylinder.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(volume_collection, num_slices=15)
        ```
        ![synthetic_collection_cylinder](../../assets/screenshots/synthetic_collection_cylinder_slices.png)

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of tubular (hollow) structures
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 10,
                                                collection_shape = (200, 200, 200),
                                                min_shape = (180, 25, 25),
                                                max_shape = (190, 35, 35),
                                                min_volume_noise = 0.02,
                                                max_volume_noise = 0.03,
                                                max_rotation_degrees = 5,
                                                min_threshold = 0.7,
                                                max_threshold = 0.9,
                                                min_gamma = 0.10,
                                                max_gamma = 0.11,
                                                volume_shape = "tube"
                                                )

        # Visualize the collection
        qim3d.viz.volumetric(volume_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_tube.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(volume_collection, num_slices=15, slice_axis=1)
        ```
        ![synthetic_collection_tube](../../assets/screenshots/synthetic_collection_tube_slices.png)

    """

    if data is not None:
        if isinstance(data, np.ndarray):  # one volume
            data_list = [data]
        elif isinstance(data, list):  # a list of volumes
            data_list = data
        else:
            msg = '`data` must be a numpy array or list of arrays'
            raise TypeError(msg)
        for idx, arr in enumerate(data_list):
            if not (
                isinstance(arr, np.ndarray) and arr.ndim == 3
            ):  # check that they must be numpy arrays and 3D
                msg = f'data[{idx}] must be a 3D numpy array'
                raise ValueError(msg)
    else:
        data_list = None

    if data_list is not None:
        valid = []
        for idx, vol in enumerate(data_list):
            # check each dimension
            if all(v <= c for v, c in zip(vol.shape, collection_shape)):
                valid.append(vol)
            else:
                msg = f'Skipping custom volume {idx} with shape {vol.shape} — larger than collection {collection_shape}'
                log.warning(msg)
        data_list = valid
        # if none remain, we can't build anything
        if not data_list:
            msg = f'No custom volumes fit within collection size {collection_shape}.'
            raise ValueError(msg)

    if rotation_axes is None:
        rotation_axes = [(0, 1), (0, 2), (1, 2)]

    if verbose:
        original_log_level = log.getEffectiveLevel()
        log.setLevel('DEBUG')

    # Check valid input types
    if not isinstance(collection_shape, tuple) or len(collection_shape) != 3:
        message = 'Shape of collection must be a tuple with three dimensions (z, y, x)'
        raise TypeError(message)

    if len(min_shape) != len(max_shape):
        message = 'Object shapes must be tuples of the same length'
        raise ValueError(message)

    if (positions is not None) and (len(positions) != num_volumes):
        message = 'Number of volumes must match number of positions, otherwise set positions = None'
        raise ValueError(message)

    if (positions is not None) and return_positions:
        log.debug('positions are given and thus not returned')
        return_positions = False

    # Set seed for random number generator
    rng = np.random.default_rng(seed)

    # Set seed for random number generator for placement
    rng_pos = np.random.default_rng(seed)

    # Initialize the 3D array for the shape
    collection_array = np.zeros(
        (collection_shape[0], collection_shape[1], collection_shape[2]), dtype=np.uint8
    )
    labels = np.zeros_like(collection_array)

    # Initialize saved positions
    placed_positions = []

    # Fill the 3D array with synthetic blobs
    for i in tqdm(range(num_volumes), desc='Objects placed'):
        log.debug(f'\nObject #{i+1}')

        # Sample from blob parameter ranges
        if min_shape == max_shape:
            blob_shape = min_shape
        else:
            blob_shape = tuple(
                rng.integers(low=min_shape[i], high=max_shape[i]) for i in range(3)
            )
        log.debug(f'- Blob shape: {blob_shape}')

        # Scale volume shape
        final_shape = tuple(
            dim * zoom for dim, zoom in zip(blob_shape, volume_shape_zoom)
        )
        final_shape = tuple(int(x) for x in final_shape)  # NOTE: Added this

        # Sample noise scale
        noise_scale = rng.uniform(low=min_volume_noise, high=max_volume_noise)
        log.debug(f'- Object noise scale: {noise_scale:.4f}')

        gamma = rng.uniform(low=min_gamma, high=max_gamma)
        log.debug(f'- Gamma correction: {gamma:.3f}')

        if max_high_value > min_high_value:
            max_value = rng.integers(low=min_high_value, high=max_high_value)
        else:
            max_value = min_high_value
        log.debug(f'- Max value: {max_value}')

        threshold = rng.uniform(low=min_threshold, high=max_threshold)
        log.debug(f'- Threshold: {threshold:.3f}')

        # Pick volume from the list if provided, otherwise generate synthetic volume
        if data_list is not None:
            choice_i = rng.integers(len(data_list))
            blob = data_list[choice_i].copy()
        else:
            blob = qim3d.generate.volume(
                base_shape=blob_shape,
                final_shape=final_shape,
                noise_scale=noise_scale,
                gamma=gamma,
                max_value=max_value,
                threshold=threshold,
                smooth_borders=smooth_borders,
                volume_shape=volume_shape,
            )

        # Rotate volume
        if max_rotation_degrees > 0:
            angle = rng.uniform(
                low=min_rotation_degrees, high=max_rotation_degrees
            )  # Sample rotation angle
            axes = rng.choice(rotation_axes)  # Sample the two axes to rotate around
            log.debug(f'- Rotation angle: {angle:.2f} at axes: {axes}')

            blob = scipy.ndimage.rotate(blob, angle, axes, order=1)

        # Place synthetic volume into the collection
        # If positions are specified, place volume at one of the specified positions
        collection_before = collection_array.copy()
        if positions:
            collection_array, placed, positions = specific_placement(
                collection_array, blob, positions
            )

        # Otherwise, place volume at a random available position
        else:
            collection_array, placed, pos = random_placement(
                collection_array, blob, rng_pos
            )
            if return_positions and placed:
                placed_positions.append(tuple(pos))

        # Break if volume could not be placed
        if not placed:
            break

        # Update labels
        new_labels = np.where(collection_array != collection_before, i + 1, 0).astype(
            labels.dtype
        )
        labels += new_labels

    if not placed:
        # Log error if not all num_volumes could be placed (this line of code has to be here, otherwise it will interfere with tqdm progress bar)
        log.error(
            f'Object #{i+1} could not be placed in the collection, no space found. Collection contains {i}/{num_volumes} volumes.'
        )

    if verbose:
        log.setLevel(original_log_level)

    if return_positions:
        return collection_array, labels, placed_positions
    else:
        return collection_array, labels


def volume_collection(
    num_volumes: int = 15,
    collection_shape: tuple = (200, 200, 200),
    data: np.ndarray | list[np.ndarray] | None = None,
    positions: list[tuple] = None,
    shape_range: tuple[tuple] = ((40, 40, 40), (60, 60, 60)),
    shape_magnification_range: tuple[float] = (1.0, 1.0),
    noise_type: str = 'perlin',
    noise_range: tuple[float] = (0.02, 0.03),
    rotation_degree_range: tuple[int] = (0, 360),
    rotation_axes: list[tuple] = None,
    gamma_range: tuple[float] = (0.9, 1),
    value_range: tuple[int] = (128, 255),
    threshold_range: tuple[float] = (0.5, 0.55),
    decay_rate_range: tuple[float] = (5, 10),
    shape: str = None,
    tube_hole_ratio: float = 0.5,
    axis: int = 0,
    verbose: bool = False,
    same_seed: bool = False,
    hollow: bool = False,
    seed: int = 0,
    dtype: str = 'uint8',
    return_positions: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3D volume of multiple synthetic volumes using Perlin or Simplex noise.

    Args:
        num_volumes (int, optional): Number of synthetic volumes to include in the collection. Defaults to 15.
        collection_shape (tuple of ints, optional): Shape of the final collection volume to generate. Defaults to (200, 200, 200).
        data (numpy.ndarray or list[numpy.ndarray], optional): Predefined volume(s) to use for the collection. If provided, the function will use these volumes instead of generating new ones. Defaults to None.
        positions (list[tuple], optional): List of specific positions as (z, y, x) coordinates for the volumes. If not provided, they are placed randomly into the collection. Defaults to None.
        shape_range (tuple of tuple of ints, optional): Determines the shape of the generated volumes with first element defining the minimum size and second element defining maximum. Defaults to ((40,40,40), (60,60,60)).
        shape_magnification_range (tuple of floats, optional): Range for scaling of volume shape in all dimensions. Defaults to (1.0, 1.0).
        noise_type (str, optional): Type of noise to be used for volume generation. Should be `simplex`, `perlin` or `mixed`. Defaults to perlin.
        noise_range (tuple of floats, optional): Determines range for noise. First element is minimum and second is maximum. Defaults to (0.02, 0.03).
        rotation_degree_range (tuple of ints, optional): Determines range for rotation angle in degrees. First element is minimum and second is maximum. Defaults to (0, 360).
        rotation_axes (list[tuple], optional): List of axis pairs that will be randomly chosen to rotate around. Defaults to [(0, 1), (0, 2), (1, 2)].
        gamma_range (tuple of floats, optional): Determines minimum and maximum gamma correctness factor. Defaults to (0.9, 1.0)
        value_range (tuple of ints, optional): Determines minimum and maximum value for volume intensity. Defaults to (128, 255).
        threshold_range (tuple of ints, optional): Determines minimum and maximum value for thresholding. Defaults to (0.5, 0.55).
        decay_rate_range (float, optional): Determines minimum and maximum value for the decay_range. Defaults to (5,10).
        shape (str or None, optional): Shape of the volume to generate, either "cylinder", or "tube". Defaults to None.
        tube_hole_ratio (float, optional): Ratio for the inverted fade mask used to generate tubes. Will only have an effect if shape=`tube`. Defaults to 0.5.
        axis (int, optional): Determines the axis of the volume_shape if this is defined. Defaults to 0.
        verbose (bool, optional): Flag to enable verbose logging. Defaults to False.
        same_seed (bool, optional): Use the same seed for each generated volume. Note that in order to generate identical volumes, the min and max for the different parameters should be identical.
        hollow (bool, optional): Create hollow objects using qim3d.operations.make_hollow(). Defaults to False.
        seed (int, optional): Seed for reproducibility. Defaults to 0. Each generated volume will be generated with a randomly selected sub-seed generated from the original seed.
        dtype (str, optional): dtype for resulting volume. Defaults to uint8.
        return_positions (bool, optional): Flag to return position of randomly placed blobs.

    Returns:
        volume_collection (numpy.ndarray): 3D volume of the generated collection of synthetic volumes with specified parameters.
        labels (numpy.ndarray): Array with labels for each voxel, same shape as volume_collection.

    Raises:
        TypeError: If `data` is not a numpy array or list of numpy arrays.
        ValueError: If `data` is provided but not a 3D numpy array.
        ValueError: If `noise_type` is invalid.
        TypeError: If `collection_shape` is not 3D.
        ValueError: If volume parameters are invalid.
        ValueError: If the `shape_range` is incorrectly defined.
        ValueError: If the `positions` are incorrectly defined.

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        num_volumes = 15
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes=num_volumes)

        # Visualize the collection
        qim3d.viz.volumetric(volume_collection, grid_visible=True)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_default_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        qim3d.viz.slicer(volume_collection)
        ```
        ![synthetic_collection](../../assets/screenshots/synthetic_collection_default.gif)

        ```python
        # Visualize labels
        cmap = qim3d.viz.colormaps.segmentation(num_labels=num_volumes)
        qim3d.viz.slicer(labels, color_map=cmap, value_max=num_volumes)
        ```
        ![synthetic_collection](../../assets/screenshots/synthetic_collection_default_labels.gif)

    Example: Collection of fiber-like structures
        ```python
        import qim3d

        # Generate synthetic collection of cylindrical structures
        volume_collection, labels = qim3d.generate.volume_collection(
            num_volumes = 40,
            collection_shape = (300, 150, 150),
            shape_range = ((280, 10, 10), (290, 15, 15)),
            noise_range = (0.06,0.09),
            rotation_degree_range = (0,5),
            threshold_range = (0.1,0.3),
            gamma_range = (0.10, 0.20),
            shape = "cylinder"
            )

        # Visualize the collection
        qim3d.viz.volumetric(volume_collection)

        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_cylinder_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(volume_collection, num_slices=15)
        ```
        ![synthetic_collection_cylinder](../../assets/screenshots/synthetic_collection_cylinder_slices.png)

    Example: Create a collection of tubular (hollow) structures
        ```python
        import qim3d

        # Generate synthetic collection of tubular (hollow) structures
        volume_collection, labels = qim3d.generate.volume_collection(
            num_volumes = 10,
            collection_shape = (200, 200, 200),
            shape_range = ((185,35,35), (190,45,45)),
            noise_range = (0.02, 0.03),
            rotation_degree_range = (0,5),
            threshold_range = (0.6, 0.7),
            gamma_range = (0.1, 0.11),
            shape = "tube",
            tube_hole_ratio = 0.15,
            )

        # Visualize the collection
        qim3d.viz.volumetric(volume_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_tube_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(volume_collection, num_slices=15, slice_axis=1)
        ```
        ![synthetic_collection_tube](../../assets/screenshots/synthetic_collection_tube_slices.png)

    Example: Using predefined volumes
        ```python
        import qim3d

        # Generate two unique volumes to be used
        volume_1 = qim3d.generate.volume(base_shape = (32,32,32), noise_scale = 0.0)
        volume_2 = qim3d.generate.volume(base_shape = (32,32,32), noise_scale = 0.2)

        # Generate collection from predefined volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 30,
                                                                     data = [volume_1, volume_2])
        # Visualize
        qim3d.viz.volumetric(volume_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_from_given_volumes.html" width="100%" height="500" frameborder="0"></iframe>

    """

    # Check valid input types
    if data is not None:
        if isinstance(data, np.ndarray):
            data_list = [data]
        elif isinstance(data, list):
            data_list = data
        else:
            msg = '`data` must be a numpy array or list of arrays'
            raise TypeError(msg)
        for idx, arr in enumerate(data_list):
            if not (isinstance(arr, np.ndarray) and arr.ndim == 3):
                msg = f'data[{idx}] must be a 3D numpy array'
                raise ValueError(msg)
    else:
        data_list = None

    if data_list is not None:
        valid = []
        for idx, vol in enumerate(data_list):
            # check each dimension
            if all(v <= c for v, c in zip(vol.shape, collection_shape)):
                valid.append(vol)
            else:
                msg = f'Skipping custom volume {idx} with shape {vol.shape} — larger than collection {collection_shape}'
                log.warning(msg)
        data_list = valid
        # if none remain, we can't build anything
        if not data_list:
            msg = f'No custom volumes fit within collection size {collection_shape}.'
            raise ValueError(msg)

    noise_types = ['pnoise', 'perlin', 'p', 'snoise', 'simplex', 's', 'mixed', 'm']
    if noise_type not in noise_types:
        err = f'noise_type should be one of: {noise_types}'
        raise ValueError(err)

    if not isinstance(collection_shape, tuple) or len(collection_shape) != 3:
        message = 'Shape of collection must be a tuple with three dimensions (z, y, x)'
        raise TypeError(message)

    if len(shape_range[0]) != len(shape_range[1]):
        message = 'Object shapes must be tuples of the same length'
        raise ValueError(message)
    if len(shape_range[0]) != 3 or len(shape_range[1]) != 3 or len(shape_range) != 2:
        message = 'shape_range should be defined as a tuple with two elements, each containing a tuple with three elements.'
        raise ValueError(message)

    if (positions is not None) and (len(positions) != num_volumes):
        message = 'Number of volumes must match number of positions, otherwise set positions = None'
        raise ValueError(message)

    if (positions is not None) and return_positions:
        log.debug('positions are given and thus not returned')
        return_positions = False

    if rotation_axes is None:
        rotation_axes = [(0, 1), (0, 2), (1, 2)]

    if verbose:
        original_log_level = log.getEffectiveLevel()
        log.setLevel('DEBUG')

    # Set seed for random number generator
    rng = np.random.default_rng(seed)

    # Set seed for random number generator for placement
    rng_pos = np.random.default_rng(seed)

    # Initialize the 3D array for the shape
    collection_array = np.zeros(
        (collection_shape[0], collection_shape[1], collection_shape[2]), dtype=dtype
    )
    labels = np.zeros_like(collection_array)

    # Initialize saved positions
    placed_positions = []

    if same_seed:
        seeds = rng.integers(0, 255, size=1).repeat(5000)
    else:
        seeds = rng.integers(low=0, high=255, size=5000)
    nt = rng.random(size=1000)

    # Fill the 3D array with synthetic blobs
    for i in tqdm(range(num_volumes), desc='Objects placed'):
        log.debug(f'\nObject #{i+1}')

        # Sample from blob parameter ranges
        if shape_range[0] == shape_range[1]:
            blob_shape = shape_range[0]
        else:
            blob_shape = tuple(
                rng.integers(low=shape_range[0][i], high=shape_range[1][i])
                for i in range(3)
            )

        magnification = rng.uniform(
            low=shape_magnification_range[0], high=shape_magnification_range[1]
        )

        final_shape = tuple(int(dim * magnification) for dim in blob_shape)
        log.debug(f'- Blob shape: {final_shape}')
        # Check if should keep final_shape separate from base_shape
        # Sample noise scale
        noise_scale = rng.uniform(low=noise_range[0], high=noise_range[1])
        log.debug(f'- Object noise scale: {noise_scale:.4f}')

        gamma = rng.uniform(low=gamma_range[0], high=gamma_range[1])
        log.debug(f'- Gamma correction: {gamma:.3f}')

        threshold = rng.uniform(low=threshold_range[0], high=threshold_range[1])
        log.debug(f'- Threshold: {threshold:.3f}')

        decay_rate = rng.uniform(low=decay_rate_range[0], high=decay_rate_range[1])
        log.debug(f'- Decay rate: {decay_rate:.3f}')

        if value_range[1] > value_range[0]:
            max_value = rng.integers(low=value_range[0], high=value_range[1])
        else:
            max_value = value_range[0]
        log.debug(f'- Max value: {max_value}')

        if noise_type == 'mixed' or noise_type == 'm':
            nti = 'perlin' if nt[i] >= 0.5 else 'simplex'
        else:
            nti = noise_type
        log.debug(f'- Noise type: {nti}')

        log.debug(f'- Seed: {seeds[i]}')

        # Pick volume from the list if provided, otherwise generate synthetic volume
        if data_list is not None:
            choice_i = rng.integers(len(data_list))
            blob = data_list[choice_i].copy()
        else:
            # Generate synthetic volume
            blob = qim3d.generate.volume(
                base_shape=final_shape,
                final_shape=final_shape,
                noise_scale=noise_scale,
                noise_type=nti,
                decay_rate=decay_rate,
                gamma=gamma,
                threshold=threshold,
                max_value=max_value,
                shape=shape,
                tube_hole_ratio=tube_hole_ratio,
                axis=axis,
                order=1,
                dtype=dtype,
                hollow=hollow,
                seed=seeds[i],
            )

        # Rotate volume
        if rotation_degree_range[1] > 0:
            angle = rng.uniform(
                low=rotation_degree_range[0], high=rotation_degree_range[1]
            )  # Sample rotation angle
            axes = rng.choice(rotation_axes)  # Sample the two axes to rotate around
            log.debug(f'- Rotation angle: {angle:.2f} at axes: {axes}')

            blob = scipy.ndimage.rotate(blob, angle, axes, order=1)

        # Place synthetic volume into the collection
        # If positions are specified, place volume at one of the specified positions
        collection_before = collection_array.copy()
        if positions:
            collection_array, placed, positions = specific_placement(
                collection_array, blob, positions.copy()
            )

        # Otherwise, place volume at a random available position
        else:
            collection_array, placed, pos = random_placement(
                collection_array, blob, rng_pos
            )
            if return_positions and placed:
                placed_positions.append(tuple(pos))

            log.debug(f'- Center placement (z,y,x): {pos}')
        # Break if volume could not be placed
        if not placed:
            break

        # Update labels
        new_labels = np.where(collection_array != collection_before, i + 1, 0).astype(
            labels.dtype
        )
        labels += new_labels

    if not placed:
        # Log error if not all num_volumes could be placed (this line of code has to be here, otherwise it will interfere with tqdm progress bar)
        log.error(
            f'Object #{i+1} could not be placed in the collection, no space found. Collection contains {i}/{num_volumes} volumes.'
        )
    if verbose:
        log.setLevel(original_log_level)

    if return_positions:
        return collection_array, labels, placed_positions
    else:
        return collection_array, labels
