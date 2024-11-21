import numpy as np
import scipy.ndimage
from tqdm.notebook import tqdm

import qim3d.generate
from qim3d.utils.logger import log


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

        # Increment attempt counter
        j += 1

    return collection, placed


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


def collection(
    collection_shape: tuple = (200, 200, 200),
    num_objects: int = 15,
    positions: list[tuple] = None,
    min_shape: tuple = (40, 40, 40),
    max_shape: tuple = (60, 60, 60),
    object_shape_zoom: tuple = (1.0, 1.0, 1.0),
    min_object_noise: float = 0.02,
    max_object_noise: float = 0.05,
    min_rotation_degrees: int = 0,
    max_rotation_degrees: int = 360,
    rotation_axes: list[tuple] = [(0, 1), (0, 2), (1, 2)],
    min_gamma: float = 0.8,
    max_gamma: float = 1.2,
    min_high_value: int = 128,
    max_high_value: int = 255,
    min_threshold: float = 0.5,
    max_threshold: float = 0.6,
    smooth_borders: bool = False,
    object_shape: str = None,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, object]:
    """
    Generate a 3D volume of multiple synthetic objects using Perlin noise.

    Args:
        collection_shape (tuple, optional): Shape of the final collection volume to generate. Defaults to (200, 200, 200).
        num_objects (int, optional): Number of synthetic objects to include in the collection. Defaults to 15.
        positions (list[tuple], optional): List of specific positions as (z, y, x) coordinates for the objects. If not provided, they are placed randomly into the collection. Defaults to None.
        min_shape (tuple, optional): Minimum shape of the objects. Defaults to (40, 40, 40).
        max_shape (tuple, optional): Maximum shape of the objects. Defaults to (60, 60, 60).
        object_shape_zoom (tuple, optional): Scaling factors for each dimension of each object. Defaults to (1.0, 1.0, 1.0).
        min_object_noise (float, optional): Minimum scale factor for Perlin noise. Defaults to 0.02.
        max_object_noise (float, optional): Maximum scale factor for Perlin noise. Defaults to 0.05.
        min_rotation_degrees (int, optional): Minimum rotation angle in degrees. Defaults to 0.
        max_rotation_degrees (int, optional): Maximum rotation angle in degrees. Defaults to 360.
        rotation_axes (list[tuple], optional): List of axis pairs that will be randomly chosen to rotate around. Defaults to [(0, 1), (0, 2), (1, 2)].
        min_gamma (float, optional): Minimum gamma correction factor. Defaults to 0.8.
        max_gamma (float, optional): Maximum gamma correction factor. Defaults to 1.2.
        min_high_value (int, optional): Minimum maximum value for the volume intensity. Defaults to 128.
        max_high_value (int, optional): Maximum maximum value for the volume intensity. Defaults to 255.
        min_threshold (float, optional): Minimum threshold value for clipping low intensity values. Defaults to 0.5.
        max_threshold (float, optional): Maximum threshold value for clipping low intensity values. Defaults to 0.6.
        smooth_borders (bool, optional): Flag for smoothing object borders to avoid straight edges in the objects. If True, the `min_threshold` and `max_threshold` parameters are ignored. Defaults to False.
        object_shape (str, optional): Shape of the object to generate, either "cylinder", or "tube". Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to 0.
        verbose (bool, optional): Flag to enable verbose logging. Defaults to False.

    Returns:
        synthetic_collection (numpy.ndarray): 3D volume of the generated collection of synthetic objects with specified parameters.
        labels (numpy.ndarray): Array with labels for each voxel, same shape as synthetic_collection.

    Raises:
        TypeError: If `collection_shape` is not 3D.
        ValueError: If object parameters are invalid.

    Note:
        - The function places objects without overlap.
        - The function can either place objects at random positions in the collection (if `positions = None`) or at specific positions provided in the `positions` argument. If specific positions are provided, the number of blobs must match the number of positions (e.g. `num_objects = 2` with `positions = [(12, 8, 10), (24, 20, 18)]`).
        - If not all `num_objects` can be placed, the function returns the `synthetic_collection` volume with as many objects as possible in it, and logs an error.
        - Labels for all objects are returned, even if they are not a single connected component.

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of objects
        num_objects = 15
        synthetic_collection, labels = qim3d.generate.collection(num_objects = num_objects)

        # Visualize synthetic collection
        qim3d.viz.vol(synthetic_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_default.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        qim3d.viz.slicer(synthetic_collection)
        ```
        ![synthetic_collection](assets/screenshots/synthetic_collection_default.gif)

        ```python
        # Visualize labels
        cmap = qim3d.viz.colormaps.objects(nlabels=num_objects)
        qim3d.viz.slicer(labels, cmap=cmap, vmax=num_objects)
        ```
        ![synthetic_collection](assets/screenshots/synthetic_collection_default_labels.gif)

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of dense objects
        synthetic_collection, labels = qim3d.generate.collection(
                                    min_high_value = 255,
                                    max_high_value = 255,
                                    min_object_noise = 0.05,
                                    max_object_noise = 0.05,
                                    min_threshold = 0.99,
                                    max_threshold = 0.99,
                                    min_gamma = 0.02,
                                    max_gamma = 0.02)

        # Visualize synthetic collection
        qim3d.viz.vol(synthetic_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_dense.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of cylindrical structures
        vol, labels = qim3d.generate.collection(num_objects = 40,
                                                collection_shape = (300, 150, 150),
                                                min_shape = (280, 10, 10),
                                                max_shape = (290, 15, 15),
                                                min_object_noise = 0.08,
                                                max_object_noise = 0.09,
                                                max_rotation_degrees = 5,
                                                min_threshold = 0.7,
                                                max_threshold = 0.9,
                                                min_gamma = 0.10,
                                                max_gamma = 0.11,
                                                object_shape = "cylinder"
                                                )

        # Visualize synthetic collection
        qim3d.viz.vol(vol)

        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_cylinder.html" width="100%" height="500" frameborder="0"></iframe>
        
        ```python
        # Visualize slices
        qim3d.viz.slices(vol, n_slices=15)
        ```
        ![synthetic_collection_cylinder](assets/screenshots/synthetic_collection_cylinder_slices.png)    
        
    Example:
        ```python
        import qim3d

        # Generate synthetic collection of tubular (hollow) structures
        vol, labels = qim3d.generate.collection(num_objects = 10,
                                                collection_shape = (200, 200, 200),
                                                min_shape = (180, 25, 25),
                                                max_shape = (190, 35, 35),
                                                min_object_noise = 0.02,
                                                max_object_noise = 0.03,
                                                max_rotation_degrees = 5,
                                                min_threshold = 0.7,
                                                max_threshold = 0.9,
                                                min_gamma = 0.10,
                                                max_gamma = 0.11,
                                                object_shape = "tube"
                                                )

        # Visualize synthetic collection
        qim3d.viz.vol(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_collection_tube.html" width="100%" height="500" frameborder="0"></iframe>
        
        ```python
        # Visualize slices
        qim3d.viz.slices(vol, n_slices=15, axis=1)
        ```
        ![synthetic_collection_tube](assets/screenshots/synthetic_collection_tube_slices.png)
    """
    if verbose:
        original_log_level = log.getEffectiveLevel()
        log.setLevel("DEBUG")

    # Check valid input types
    if not isinstance(collection_shape, tuple) or len(collection_shape) != 3:
        raise TypeError(
            "Shape of collection must be a tuple with three dimensions (z, y, x)"
        )

    if len(min_shape) != len(max_shape):
        raise ValueError("Object shapes must be tuples of the same length")

    if (positions is not None) and (len(positions) != num_objects):
        raise ValueError(
            "Number of objects must match number of positions, otherwise set positions = None"
        )

    # Set seed for random number generator
    rng = np.random.default_rng(seed)

    # Initialize the 3D array for the shape
    collection_array = np.zeros(
        (collection_shape[0], collection_shape[1], collection_shape[2]), dtype=np.uint8
    )
    labels = np.zeros_like(collection_array)

    # Fill the 3D array with synthetic blobs
    for i in tqdm(range(num_objects), desc="Objects placed"):
        log.debug(f"\nObject #{i+1}")

        # Sample from blob parameter ranges
        if min_shape == max_shape:
            blob_shape = min_shape
        else:
            blob_shape = tuple(
                rng.integers(low=min_shape[i], high=max_shape[i]) for i in range(3)
            )
        log.debug(f"- Blob shape: {blob_shape}")

        # Scale object shape
        final_shape = tuple(l * r for l, r in zip(blob_shape, object_shape_zoom))
        final_shape = tuple(int(x) for x in final_shape) # NOTE: Added this 

        # Sample noise scale
        noise_scale = rng.uniform(low=min_object_noise, high=max_object_noise)
        log.debug(f"- Object noise scale: {noise_scale:.4f}")

        gamma = rng.uniform(low=min_gamma, high=max_gamma)
        log.debug(f"- Gamma correction: {gamma:.3f}")

        if max_high_value > min_high_value:
            max_value = rng.integers(low=min_high_value, high=max_high_value)
        else:
            max_value = min_high_value
        log.debug(f"- Max value: {max_value}")

        threshold = rng.uniform(low=min_threshold, high=max_threshold)
        log.debug(f"- Threshold: {threshold:.3f}")

        # Generate synthetic object
        blob = qim3d.generate.blob(
            base_shape=blob_shape,
            final_shape=final_shape,
            noise_scale=noise_scale,
            gamma=gamma,
            max_value=max_value,
            threshold=threshold,
            smooth_borders=smooth_borders,
            object_shape=object_shape,
        )

        # Rotate object
        if max_rotation_degrees > 0:
            angle = rng.uniform(
                low=min_rotation_degrees, high=max_rotation_degrees
            )  # Sample rotation angle
            axes = rng.choice(rotation_axes)  # Sample the two axes to rotate around
            log.debug(f"- Rotation angle: {angle:.2f} at axes: {axes}")

            blob = scipy.ndimage.rotate(blob, angle, axes, order=1)

        # Place synthetic object into the collection
        # If positions are specified, place object at one of the specified positions
        collection_before = collection_array.copy()
        if positions:
            collection_array, placed, positions = specific_placement(
                collection_array, blob, positions
            )

        # Otherwise, place object at a random available position
        else:
            collection_array, placed = random_placement(collection_array, blob, rng)

        # Break if object could not be placed
        if not placed:
            break

        # Update labels
        new_labels = np.where(collection_array != collection_before, i + 1, 0).astype(
            labels.dtype
        )
        labels += new_labels

    if not placed:
        # Log error if not all num_objects could be placed (this line of code has to be here, otherwise it will interfere with tqdm progress bar)
        log.error(
            f"Object #{i+1} could not be placed in the collection, no space found. Collection contains {i}/{num_objects} objects."
        )

    if verbose:
        log.setLevel(original_log_level)

    return collection_array, labels
