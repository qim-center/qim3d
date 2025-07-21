import numpy as np
import scipy


def pad(
    volume: np.ndarray, x_axis: float = 0, y_axis: float = 0, z_axis: float = 0
) -> np.ndarray:
    """
    Pads the input 3D volume.

    Args:
        volume (numpy.ndarray): The input 3D volume (shape: n, h, w).
        x_axis (float): Amount of pixels to pad the x-dimension. Must be an integer or a half-integer (e.g., 5, 5.5). The padding is symmetric and applied to both sides of the volume. Defaults to 0.
        y_axis (float): Amount of pixels to pad the y-dimension. Must be an integer or a half-integer. Defaults to 0.
        z_axis (float): Amount of pixels to pad the z-dimension. Must be an integer or a half-integer. Defaults to 0.

    Returns:
        numpy.ndarray: The padded volume.

    Raises:
        AssertionError: If the input volume is not 3D.
        AssertionError: If any padding value is negative.

    Example:
        ```python
        import qim3d
        import numpy as np

        vol = np.zeros((100, 100, 100))
        print(vol.shape)
        ```
        (100, 100, 100)
        ```python
        # Pad x-axis with 10 pixels on each side and y-axis with 20% of the original volume size
        padded_volume = qim3d.operations.pad(vol, x_axis=10, y_axis=vol.shape[1] * 0.1)
        print(padded_volume.shape)
        ```
        (100, 120, 120)

    """
    assert len(volume.shape) == 3, 'Volume must be 3D'
    assert z_axis >= 0, 'Padded shape must be positive in z-axis.'
    assert y_axis >= 0, 'Padded shape must be positive in y-axis.'
    assert x_axis >= 0, 'Padded shape must be positive in x-axis.'

    n, h, w = volume.shape

    # Round to nearest half integer
    x_axis = round(x_axis * 2) / 2
    y_axis = round(y_axis * 2) / 2
    z_axis = round(z_axis * 2) / 2

    # Add to both sides and determine new sizes
    new_w = w + int(2 * x_axis)
    new_h = h + int(2 * y_axis)
    new_n = n + int(2 * z_axis)

    # Create a new volume with padding and center the original in the padded volume
    padded_volume = np.zeros((new_n, new_h, new_w))
    padded_volume[
        int(z_axis) : int(z_axis) + n,
        int(y_axis) : int(y_axis) + h,
        int(x_axis) : int(x_axis) + w,
    ] = volume

    return padded_volume


def pad_to(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """
    Pads the input 3D volume to a certain shape.

    Args:
        volume (numpy.ndarray): The input 3D volume (shape: n, h, w).
        shape (tuple[int, int, int]): The shape to pad the volume to.

    Returns:
        padded_volume (numpy.ndarray): The padded volume.

    Raises:
        AssertionError: If the input shape is not 3D.
        AssertionError: If the input volume is not 3D.
        AssertionError: If the shape tuple is not integers.
        AssertionError: If the padded shape is not larger than the original shape.

    Example:
        ```python
        import qim3d
        import numpy as np

        # Create volume of shape (100,100,100) and add values in a box inside
        vol = np.zeros((100,100,100))
        print(vol.shape)
        ```
        (100, 100, 100)
        ```python
        # Pad the volume to shape (110, 110, 110)
        padded_volume = qim3d.operations.pad_to(vol, (110,110,110))
        print(padded_volume.shape)
        ```
        (110, 110, 110)

    """
    assert len(shape) == 3, 'Shape must be 3D'
    assert len(volume.shape) == 3, 'Volume must be 3D'
    assert all(isinstance(x, int) for x in shape), 'Shape tuple must contain integers'

    shape_np = np.array(shape)
    for i in range(len(shape_np)):
        if shape_np[i] < volume.shape[i]:
            print(
                'Pad shape is smaller than the volume shape. Changing it to original shape volume.'
            )
            shape_np[i] = volume.shape[i]

    new_z = (shape_np[0] - volume.shape[0]) / 2
    new_y = (shape_np[1] - volume.shape[1]) / 2
    new_x = (shape_np[2] - volume.shape[2]) / 2

    return pad(volume, x_axis=new_x, y_axis=new_y, z_axis=new_z)


def trim(volume: np.ndarray) -> np.ndarray:
    """
    Removes all empty slices (i.e., slices that contain all zeros) along the x, y, and z axes.

    Args:
        volume (numpy.ndarray): The 3D input volume (shape: n, h, w).

    Returns:
        trimmed_volume (numpy.ndarray): The transformed volume with empty slices removed along all axes.

    Raises:
        AssertionError: If the input shape is not 3D.

    Example:
        ```python
        import qim3d
        import numpy as np

        # Create volume of shape (100,100,100) and add values in a box inside
        vol = np.zeros((100,100,100))
        vol[10:90, 10:90, 10:90] = 1
        print(vol.shape)
        ```
        (100, 100, 100)
        ```python
        # Trim the slices without voxel values on all axes
        trimmed_volume = qim3d.operations.trim(vol)
        print(trimmed_volume.shape)
        ```
        (80, 80, 80)

    """
    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'

    # Remove empty slices along the x-axis (columns)
    non_empty_x = np.any(volume, axis=(1, 2))  # Check non-empty slices in the y-z plane
    volume = volume[non_empty_x, :, :]  # Keep only non-empty slices along x

    # Remove empty slices along the y-axis (rows)
    non_empty_y = np.any(volume, axis=(0, 2))  # Check non-empty slices in the x-z plane
    volume = volume[:, non_empty_y, :]  # Keep only non-empty slices along y

    # Remove empty slices along the z-axis (depth)
    non_empty_z = np.any(volume, axis=(0, 1))  # Check non-empty slices in the x-y plane
    volume = volume[:, :, non_empty_z]  # Keep only non-empty slices along z

    trimmed_volume = volume

    return trimmed_volume


def shear3d(
    volume: np.ndarray,
    x_shift_y: int = 0,
    x_shift_z: int = 0,
    y_shift_x: int = 0,
    y_shift_z: int = 0,
    z_shift_x: int = 0,
    z_shift_y: int = 0,
    order: int = 1,
) -> np.ndarray:
    """
    Applies a shear transformation to a 3D volume using pixel-based shifts.

    Args:
        volume (numpy.ndarray): The input 3D volume (shape: n, h, w).
        x_shift_y (int): Maximum pixel shift in the x-direction, applied progressively along the y-axis.
        x_shift_z (int): Maximum pixel shift in the x-direction, applied progressively along the z-axis.
        y_shift_x (int): Maximum pixel shift in the y-direction, applied progressively along the x-axis.
        y_shift_z (int): Maximum pixel shift in the y-direction, applied progressively along the z-axis.
        z_shift_x (int): Maximum pixel shift in the z-direction, applied progressively along the x-axis.
        z_shift_y (int): Maximum pixel shift in the z-direction, applied progressively along the y-axis.
        order (int): Order of interpolation. Order=0 (nearest-neighbor) keeps voxel values unchanged. Defaults to 1.

    Returns:
        sheared_volume (numpy.ndarray): The transformed volume.

    Raises:
        AssertionError: If the input shape is not 3D.
        AssertionError: If the order is not integer and in the range of 0-5.
        AssertionError: If the shift values are not integer.

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate box for shearing
        vol = np.zeros((60,100,100))
        vol[:, 20:80, 20:80] = 1

        qim3d.viz.slicer(vol, slice_axis=1)
        ```
        ![warp_box](../../assets/screenshots/warp_box_1.png)
        ```python
        # Shear the volume by 20% factor in x-direction along z-axis
        factor = 0.2
        shift = int(vol.shape[0]*factor)
        sheared_vol = qim3d.operations.shear3d(vol, x_shift_z=shift, order=1)

        qim3d.viz.slicer(sheared_vol, slice_axis=1)
        ```
        ![warp_box_shear](../../assets/screenshots/warp_box_shear.png)

    """
    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'
    assert isinstance(order, int), 'Order must be an integer.'
    assert 0 <= order <= 5, 'Order must be in the range 0-5.'
    assert all(
        isinstance(var, int)
        for var in (x_shift_y, x_shift_z, y_shift_x, y_shift_z, z_shift_x, z_shift_y)
    ), 'All shift values must be integers.'

    n, h, w = volume.shape

    # Create coordinate grid
    z, y, x = np.mgrid[0:n, 0:h, 0:w]

    # Generate linearly increasing shift maps
    x_shear_y = np.linspace(-x_shift_y, x_shift_y, h)  # X shift varies along Y
    x_shear_z = np.linspace(-x_shift_z, x_shift_z, n)  # X shift varies along Z

    y_shear_x = np.linspace(-y_shift_x, y_shift_x, w)  # Y shift varies along X
    y_shear_z = np.linspace(-y_shift_z, y_shift_z, n)  # Y shift varies along Z

    z_shear_x = np.linspace(-z_shift_x, z_shift_x, w)  # Z shift varies along X
    z_shear_y = np.linspace(-z_shift_y, z_shift_y, h)  # Z shift varies along Y

    # Apply pixelwise shifts
    x_new = x + x_shear_y[y] + x_shear_z[z]
    y_new = y + y_shear_x[x] + y_shear_z[z]
    z_new = z + z_shear_x[x] + z_shear_y[y]

    # Stack the new coordinates
    coords = np.array([z_new, y_new, x_new])

    # Apply transformation
    sheared_volume = scipy.ndimage.map_coordinates(
        volume, coords, order=order, mode='nearest'
    )

    return sheared_volume


def curve_warp(
    volume: np.ndarray,
    x_amp: float = 0,
    y_amp: float = 0,
    x_periods: float = 1.0,
    y_periods: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    order: int = 1,
) -> np.ndarray:
    """
    Applies an curve transformation along the z-axis using sine functions.

    Args:
        volume (numpy.ndarray): The input 3D volume (shape: n, h, w).
        x_amp (float): Determines the amplitude (height) of the curve in the x-direction. Defaults to 0.
        y_amp (float): Determines the amplitude (height) of the curve in the y-direction. Defautls to 0.
        x_periods (float): Determines the amount of periods (amount of wave crests) along the x-direction. Defaults to 1.0.
        y_periods (float): Determines the amount of periods (amount of wave crests) along the y-direction. Defaults to 1.0.
        x_offset (float): Determines pixelwise curve offset in x-direction. Defaults to 0.0.
        y_offset (float): Determines pixelwise curve offset in y-direction. Defaults to 0.0.
        order (int): Order of spline interpolation. Order=0 (nearest-neighbor) will keep voxel values unchanged. Defaults to 1.

    Returns:
        warped_volume (numpy.ndarray): The transformed volume.

    Raises:
        AssertionError: If the input shape is not 3D.
        AssertionError: If the order is not integer and in the range of 0-5.

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate box for warping
        vol = np.zeros((100,100,100))
        vol[:,40:60, 40:60] = 1
        qim3d.viz.slicer(vol, slice_axis=1)
        ```
        ![warp_box_long](../../assets/screenshots/warp_box_long.png)
        ```python
        # Warp the box along the x dimension
        warped_volume = qim3d.operations.curve_warp(vol, x_amp=10, x_periods=4)
        qim3d.viz.slicer(warped_volume, slice_axis=1)
        ```
        ![warp_box_curved](../../assets/screenshots/warp_box_curve.png)

    """
    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'
    assert isinstance(order, int), 'Order must be an integer.'
    assert 0 <= order <= 5, 'Order must be in the range 0-5.'

    n, h, w = volume.shape

    # Create a coordinate grid for the expanded volume
    z, y, x = np.mgrid[0:n, 0:h, 0:w]

    # Normalize z for smooth oscillations
    z_norm = z / (n - 1)  # Ranges from 0 to 1

    # Compute sinusoidal shifts
    x_amp = x_amp * np.sin(2 * np.pi * x_periods * z_norm + x_offset)
    x_new = x + x_amp

    y_amp = y_amp * np.sin(2 * np.pi * y_periods * z_norm + y_offset)
    y_new = y + y_amp

    # Stack the new coordinates for interpolation and interpolate
    coords = np.array([z, y_new, x_new])
    warped_volume = scipy.ndimage.map_coordinates(
        volume, coords, order=order, mode='nearest'
    )

    return warped_volume


def stretch(
    volume: np.ndarray,
    x_stretch: int = 0,
    y_stretch: int = 0,
    z_stretch: int = 0,
    order: int = 1,
) -> np.ndarray:
    """
    Stretches a volume by increasing the size of the volume in the input dimension with interpolation. The volume will therefore increase (or decrease if the stretch is negative) at the same rate as the volume, keeping its relative size.

    Args:
        volume (numpy.ndarray): The input 3D volume (shape: n, h, w).
        x_stretch (int): Amount of pixels to stretch the x-dimension. The operation is symmetric, and will be effective on both sides of the volume. Defaults to 0.
        y_stretch (int): Amount of pixels to stretch the x-dimension. The operation is symmetric, and will be effective on both sides of the volume. Defaults to 0.
        z_stretch (int): Amount of pixels to stretch the x-dimension. The operation is symmetric, and will be effective on both sides of the volume. Defaults to 0.
        order (int): Order of spline interpolation. Order=0 (nearest-neighbor) will keep voxel values unchanged. Defaults to 1.

    Returns:
        stretched_volume (numpy.ndarray): The stretched volume.

    Raises:
        AssertionError: If the input shape is not 3D.
        AssertionError: If the order is not integer and in the range of 0-5.
        AssertionError: If the stretching inputs are not integer.

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate box for stretching
        vol = np.zeros((100,100,100))
        vol[:,20:80, 20:80] = 1

        qim3d.viz.slicer(vol)
        ```
        ![warp_box](../../assets/screenshots/warp_box_0.png)

        ```python
        # Stretch the box along the x dimension
        stretched_volume = qim3d.operations.stretch(vol, x_stretch=20)
        print(stretched_volume.shape)
        qim3d.viz.slicer(stretched_volume)
        ```
        (100, 100, 140)

        ![warp_box_stretch](../../assets/screenshots/warp_box_stretch.png)
        ```python
        # Squeeze the box along the y dimension
        squeezed_volume = qim3d.operations.stretch(vol, x_stretch=-20)
        print(squeezed_volume.shape)
        qim3d.viz.slicer(squeezed_volume)
        ```
        (100, 100, 60)

        ![warp_box_squeeze](../../assets/screenshots/warp_box_squeeze.png)

    """
    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'
    assert isinstance(order, int), 'Order must be an integer.'
    assert 0 <= order <= 5, 'Order must be in the range 0-5.'
    assert all(
        isinstance(var, int) for var in (x_stretch, y_stretch, z_stretch)
    ), 'Amount of pixel stretching must be integer'

    n, h, w = volume.shape

    # New dimensions after stretching
    new_n = n + 2 * z_stretch
    new_h = h + 2 * y_stretch
    new_w = w + 2 * x_stretch

    # Generate coordinate grid for the original volume
    z_grid, y_grid, x_grid = np.meshgrid(
        np.linspace(0, n - 1, new_n),
        np.linspace(0, h - 1, new_h),
        np.linspace(0, w - 1, new_w),
        indexing='ij',
    )

    # Stack coordinates and reshape for map_coordinates
    coords = np.vstack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()])

    # Perform interpolation
    stretched_volume = scipy.ndimage.map_coordinates(
        volume, coords, order=order, mode='nearest'
    )

    # Reshape back to the new volume dimensions
    return stretched_volume.reshape((new_n, new_h, new_w))


def center_twist(
    volume: np.ndarray, rotation_angle: float = 90, axis: str = 'z', order: int = 1
) -> np.ndarray:
    """
    Applies a warping transformation that twists the volume around the center along the given axis.

    Args:
        volume (numpy.ndarray): The input 3D volume (shape: n, h, w).
        rotation_angle (float): Amount of rotation from bottom of rotation axis to top. Defaults to 90.
        axis (str): Axis for rotation. Should either take value 'x', 'y' or 'z'. Defaults to 'z'.
        order (int): Order of spline interpolation. Order=0 (nearest-neighbor) will keep voxel values unchanged. Defaults to 1.

    Returns:
        twisted_volume (numpy.ndarray): The center rotated volume.

    Raises:
        AssertionError: If the input shape is not 3D.
        AssertionError: If the order is not integer and in the range of 0-5.
        AssertionError: If the axis are not x, y or z

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate box for stretching
        vol = np.zeros((100,100,100))
        vol[:,20:80, 20:80] = 1
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/warp_box.html" width="100%" height="500" frameborder="0"></iframe>
        ```python
        # Twist the box 180 degrees along the z-axis
        twisted_volume = qim3d.operations.center_twist(vol, rotation_angle=180, axis='z', order=1)
        qim3d.viz.volumetric(twisted_volume)
        ```
        <iframe src="https://platform.qim.dk/k3d/warp_box_twist.html" width="100%" height="500" frameborder="0"></iframe>

    """
    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'
    assert isinstance(order, int), 'Order must be an integer.'
    assert 0 <= order <= 5, 'Order must be in the range 0-5.'
    assert axis in ['x', 'y', 'z'], 'Axis for rotation not recognized'

    # Get original dimensions
    n, h, w = volume.shape

    # Create a coordinate grid
    z, y, x = np.mgrid[0:n, 0:h, 0:w]

    if axis == 'z' or not axis:
        # Normalize
        z_norm = z / (n - 1)
        # Compute rotation angle per z-layer
        angles = np.radians(rotation_angle * z_norm)  # Convert to radians

        # Compute center and shift
        x_center, y_center = w / 2, h / 2
        x_shifted, y_shifted = x - x_center, y - y_center
        # Calculate new coordinates
        x_rot = x_center + x_shifted * np.cos(angles) - y_shifted * np.sin(angles)
        y_rot = y_center + x_shifted * np.sin(angles) + y_shifted * np.cos(angles)
        coords = np.array([z, y_rot, x_rot])
    elif axis == 'x':
        # Normalize
        x_norm = x / (w - 1)
        # Compute rotation angle per x-layer
        angles = np.radians(rotation_angle * x_norm)  # Convert to radians

        # Compute center and shift
        z_center, y_center = n / 2, h / 2
        z_shifted, y_shifted = z - z_center, y - y_center
        # Calculate new coordinates
        z_rot = z_center + z_shifted * np.cos(angles) - y_shifted * np.sin(angles)
        y_rot = y_center + z_shifted * np.sin(angles) + y_shifted * np.cos(angles)
        coords = np.array([z_rot, y_rot, x])
    elif axis == 'y':
        # Normalize
        y_norm = y / (h - 1)
        # Compute rotation angle per y-layer
        angles = np.radians(rotation_angle * y_norm)  # Convert to radians

        # Compute center and shift
        x_center, z_center = w / 2, n / 2
        x_shifted, z_shifted = x - x_center, z - z_center
        # Calculate new coordinates
        x_rot = x_center + z_shifted * np.sin(angles) + x_shifted * np.cos(angles)
        z_rot = z_center + z_shifted * np.cos(angles) - x_shifted * np.sin(angles)
        coords = np.array([z_rot, y, x_rot])

    # Interpolate at new coordinates
    swirled_volume = scipy.ndimage.map_coordinates(
        volume, coords, order=order, mode='nearest'
    )

    return swirled_volume
