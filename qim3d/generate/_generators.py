import ipywidgets as widgets
import k3d
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from IPython.display import display

import qim3d
from qim3d.utils import log, scale_to_float16
from qim3d.utils._dependencies import optional_import

# Import noise as optional dependency
noise = optional_import('noise', extra='synthetic-data')

pnoise3 = noise.pnoise3
snoise3 = noise.snoise3

__all__ = ['volume', 'background']


def background(
    background_shape: tuple,
    baseline_value: float = 0,
    min_noise_value: float = 0,
    max_noise_value: float = 20,
    generate_method: str = 'add',
    apply_method: str = None,
    seed: int = 0,
    dtype: str = 'uint8',
    apply_to: np.ndarray = None,
) -> np.ndarray:
    """
    Generates a 3D noise field or adds synthetic background noise to an existing volume.

    Unlike `volume` (which creates structures), this function generates **unstructured uniform noise**.
    It is useful for simulating:

    * **Sensor Noise:** Electronic noise or grain common in CT/microscopy scans.
    * **Imaging Artifacts:** Low-contrast background variations.
    * **Data Augmentation:** Making training data more robust by adding random interference.

    Args:
        background_shape (tuple):
            The shape of the noise volume to generate (Z, Y, X).
        baseline_value (float, optional):
            The constant base intensity level of the background.
        min_noise_value (float, optional):
            The lower bound of the random noise distribution.
        max_noise_value (float, optional):
            The upper bound of the random noise distribution.
        generate_method (str, optional):
            How to combine the baseline with the noise: `'add'`, `'subtract'`, `'multiply'`, or `'divide'`.
        apply_method (str, optional):
            If `apply_to` is provided, this defines how the noise is merged with the input volume.
        seed (int, optional):
            Random seed for reproducibility.
        dtype (str, optional):
            Output data type.
        apply_to (numpy.ndarray, optional):
            An existing 3D volume. If provided, the noise is applied directly to this array
            using `apply_method`.

    Returns:
        background (numpy.ndarray):
            The noise volume (if `apply_to` is None) or the modified input volume.

    Raises:
        ValueError: If `apply_method` is not one of 'add', 'subtract', 'multiply', or 'divide'.
        ValueError: If `apply_method` is provided without `apply_to` input volume provided, or vice versa.

    Example:
        ```python
        import qim3d

        # Generate noise volume
        background = qim3d.generate.background(
            background_shape = (128, 128, 128),
            baseline_value = 20,
            min_noise_value = 100,
            max_noise_value = 200,
        )

        qim3d.viz.volumetric(background)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_noise_background.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 15)

        # Apply noise to the synthetic collection
        noisy_collection = qim3d.generate.background(
            background_shape = volume_collection.shape,
            min_noise_value = 0,
            max_noise_value = 20,
            generate_method = 'add',
            apply_method = 'add',
            apply_to = volume_collection
        )

        qim3d.viz.volumetric(noisy_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_noisy_collection_1.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 15)

        # Apply noise to the synthetic collection
        noisy_collection = qim3d.generate.background(
            background_shape = volume_collection.shape,
            baseline_value = 0,
            min_noise_value = 0,
            max_noise_value = 30,
            generate_method = 'add',
            apply_method = 'divide',
            apply_to = volume_collection
        )

        qim3d.viz.volumetric(noisy_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_noisy_collection_2.html" width="100%" height="500" frameborder="0"></iframe>
        ```python
        qim3d.viz.slices_grid(noisy_collection, num_slices=10, color_bar=True, color_bar_style="large")
        ```
        ![synthetic_noisy_collection_slices](../../assets/screenshots/synthetic_noisy_collection_slices_2.png)

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 15)

        # Apply noise to the synthetic collection
        noisy_collection = qim3d.generate.background(
            background_shape = (200, 200, 200),
            baseline_value = 100,
            min_noise_value = 0.8,
            max_noise_value = 1.2,
            generate_method = "multiply",
            apply_method = "add",
            apply_to = volume_collection
        )

        qim3d.viz.slices_grid(noisy_collection, num_slices=10, color_bar=True, color_bar_style="large")
        ```
        ![synthetic_noisy_collection_slices](../../assets/screenshots/synthetic_noisy_collection_slices_3.png)

    """
    # Ensure dtype is a valid NumPy type
    dtype = np.dtype(dtype)

    # Define supported apply methods
    apply_operations = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / (b + 1e-8),  # Avoid division by zero
    }

    # Check if apply_method is provided without apply_to volume, or vice versa
    if (apply_to is None and apply_method is not None) or (
        apply_to is not None and apply_method is None
    ):
        msg = 'Supply both apply_method and apply_to when applying background to a volume.'
        # Validate apply_method
        raise ValueError(msg)

    # Validate apply_method
    if apply_method is not None and apply_method not in apply_operations:
        msg = f"Invalid apply_method '{apply_method}'. Choose from {list(apply_operations.keys())}."
        raise ValueError(msg)

    # Validate generate_method
    if generate_method not in apply_operations:
        msg = f"Invalid generate_method '{generate_method}'. Choose from {list(apply_operations.keys())}."
        raise ValueError(msg)

    # Check for shape mismatch
    if (apply_to is not None) and (apply_to.shape != background_shape):
        msg = f'Shape of input volume {apply_to.shape} does not match requested background_shape {background_shape}. Using input shape instead.'
        background_shape = apply_to.shape
        log.info(msg)

    # Generate the noise volume
    baseline = np.full(shape=background_shape, fill_value=baseline_value)

    # Start seeded generator
    rng = np.random.default_rng(seed=seed)
    noise = rng.uniform(
        low=float(min_noise_value), high=float(max_noise_value), size=background_shape
    )

    # Return error if multiplying or dividing with 0
    if baseline_value == 0.0 and (
        generate_method == 'multiply' or generate_method == 'divide'
    ):
        msg = f'Selection of baseline_value=0 and generate_method="{generate_method}" will not generate background noise. Either add baseline_value>0 or change generate_method.'
        raise ValueError(msg)

    # Apply method to initial background computation
    background_volume = apply_operations[generate_method](baseline, noise)

    # Warn user if the background noise is constant or none
    if np.min(background_volume) == np.max(background_volume):
        msg = 'Warning: The used settings have generated a background with a uniform value.'
        log.info(msg)

    # Apply method to the target volume if specified
    if apply_to is not None:
        background_volume = apply_operations[apply_method](apply_to, background_volume)

    # Clip value before dtype convertion
    clip_value = (
        np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.finfo(dtype).max
    )
    background_volume = np.clip(background_volume, 0, clip_value).astype(dtype)

    return background_volume


def volume(
    base_shape: tuple = (128, 128, 128),
    final_shape: tuple = None,
    noise_scale: float = 0.02,
    noise_type: str = 'perlin',
    decay_rate: float = 10,
    gamma: float = 1,
    threshold: float = 0.5,
    max_value: float = 255,
    shape: str = None,
    tube_hole_ratio: float = 0.5,
    axis: int = 0,
    order: int = 1,
    dtype: str = 'uint8',
    hollow: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    Generates a synthetic 3D volume using structured Perlin noise.

    Creates valid 3D morphological structures that resemble biological or material samples
    (e.g., cells, tissues, pores). By default, it generates a "blob-like" object, but it can
    also create specific geometric shapes like **cylinders** or **tubes**.

    This function is ideal for:

    * **Benchmarking:** Creating standard inputs for testing algorithms.
    * **Augmentation:** Generating synthetic samples to train deep learning models.
    * **Simulation:** Modeling physical structures with controlled noise properties.

    **Supported Shapes:**

    * **Blob:** (Default) amorphous, organic-looking structure.
    * **Cylinder:** A solid cylindrical rod.
    * **Tube:** A hollow cylinder.

    Args:
        base_shape (tuple, optional):
            The resolution of the internal noise grid. Higher values create finer details
            but require more computation.
        final_shape (tuple, optional):
            The final output resolution. If `None`, matches `base_shape`.
            Use this to upsample the generated volume.
        noise_scale (float, optional):
            Controls the "zoom" of the noise texture. Smaller values = smooth, large features.
            Larger values = rough, high-frequency details.
        noise_type (str, optional):
            The noise algorithm: `perlin` (standard) or `simplex` (faster, different artifacts).
        decay_rate (float, optional):
            Controls how quickly the object fades into the background at the edges.
            Higher values create sharper, distinct boundaries.
        gamma (float, optional):
            Adjusts contrast. `<1` flattens values (fuzzier), `>1` pushes values to extremes (binary-like).
        threshold (float, optional):
            The cut-off value (0.0 to 1.0) defining the object's surface.
            Lower values make the object larger/fatter; higher values make it smaller/thinner.
        max_value (float, optional):
            The maximum intensity value in the output array (e.g., 255 for 8-bit images).
        shape (str, optional):
            Forces the volume into a geometric shape: `None` (Blob), `cylinder`, or `tube`.
        tube_hole_ratio (float, optional):
            Only for `tube`. Defines the relative thickness of the wall.
            `0.1` is a thick wall, `0.9` is a thin shell.
        axis (int, optional):
            The orientation axis (0, 1, or 2) for cylinders and tubes.
        order (int, optional):
            Interpolation order when resizing to `final_shape` (0=Nearest, 1=Linear, etc.).
        dtype (str, optional):
            Output data type (e.g., `uint8`, `float32`).
        hollow (int, optional):
            If > 0, hollows out the blob by eroding the center, creating a shell of thickness `hollow`.
        seed (int, optional):
            Random seed for reproducibility.

    Returns:
        vol (numpy.ndarray):
            The generated 3D volume.

    Raises:
        ValueError: If `shape` or `noise_type` is invalid.
        TypeError: If either `base_shape` or `final_shape` is not a tuple or does not have three elements.
        TypeError: If `dtype` is not a valid numpy number type.
        ValueError: If `hollow` is not 0 or a positive integer.

    Example:
        Example:
        ```python
        import qim3d

        # Generate synthetic blob
        vol = qim3d.generate.volume(noise_scale = 0.02)

        # Visualize 3D volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(vol, value_min = 0, value_max = 255, num_slices = 15)
        ```
        ![synthetic_blob](../../assets/screenshots/synthetic_blob_slices.png)

    Example:
        ```python
        import qim3d

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(base_shape = (200, 100, 100),
                                    final_shape = (400,100,100),
                                    noise_scale = 0.03,
                                    threshold = 0.85,
                                    decay_rate=20,
                                    gamma=0.15,
                                    shape = "tube",
                                    tube_hole_ratio = 0.4,
                                    )

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_cylinder_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(vol, num_slices=15, slice_axis=1)
        ```
        ![synthetic_blob_cylinder_slice](../../assets/screenshots/synthetic_blob_cylinder_slice.png)

    Example:
        ```python
        import qim3d

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(base_shape = (200, 100, 100),
                                final_shape = (400, 100, 100),
                                noise_scale = 0.03,
                                gamma = 0.12,
                                threshold = 0.85,
                                shape = "tube",
                                )

        # Visualize synthetic blob
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_tube_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize
        qim3d.viz.slices_grid(vol, num_slices=15)
        ```
        ![synthetic_blob_tube_slice](../../assets/screenshots/synthetic_blob_tube_slice.png)

    """
    # Control
    shape_types = ['cylinder', 'tube']
    if shape and shape not in shape_types:
        err = f'shape should be one of: {shape_types}'
        raise ValueError(err)
    noise_types = ['pnoise', 'perlin', 'p', 'snoise', 'simplex', 's']
    if noise_type not in noise_types:
        err = f'noise_type should be one of: {noise_types}'
        raise ValueError(err)

    if not isinstance(base_shape, tuple) or len(base_shape) != 3:
        message = 'base_shape must be a tuple with three dimensions (z, y, x)'
        raise TypeError(message)

    if final_shape and (not isinstance(final_shape, tuple) or len(final_shape) != 3):
        message = 'final_shape must be a tuple with three dimensions (z, y, x)'
        raise TypeError(message)

    try:
        d = np.dtype(dtype)
    except TypeError as e:
        err = f'Datatype {dtype} is not a valid dtype.'
        raise TypeError(err) from e

    if hollow < 0 or isinstance(hollow, float):
        err = 'Argument "hollow" should be 0 or a positive integer'
        raise ValueError(err)

    # Generate grid of coordinates
    z, y, x = np.indices(base_shape)

    # Generate noise
    if (
        np.round(noise_scale, 3) == 0
    ):  # Only detect three decimal position (0.001 is ok, but 0.0001 is 0)
        noise_scale = 0

    if noise_scale == 0:
        noise = np.ones(base_shape)
    else:
        if noise_type in noise_types[:3]:
            vectorized_noise = np.vectorize(pnoise3)
            noise = vectorized_noise(
                z.flatten() * noise_scale,
                y.flatten() * noise_scale,
                x.flatten() * noise_scale,
                base=seed,
            ).reshape(base_shape)
        elif noise_type in noise_types[3:]:
            vectorized_noise = np.vectorize(snoise3)
            noise = vectorized_noise(
                z.flatten() * noise_scale,
                y.flatten() * noise_scale,
                x.flatten() * noise_scale,
            ).reshape(base_shape)
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    # Calculate the center of the array
    center = np.array([(s - 1) / 2 for s in base_shape])

    # Calculate the distance of each point from the center
    if not shape:
        distance = np.linalg.norm(
            [
                (z - center[0]) / center[0],
                (y - center[1]) / center[1],
                (x - center[2]) / center[2],
            ],
            axis=0,
        )
        max_distance = np.sqrt(3)
        # Set ratio
        miin = np.max(
            [
                distance[distance.shape[0] // 2, distance.shape[1] // 2, 0],
                distance[distance.shape[0] // 2, 0, distance.shape[2] // 2],
                distance[0, distance.shape[1] // 2, distance.shape[2] // 2],
            ]
        )
        ratio = miin / max_distance  # 0.577

    elif shape == 'cylinder' or shape == 'tube':
        distance_list = np.array(
            [
                (z - center[0]) / center[0],
                (y - center[1]) / center[1],
                (x - center[2]) / center[2],
            ]
        )
        # remove the axis along which the fading is not applied
        distance_list = np.delete(distance_list, axis, axis=0)
        distance = np.linalg.norm(distance_list, axis=0)
        max_distance = np.sqrt(2)
        # Set ratio
        miin = np.max(
            [
                distance[distance.shape[0] // 2, distance.shape[1] // 2, 0],
                distance[distance.shape[0] // 2, 0, distance.shape[2] // 2],
                distance[0, distance.shape[1] // 2, distance.shape[2] // 2],
            ]
        )
        ratio = miin / max_distance  # 0.707

    # Scale the distance such that the shortest distance (from center to any edge) is 1 (prevents clipping)
    scaled_distance = distance / (max_distance * ratio)

    # Apply decay rate
    faded_distance = np.power(scaled_distance, decay_rate)

    # Invert the distances to have 1 at the center and 0 at the edges
    fade_array = 1 - faded_distance
    fade_array[fade_array <= 0] = 0

    # Apply the fading to the volume
    vol_faded = noise * fade_array

    # Normalize the volume
    vol_normalized = vol_faded / np.max(vol_faded)

    # Apply gamma
    generated_vol = np.power(vol_normalized, gamma)

    # Scale to max_value
    generated_vol = generated_vol * max_value

    # Threshold
    generated_vol[generated_vol < threshold * max_value] = 0

    # Apply fade mask for creation of tube
    if shape == 'tube':
        generated_vol = qim3d.operations.fade_mask(
            generated_vol,
            geometry='cylindrical',
            axis=axis,
            ratio=tube_hole_ratio,
            decay_rate=5,
            invert=True,
        )

    # Scale up the volume of volume to size
    if final_shape:
        generated_vol = scipy.ndimage.zoom(
            generated_vol, np.array(final_shape) / np.array(base_shape), order=order
        )

    generated_vol = generated_vol.astype(dtype)

    if hollow > 0:
        generated_vol = qim3d.operations.make_hollow(generated_vol, hollow)

    return generated_vol


class ParameterVisualizer:
    """
    Interactive Jupyter widget for exploring synthetic data generation parameters.

    Provides a Graphical User Interface (GUI) to tune the parameters of `qim3d.generate.volume`
    in real-time. Users can adjust sliders for noise scale, threshold, and gamma while immediately
    seeing the resulting 3D structure.

    Args:
        base_shape (tuple, optional): Determines the shape of the generate volume. This will not be updated when exploring parameters and must be determined when generating the visualizer.
        final_shape (tuple, optional): Desired shape of the final volume. If unspecified, will assume same shape as base_shape.
        seed (int, optional): Determines the seed for the volume generation. Enables the user to generate different volumes with the same parameters.
        hollow (int, optional): Determines thickness of the hollowing operation. Volume is only hollowed if hollow>0.
        initial_config (dict, optional): Dictionary that defines the starting parameters of the visualizer. Can be used if a specific setup is needed. The dictionary may contain the keywords: `noise_type`, `noise_scale`, `decay_rate`, `gamma`, `threshold`, `shape` and `tube_hole_ratio`.
        nsmin (float, optional): Determines minimum value for the noise scale slider.
        nsmax (float, optional): Determines maximum value for the noise scale slider.
        dsmin (float, optional): Determines minimum value for the decay rate slider.
        dsmax (float, optional): Determines maximum value for the decay rate slider.
        gsmin (float, optional): Determines minimum value for the gamma slider.
        gsmax (float, optional): Determines maximum value for the gamma slider.
        tsmin (float, optional): Determines minimum value for the threshold slider.
        tsmax (float, optional): Determines maximum value for the threshold slider.
        grid_visible (bool, optional): Determines if the grid should be visible upon plot generation.

    Raises:
        ValueError: If either `base_shape`, `noise slider`, `decay slider`, `gamma slider`, or `threshold slider` are invalid.

    Example:
        ```python
        import qim3d

        viz = qim3d.generate.ParameterVisualizer()
        ```
        ![paramter_visualizer](../../assets/screenshots/viz-synthetic_parameters.gif)

    Accessing the current volume:
            The most recently generated 3D volume can be retrieved at any time using the `.get_volume()` method:

            ```python
            vol = viz.get_volume()
            ```
            This returns the synthetic volume as a NumPy ndarray corresponding to the current widget parameters.

    """

    def __init__(
        self,
        base_shape: tuple = (128, 128, 128),
        final_shape: tuple = None,
        seed: int = 0,
        hollow: int = 0,
        initial_config: dict = None,
        nsmin: float = 0.0,
        nsmax: float = 0.1,
        dsmin: float = 0.1,
        dsmax: float = 20.0,
        gsmin: float = 0.1,
        gsmax: float = 2.0,
        tsmin: float = 0.0,
        tsmax: float = 1.0,
        grid_visible: bool = False,
    ):
        # Error checking:
        if not isinstance(base_shape, tuple) or len(base_shape) != 3:
            err = 'base_shape should be a tuple of three sizes.'
            raise ValueError(err)

        if final_shape is not None:
            if not isinstance(final_shape, tuple) or len(final_shape) != 3:
                err = 'final_shape should be a tuple of three sizes or None.'
                raise ValueError(err)

        if hollow < 0 or isinstance(hollow, float):
            err = 'Argument "hollow" should be 0 or a positive integer'
            raise ValueError(err)

        if nsmin > nsmax:
            err = f'Minimum slider value for noise must be less than or equal to the maximum. Given: min = {nsmin}, max = {nsmax}.'
            raise ValueError(err)

        if dsmin > dsmax:
            err = f'Minimum decay rate value must be less than or equal to the maximum. Given: min = {dsmin}, max = {dsmax}.'
            raise ValueError(err)

        if gsmin > gsmax:
            err = f'Minimum gamma value must be less than or equal to the maximum. Given: min = {gsmin}, max = {gsmax}.'
            raise ValueError(err)

        if tsmin > tsmax:
            err = f'Minimum threshold value must be less than or equal to the maximum. Given: min = {tsmin}, max = {tsmax}.'
            raise ValueError(err)

        self.base_shape = base_shape
        self.final_shape = final_shape
        self.hollow = hollow
        self.seed = int(seed)
        self.axis = 0  # Not customizable
        self.max_value = 255  # Not customizable

        # Min and max values for sliders
        self.nsmin = nsmin
        self.nsmax = nsmax
        self.dsmin = dsmin
        self.dsmax = dsmax
        self.gsmin = gsmin
        self.gsmax = gsmax
        self.tsmin = tsmin
        self.tsmax = tsmax

        self.grid_visible = grid_visible
        self.config = {
            'noise_scale': 0.02,
            'decay_rate': 10,
            'gamma': 1.0,
            'threshold': 0.5,
            'tube_hole_ratio': 0.5,
            'shape': None,
            'noise_type': 'perlin',
        }
        if initial_config:
            self.config.update(initial_config)

        self.state = {}
        self._build_widgets()
        self._setup_plot()
        self._display_ui()

    def _compute_volume(self) -> None:
        vol = volume(
            base_shape=self.base_shape,
            final_shape=self.final_shape,
            noise_type=self.config['noise_type'],
            noise_scale=self.config['noise_scale'],
            decay_rate=self.config['decay_rate'],
            gamma=self.config['gamma'],
            threshold=self.config['threshold'],
            shape=self.config['shape'],
            tube_hole_ratio=self.config['tube_hole_ratio'],
            seed=self.seed,
            hollow=self.hollow,
        )
        return scale_to_float16(vol)

    def _build_widgets(self) -> None:
        # Widgets
        self.noise_slider = widgets.FloatSlider(
            value=self.config['noise_scale'],
            min=self.nsmin,
            max=self.nsmax,
            step=0.001,
            description='Noise',
            readout_format='.3f',
            continuous_update=False,
        )
        self.decay_slider = widgets.FloatSlider(
            value=self.config['decay_rate'],
            min=self.dsmin,
            max=self.dsmax,
            step=0.1,
            description='Decay',
            continuous_update=False,
        )
        self.gamma_slider = widgets.FloatSlider(
            value=self.config['gamma'],
            min=self.gsmin,
            max=self.gsmax,
            step=0.1,
            description='Gamma',
            continuous_update=False,
        )
        self.threshold_slider = widgets.FloatSlider(
            value=self.config['threshold'],
            min=self.tsmin,
            max=self.tsmax,
            step=0.05,
            description='Threshold',
            continuous_update=False,
        )
        self.noise_type_dropdown = widgets.Dropdown(
            options=['perlin', 'simplex'], value='perlin', description='Noise Type'
        )
        self.shape_dropdown = widgets.Dropdown(
            options=[None, 'cylinder', 'tube'], value=None, description='Shape'
        )
        self.tube_hole_ratio_slider = widgets.FloatSlider(
            value=self.config['tube_hole_ratio'],
            min=0.0,
            max=1.0,
            step=0.05,
            description='Tube hole ratio',
            style={'description_width': 'initial'},
            continuous_update=False,
        )
        self.base_shape_x_text = widgets.IntText(
            value=self.base_shape[0],
        )
        self.base_shape_y_text = widgets.IntText(
            value=self.base_shape[1],
        )
        self.base_shape_z_text = widgets.IntText(
            value=self.base_shape[2],
        )
        self.final_same_as_base_checkbox = widgets.Checkbox(
            value=True, description='Same as base_shape'
        )
        self.final_shape_x_text = widgets.IntText(
            value=self.base_shape[0],
        )
        self.final_shape_y_text = widgets.IntText(
            value=self.base_shape[1],
        )
        self.final_shape_z_text = widgets.IntText(
            value=self.base_shape[2],
        )
        self.hollow_text = widgets.BoundedIntText(
            value=self.hollow,
            min=0,
            max=1000,  # chosen arbitrarily atm.
            step=1,
            description='Hollow',
        )
        self.colormap_dropdown = widgets.Dropdown(
            options=['magma', 'viridis', 'gray', 'plasma'],
            value='magma',
            description='Colormap',
        )
        self.grid_checkbox = widgets.Checkbox(
            value=self.grid_visible, description='Show grid'
        )

        # Observers
        self.noise_slider.observe(self._on_change, names='value')
        self.noise_type_dropdown.observe(self._on_change, names='value')
        self.decay_slider.observe(self._on_change, names='value')
        self.gamma_slider.observe(self._on_change, names='value')
        self.threshold_slider.observe(self._on_change, names='value')
        self.shape_dropdown.observe(self._on_change, names='value')
        self.tube_hole_ratio_slider.observe(self._on_change, names='value')
        self.base_shape_x_text.observe(self._on_change, names='value')
        self.base_shape_y_text.observe(self._on_change, names='value')
        self.base_shape_z_text.observe(self._on_change, names='value')
        self.final_shape_x_text.observe(self._on_change, names='value')
        self.final_shape_y_text.observe(self._on_change, names='value')
        self.final_shape_z_text.observe(self._on_change, names='value')
        self.hollow_text.observe(self._on_change, names='value')
        self.colormap_dropdown.observe(self._on_change, names='value')
        self.grid_checkbox.observe(self._on_change, names='value')
        self.final_same_as_base_checkbox.observe(
            self._on_checkbox_change, names='value'
        )
        self.final_same_as_base_checkbox.observe(self._on_change, names='value')
        # Initial state
        self._on_checkbox_change({'new': self.final_same_as_base_checkbox.value})

    def _on_checkbox_change(self, change) -> None:
        disabled = change['new']
        self.final_shape_x_text.disabled = disabled
        self.final_shape_y_text.disabled = disabled
        self.final_shape_z_text.disabled = disabled

    def _get_base_shape(self) -> tuple:
        # Check valid axes
        for axis in [
            self.base_shape_x_text,
            self.base_shape_y_text,
            self.base_shape_z_text,
        ]:
            if axis.value < 1:
                axis.value = 1
        return (
            self.base_shape_x_text.value,
            self.base_shape_y_text.value,
            self.base_shape_z_text.value,
        )

    def _get_final_shape(self) -> tuple:
        if self.final_same_as_base_checkbox.value:
            return None
        else:
            return (
                self.final_shape_x_text.value,
                self.final_shape_y_text.value,
                self.final_shape_z_text.value,
            )

    def _setup_plot(self) -> None:
        vol = self._compute_volume()

        cmap = plt.get_cmap(self.colormap_dropdown.value)
        attr_vals = np.linspace(0.0, 1.0, num=cmap.N)
        rgb_vals = cmap(np.arange(0, cmap.N))[:, :3]
        color_map = np.column_stack((attr_vals, rgb_vals)).tolist()

        pixel_count = np.prod(vol.shape)
        y1, x1 = 256, 16777216  # 256 samples at res 256*256*256=16.777.216
        y2, x2 = 32, 134217728  # 32 samples at res 512*512*512=134.217.728
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1
        samples = int(min(max(a * pixel_count + b, 64), 512))

        self.plot = k3d.plot(grid_visible=self.grid_visible)
        self.plt_volume = k3d.volume(
            vol,
            bounds=[0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2]],
            color_map=color_map,
            samples=samples,
            color_range=[np.min(vol), np.max(vol)],
            opacity_function=[],
            interpolation=True,
        )
        self.plot += self.plt_volume

    def _on_change(self, change: None = None) -> None:
        self.config['noise_type'] = self.noise_type_dropdown.value
        self.config['noise_scale'] = self.noise_slider.value
        self.config['decay_rate'] = self.decay_slider.value
        self.config['gamma'] = self.gamma_slider.value
        self.config['threshold'] = self.threshold_slider.value
        self.config['shape'] = self.shape_dropdown.value
        self.config['tube_hole_ratio'] = self.tube_hole_ratio_slider.value
        self.base_shape = self._get_base_shape()
        self.final_shape = self._get_final_shape()
        self.hollow = self.hollow_text.value

        # Update colormap
        cmap = plt.get_cmap(self.colormap_dropdown.value)
        attr_vals = np.linspace(0.0, 1.0, num=cmap.N)
        rgb_vals = cmap(np.arange(0, cmap.N))[:, :3]
        color_map = np.column_stack((attr_vals, rgb_vals)).tolist()
        self.plt_volume.color_map = color_map

        # Update grid
        self.plot.grid_visible = self.grid_checkbox.value

        # Recompute volume
        new_vol = self._compute_volume()

        # Recompute samples based on the new shape (same logic as in _setup_plot)
        pixel_count = int(np.prod(new_vol.shape))
        y1, x1 = 256, 16777216  # 256 samples at 256^3
        y2, x2 = 32, 134217728  # 32 samples  at 512^3
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1
        samples = int(min(max(a * pixel_count + b, 64), 512))

        # If the shape changed, rebuild the K3D volume actor (needed for K3D)
        if new_vol.shape != self.plt_volume.volume.shape:
            # Remove old actor
            self.plot -= self.plt_volume

            # Build fresh actor with updated bounds/color_range/samples
            self.plt_volume = k3d.volume(
                new_vol,
                bounds=[0, new_vol.shape[0], 0, new_vol.shape[1], 0, new_vol.shape[2]],
                color_map=color_map,
                samples=samples,
                color_range=[float(np.min(new_vol)), float(np.max(new_vol))],
                opacity_function=[],
                interpolation=True,
            )
            self.plot += self.plt_volume
        else:
            # Same shape: just update data AND color_range
            self.plt_volume.volume = new_vol
            self.plt_volume.color_range = [
                float(np.min(new_vol)),
                float(np.max(new_vol)),
            ]

    def _display_ui(self) -> None:
        small_box = widgets.Layout(width='65px')
        for box in [
            self.base_shape_x_text,
            self.base_shape_y_text,
            self.base_shape_z_text,
            self.final_shape_x_text,
            self.final_shape_y_text,
            self.final_shape_z_text,
        ]:
            box.layout = small_box

        self.base_shape_box = widgets.HBox(
            [
                widgets.Label('Base shape  '),
                widgets.Label('x'),
                self.base_shape_x_text,
                widgets.Label('y'),
                self.base_shape_y_text,
                widgets.Label('z'),
                self.base_shape_z_text,
            ]
        )
        self.final_shape_box = widgets.HBox(
            [
                widgets.Label('Final shape  '),
                widgets.Label('x'),
                self.final_shape_x_text,
                widgets.Label('y'),
                self.final_shape_y_text,
                widgets.Label('z'),
                self.final_shape_z_text,
            ]
        )

        parameters_controls = widgets.VBox(
            [
                self.base_shape_box,
                self.final_shape_box,
                self.final_same_as_base_checkbox,
                self.hollow_text,
                self.noise_type_dropdown,
                self.noise_slider,
                self.decay_slider,
                self.gamma_slider,
                self.threshold_slider,
                self.shape_dropdown,
                self.tube_hole_ratio_slider,
            ]
        )

        # Controls styling
        parameters_controls.layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            flex='0 1',
            min_width='350px',  # Ensure it doesn't get too small
            height='auto',
            overflow_y='auto',
            border='1px solid lightgray',
            padding='10px',
            margin='0 1em 0 0',
        )

        visualization_controls = widgets.VBox(
            [self.colormap_dropdown, self.grid_checkbox]
        )

        visualization_controls.layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            flex='0 1',
            min_width='350px',  # Ensure it doesn't get too small
            height='auto',
            overflow_y='auto',
            border='1px solid lightgray',
            padding='10px',
            margin='0 1em 0 0',
        )

        tabs = widgets.Tab(children=[parameters_controls, visualization_controls])
        tabs.set_title(0, 'Parameters')
        tabs.set_title(1, 'Visualization')

        plot_output = widgets.Output()
        plot_output.layout = widgets.Layout(
            flex='1 1 auto',
            height='auto',
            border='1px solid lightgray',
            overflow='auto',
            min_width='500px',
        )
        with plot_output:
            display(self.plot)

        ui = widgets.HBox(
            [tabs, plot_output],
            layout=widgets.Layout(
                width='100%', display='flex', flex_flow='row', align_items='stretch'
            ),
        )

        display(ui)

    def get_volume(self):
        """
        Extracts the generated volume from the widget's current state.

        Allows you to retrieve the numpy array resulting from your interactive adjustments
        so you can use it in your pipeline (e.g., saving it or using it for training).

        Returns:
            vol (numpy.ndarray): The 3D volume currently visualized in the widget.

        Example:
            ```python
            # After adjusting sliders in the widget:
            my_custom_blob = viz.get_volume()
            qim3d.io.save("custom_blob.tif", my_custom_blob)
            ```

        """
        return self.plt_volume.volume
