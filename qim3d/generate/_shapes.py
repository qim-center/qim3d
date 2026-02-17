import numpy as np

from qim3d.filters import gaussian

# Import qim3d modules
from qim3d.generate._generators import volume
from qim3d.operations import center_twist
from qim3d.utils import log

__all__ = ['berry', 'rope']


# =============================================================================
# Functions for Berry Generation
# =============================================================================


def _generate_berry_positions(
    core_radius: int,
    drupelet_radius: int,
    num_drupelets: int,
    drupelet_radius_jitter: int = 0,
    position_jitter: int = 0,
    seed: int = 0,
    offset: tuple = (0, 0, 0),
    top_opening_threshold: float = 0.8,
    rim_thickness: float = 0.1,
) -> tuple[list[tuple], list[int]]:
    """Generate 3D positions for berry drupelets using spherical distribution."""
    # Berry rim shaping constants
    rim_radial_compression = (
        0.15  # Maximum radial compression at rim (pulls drupelets inward)
    )
    rim_base_scale = 0.9  # Base drupelet size at rim start (90% of normal)
    rim_scale_reduction = 0.1  # Additional size reduction at rim edge
    transition_height = 0.6  # Height where transition to rim begins (60% up sphere)
    transition_radial_factor = 0.05  # Gentle radial adjustment in transition
    transition_scale_factor = 0.1  # Size reduction in transition zone

    # Fibonacci sphere distribution constants
    sphere_y_range = 2.0  # Sphere coordinate range [-1, 1]
    extra_candidate_ratio = 0.75  # Buffer for rejected drupelets

    rng = np.random.default_rng(seed)
    positions, drupelet_radii = [], []
    offset = np.array(offset, dtype=int)

    def process_drupelet(x: float, y: float, z: float, i: int) -> None:
        """Helper function to process a single drupelet."""
        rim_start = top_opening_threshold - rim_thickness
        top_factor = drupelet_scale = 1.0

        if y > rim_start:
            rim_progress = (y - rim_start) / rim_thickness
            top_factor = 1.0 - rim_radial_compression * rim_progress
            drupelet_scale = rim_base_scale - rim_scale_reduction * rim_progress
        elif y > transition_height:
            transition_factor = (y - transition_height) / (
                rim_start - transition_height
            )
            top_factor = 1.0 - transition_radial_factor * transition_factor
            drupelet_scale = 1.0 - transition_scale_factor * transition_factor

        this_drupelet_radius = max(
            1,
            int(drupelet_radius * drupelet_scale)
            + rng.integers(-drupelet_radius_jitter, drupelet_radius_jitter + 1),
        )
        drupelet_radii.append(this_drupelet_radius)

        direction = np.array([x, y, z]) / (np.linalg.norm([x, y, z]) + 1e-16)
        base_pos = (core_radius + this_drupelet_radius) * direction * top_factor

        rand_vec = rng.normal(size=3)
        rand_vec -= rand_vec.dot(direction) * direction
        rand_vec /= np.linalg.norm(rand_vec) + 1e-8
        offset_vec = rand_vec * rng.integers(-position_jitter, position_jitter + 1)

        pos = np.round(base_pos + offset_vec).astype(int) + offset
        positions.append(tuple(pos))

    extra_drupelets = int(
        num_drupelets * (1 - top_opening_threshold) * extra_candidate_ratio
    )
    total_candidates = num_drupelets + extra_drupelets
    offset_val = sphere_y_range / total_candidates
    increment = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle ≈ 137.5°

    for i in range(total_candidates):
        if len(positions) >= num_drupelets:
            break
        y = ((i * offset_val) - 1) + (offset_val / 2)
        if y <= top_opening_threshold:
            r = np.sqrt(1 - y * y)
            phi = i * increment
            process_drupelet(np.cos(phi) * r, y, np.sin(phi) * r, i)

    while len(positions) < num_drupelets:
        y = rng.uniform(-1, top_opening_threshold)
        r = np.sqrt(1 - y * y)
        phi = rng.uniform(0, 2 * np.pi)
        process_drupelet(np.cos(phi) * r, y, np.sin(phi) * r, len(positions))

    return positions, drupelet_radii


def _overlapping_placement(
    collection: np.ndarray,
    blob: np.ndarray,
    position: tuple,
    labels_array: np.ndarray = None,
    label_id: int = 0,
) -> tuple[np.ndarray, bool]:
    """Place blob at position with overlap handling."""
    z, y, x = position
    start = np.array([z, y, x]) - np.array(blob.shape) // 2
    end = start + np.array(blob.shape)

    within_bounds = np.all(start >= 0) and np.all(end <= np.array(collection.shape))

    if within_bounds:
        collection_slice = collection[
            start[0] : end[0], start[1] : end[1], start[2] : end[2]
        ]
        overlap_mask = (collection_slice > 0) & (blob > 0)

        if labels_array is not None:
            label_slice = labels_array[
                start[0] : end[0], start[1] : end[1], start[2] : end[2]
            ]
            new_blob = np.where(overlap_mask & (blob >= collection_slice), blob, 0)
        else:
            new_blob = np.where(overlap_mask, blob * 0.5 + collection_slice * 0.5, blob)

        no_conflict = ~overlap_mask
        final_blob = np.where(no_conflict, blob, new_blob)

        collection[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = (
            np.maximum(collection_slice, final_blob)
        )

        if labels_array is not None and label_id > 0:
            blob_mask = final_blob > 0
            new_labels = np.where((label_slice == 0) & blob_mask, label_id, label_slice)
            contested_areas = overlap_mask & (blob >= collection_slice)
            new_labels = np.where(contested_areas, label_id, new_labels)
            labels_array[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = (
                new_labels
            )

        return collection, True

    return collection, False


def berry(
    shape: tuple = (200, 200, 200),
    core_radius: int = 20,
    drupelet_radius: int = 15,
    num_drupelets: int = 60,
    noise_scale: float = 0.02,
    decay_rate: float = 9.9,
    gamma: float = 0.6,
    threshold: float = 0.5,
    max_value: float = 240,
    top_opening_threshold: float = 0.85,
    rim_thickness: float = 0.1,
    drupelet_radius_jitter: int = 1,
    position_jitter: int = 2,
    seed: int = 0,
    dtype: str = 'uint8',
    return_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a berry-like 3D structure with bumpy surface.

    Creates a spherical volume composed of individual drupelets (small spherical
    units) arranged in a natural raspberry-like pattern with a hollow opening at
    the top.

    Args:
        shape: Dimensions of output volume (z, y, x). Defaults to (200, 200, 200).
        core_radius: Radius of the berry core in voxels. Defaults to 20.
        drupelet_radius: Base radius for individual drupelets. Defaults to 15.
        num_drupelets: Number of drupelets to generate. Defaults to 60.
        noise_scale: Perlin noise scale for drupelet texture. Defaults to 0.02.
        decay_rate: Edge sharpness of drupelets. Defaults to 9.9.
        gamma: Gamma correction for drupelet intensity. Defaults to 0.6.
        threshold: Threshold for clipping low values (0-1). Defaults to 0.5.
        max_value: Maximum intensity value. Defaults to 240.
        top_opening_threshold: Height cutoff for top opening (0-1). Defaults to 0.85.
        rim_thickness: Thickness of the rim area (0-1). Defaults to 0.1.
        drupelet_radius_jitter: Random variation in drupelet size. Defaults to 1.
        position_jitter: Random position offset in voxels. Defaults to 2.
        seed: Random seed for reproducibility. Defaults to 0.
        dtype: Output data type. Defaults to 'uint8'.
        return_labels: If True, also return label array for each drupelet. Defaults to False.

    Returns:
        Berry volume as NumPy array, or tuple of (berry_volume, labels) if return_labels=True.

    Examples:
        >>> import qim3d
        >>> berry_vol = qim3d.generate.berry(num_drupelets=80, seed=42)
        >>> berry_vol, labels = qim3d.generate.berry(return_labels=True)

    """
    # Validate inputs
    if num_drupelets < 1:
        msg = 'num_drupelets must be at least 1'
        raise ValueError(msg)
    if core_radius <= 0:
        msg = 'core_radius must be positive'
        raise ValueError(msg)
    if drupelet_radius <= 0:
        msg = 'drupelet_radius must be positive'
        raise ValueError(msg)
    if not all(s > 0 for s in shape):
        msg = 'All shape dimensions must be positive'
        raise ValueError(msg)
    if not 0 <= threshold <= 1:
        msg = 'threshold must be between 0 and 1'
        raise ValueError(msg)
    if not 0 <= top_opening_threshold <= 1:
        msg = 'top_opening_threshold must be between 0 and 1'
        raise ValueError(msg)
    if not 0 < rim_thickness <= 1:
        msg = 'rim_thickness must be between 0 and 1 (exclusive of 0)'
        raise ValueError(msg)
    if gamma <= 0:
        msg = 'gamma must be positive'
        raise ValueError(msg)

    log.info(f'Generating berry with {num_drupelets} drupelets in shape {shape}')

    center = tuple(s // 2 for s in shape)

    positions, drupelet_radii = _generate_berry_positions(
        core_radius=core_radius,
        drupelet_radius=drupelet_radius,
        num_drupelets=num_drupelets,
        drupelet_radius_jitter=drupelet_radius_jitter,
        position_jitter=position_jitter,
        seed=seed,
        offset=center,
        top_opening_threshold=top_opening_threshold,
        rim_thickness=rim_thickness,
    )

    berry_volume = np.zeros(shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint8) if return_labels else None

    rng = np.random.default_rng(seed)

    for i, pos in enumerate(positions):
        drupelet_size = max(6, drupelet_radii[i] * 2)
        drupelet_shape = (drupelet_size, drupelet_size, drupelet_size)

        drupelet = volume(
            base_shape=drupelet_shape,
            noise_scale=noise_scale,
            decay_rate=decay_rate,
            gamma=gamma + rng.uniform(-0.1, 0.1),
            threshold=threshold + rng.uniform(-0.05, 0.05),
            max_value=int(max_value + rng.integers(-20, 21)),
            seed=seed + i,
        )

        berry_volume, placed = _overlapping_placement(
            berry_volume, drupelet, pos, labels, i + 1
        )

    berry_volume = berry_volume.astype(dtype)

    if return_labels:
        return berry_volume, labels
    return berry_volume


# =============================================================================
# Functions for Rope Generation
# =============================================================================


def _generate_twisted_thread(
    length: int,
    thickness: int,
    twist_rate: float,
    phase_offset: float,
    noise_scale: float,
    seed: int,
) -> np.ndarray:
    """Generate a continuous thread with integrated twist."""
    thread = volume(
        base_shape=(length, thickness, thickness),
        final_shape=(length, thickness, thickness),
        noise_scale=noise_scale,
        gamma=0.1,
        threshold=0.6,
        max_value=240,
        shape='cylinder',
        axis=0,
        seed=seed,
    )

    total_rotation = twist_rate * 360 + phase_offset
    return center_twist(thread, rotation_angle=total_rotation, axis='z', order=1)


def _integrate_thread(
    combined_rope: np.ndarray,
    label_volume: np.ndarray | None,
    thread: np.ndarray,
    rope_center_y: int,
    rope_center_x: int,
    radius: float,
    base_angle: float,
    twist_rate: float,
    thread_id: int,
    compression_factor: float,
) -> None:
    """Integrate thread into rope volume with cohesive blending."""
    # Thread waviness constants (create natural fiber irregularity)
    waviness_amplitude = 2  # Waviness amplitude in voxels
    waviness_freq_y = 0.05  # Y-axis oscillation frequency
    waviness_freq_x = 0.07  # X-axis oscillation frequency (different to avoid patterns)
    phase_multiplier = 1.3  # Phase shift between axes

    # Thread compression constants
    compression_boost = 40  # Intensity boost at compressed boundaries (out of 255)
    secondary_compression = 0.5  # Compression factor for secondary thread

    length, thread_thickness = thread.shape[0], thread.shape[1]
    half_thickness = thread_thickness // 2
    rope_shape = combined_rope.shape

    for z in range(length):
        z_progress = z / length
        rope_twist = twist_rate * 0.5 * 2 * np.pi * z_progress
        current_angle = base_angle + rope_twist

        center_y = (
            rope_center_y
            + int(radius * np.cos(current_angle))
            + int(waviness_amplitude * np.sin(z * waviness_freq_y + base_angle))
        )
        center_x = (
            rope_center_x
            + int(radius * np.sin(current_angle))
            + int(
                waviness_amplitude
                * np.cos(z * waviness_freq_x + base_angle * phase_multiplier)
            )
        )

        y_start, y_end = (
            max(0, center_y - half_thickness),
            min(rope_shape[1], center_y + half_thickness),
        )
        x_start, x_end = (
            max(0, center_x - half_thickness),
            min(rope_shape[2], center_x + half_thickness),
        )

        thread_y_start = max(0, half_thickness - (center_y - y_start))
        thread_y_end = min(thread_thickness, thread_y_start + (y_end - y_start))
        thread_x_start = max(0, half_thickness - (center_x - x_start))
        thread_x_end = min(thread_thickness, thread_x_start + (x_end - x_start))

        if not (
            y_end > y_start
            and x_end > x_start
            and thread_y_end > thread_y_start
            and thread_x_end > thread_x_start
        ):
            continue

        thread_section = thread[
            z, thread_y_start:thread_y_end, thread_x_start:thread_x_end
        ]
        thread_mask = thread_section > 0

        if not np.any(thread_mask):
            continue

        existing_rope = combined_rope[z, y_start:y_end, x_start:x_end]

        overlap_mask = (existing_rope > 0) & thread_mask
        no_overlap_mask = (existing_rope == 0) & thread_mask

        new_intensity = existing_rope.astype(np.float32)
        new_intensity[no_overlap_mask] = thread_section[no_overlap_mask]

        if label_volume is not None:
            existing_labels = label_volume[z, y_start:y_end, x_start:x_end]
            new_labels = existing_labels.copy()
            new_labels[no_overlap_mask] = thread_id

        if np.any(overlap_mask):
            existing_norm = existing_rope.astype(np.float32) / 255.0
            thread_norm = thread_section.astype(np.float32) / 255.0

            thread_wins = overlap_mask & (thread_norm >= existing_norm)
            existing_wins = overlap_mask & (thread_norm < existing_norm)

            if np.any(thread_wins):
                compressed = (
                    thread_section[thread_wins].astype(np.float32)
                    + existing_rope[thread_wins].astype(np.float32) * compression_factor
                    + compression_boost
                )
                new_intensity[thread_wins] = np.minimum(255, compressed)
                if label_volume is not None:
                    new_labels[thread_wins] = thread_id

            if np.any(existing_wins):
                compressed = (
                    existing_rope[existing_wins].astype(np.float32)
                    + thread_section[existing_wins].astype(np.float32)
                    * compression_factor
                    + compression_boost * secondary_compression
                )
                new_intensity[existing_wins] = np.minimum(255, compressed)

        combined_rope[z, y_start:y_end, x_start:x_end] = new_intensity.astype(np.uint8)
        if label_volume is not None:
            label_volume[z, y_start:y_end, x_start:x_end] = new_labels


def rope(
    shape: tuple = (300, 80, 80),
    num_threads: int = 18,
    thread_thickness: int = 10,
    twist_rate: float = 2.0,
    compression_factor: float = 0.0,
    thread_spacing: float = 0.8,
    noise_scale: float = 0.03,
    seed: int = 0,
    dtype: str = 'uint8',
    return_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a twisted rope-like 3D structure.

    Creates an elongated cylindrical volume composed of multiple twisted threads
    arranged in concentric rings, resembling a twisted rope or fiber bundle.

    Args:
        shape: Dimensions of output volume (length, diameter, diameter).
            First dimension is the rope length. Defaults to (300, 80, 80).
        num_threads: Number of threads in the rope. Defaults to 18.
        thread_thickness: Thickness of individual threads in voxels. Defaults to 10.
        twist_rate: Controls twist rate along length. Higher = tighter twist. Defaults to 2.0.
        compression_factor: Thread compression at overlaps (0-1). Defaults to 0.0.
        thread_spacing: Spacing between threads as ratio of thickness. Defaults to 0.8.
        noise_scale: Surface texture detail for threads. Defaults to 0.03.
        seed: Random seed for reproducibility. Defaults to 0.
        dtype: Output data type. Defaults to 'uint8'.
        return_labels: If True, also return label array for each thread. Defaults to False.

    Returns:
        Rope volume as NumPy array, or tuple of (rope_volume, labels) if return_labels=True.

    Examples:
        >>> import qim3d
        >>> rope_vol = qim3d.generate.rope(num_threads=12, twist_rate=1.5)
        >>> rope_vol, labels = qim3d.generate.rope(return_labels=True)

    """
    # Validate inputs
    if num_threads < 1:
        msg = 'num_threads must be at least 1'
        raise ValueError(msg)
    if thread_thickness <= 0:
        msg = 'thread_thickness must be positive'
        raise ValueError(msg)
    if not all(s > 0 for s in shape):
        msg = 'All shape dimensions must be positive'
        raise ValueError(msg)
    if shape[1] != shape[2]:
        msg = (
            f'Rope cross-section must be square (shape[1] == shape[2]). '
            f'Got {shape[1]} != {shape[2]}'
        )
        raise ValueError(msg)
    if twist_rate < 0:
        msg = 'twist_rate must be non-negative'
        raise ValueError(msg)
    if not 0 <= compression_factor <= 1:
        msg = 'compression_factor must be between 0 and 1'
        raise ValueError(msg)
    if thread_spacing <= 0:
        msg = 'thread_spacing must be positive'
        raise ValueError(msg)

    log.info(f'Generating rope with {num_threads} threads in shape {shape}')

    rng = np.random.default_rng(seed)

    rope_length, rope_diameter = shape[0], shape[1]
    rope_shape = (rope_length, rope_diameter, rope_diameter)
    combined_rope = np.zeros(rope_shape, dtype=np.uint8)
    label_volume = np.zeros(rope_shape, dtype=np.uint8) if return_labels else None

    rope_center_y, rope_center_x = rope_shape[1] // 2, rope_shape[2] // 2
    max_radius = (rope_diameter - thread_thickness) // 2

    # Calculate thread positions in concentric rings
    positions = []
    ring_spacing = thread_thickness * thread_spacing
    num_rings = max(1, int(max_radius / ring_spacing))
    threads_placed = 0

    for ring_idx in range(num_rings):
        if threads_placed >= num_threads:
            break

        if ring_idx == 0:
            ring_radius, threads_in_ring = 0, min(1, num_threads)
        else:
            ring_radius = ring_idx * ring_spacing
            circumference = 2 * np.pi * ring_radius
            threads_in_ring = min(
                int(circumference / (thread_thickness * thread_spacing)),
                num_threads - threads_placed,
            )

        for i in range(threads_in_ring):
            angle = (
                0
                if threads_in_ring == 1
                else (2 * np.pi * i) / threads_in_ring
                + (ring_idx % 2) * (np.pi / threads_in_ring)
            )
            positions.append((ring_radius, angle, ring_idx))
            threads_placed += 1

    for thread_idx in range(min(num_threads, len(positions))):
        radius, base_angle, ring_idx = positions[thread_idx]

        thread = _generate_twisted_thread(
            rope_length,
            thread_thickness,
            twist_rate,
            thread_idx * (360 / num_threads),
            noise_scale,
            seed + thread_idx,
        )

        _integrate_thread(
            combined_rope,
            label_volume,
            thread,
            rope_center_y,
            rope_center_x,
            radius,
            base_angle,
            twist_rate,
            thread_idx + 1,
            compression_factor,
        )

    # Apply final smoothing
    gaussian_sigma = 0.5  # Smoothing kernel size
    original_weight = 0.7  # Weight for original intensity (preserves detail)
    smoothed_weight = 0.3  # Weight for smoothed intensity (removes sharp edges)

    rope_mask = combined_rope > 0
    smoothed = gaussian(combined_rope.astype(np.float32), sigma=gaussian_sigma)
    result = combined_rope.astype(np.float32)
    result[rope_mask] = (
        original_weight * combined_rope[rope_mask]
        + smoothed_weight * smoothed[rope_mask]
    )

    result = result.astype(dtype)

    if return_labels:
        return result, label_volume
    return result
