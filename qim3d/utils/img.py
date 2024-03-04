import numpy as np

def overlay_rgb_images(background, foreground, alpha=0.5):
    """Overlay multiple RGB foreground onto an RGB background image using alpha blending.

    Args:
        background (numpy.ndarray): The background RGB image.
        foreground (numpy.ndarray): The foreground RGB image (usually masks).
        alpha (float, optional): The alpha value for blending. Defaults to 0.5.

    Returns:
        numpy.ndarray: The composite RGB image with overlaid foreground.

    Raises:
        ValueError: If input images have different shapes.

    Note:
        - The function performs alpha blending to overlay the foreground onto the background.
        - It ensures that the background and foreground have the same shape before blending.
        - It calculates the maximum projection of the foreground and blends them onto the background.
        - Brightness outside the foreground is adjusted to maintain consistency with the background.
    """

    # Igonore alpha in case its there
    background = background[..., :3]
    foreground = foreground[..., :3]

    # Ensure both images have the same shape
    if background.shape != foreground.shape:
        raise ValueError("Input images must have the same shape")

    # Perform alpha blending
    foreground_max_projection = np.amax(foreground, axis=2)
    foreground_max_projection = np.stack((foreground_max_projection,) * 3, axis=-1)

    # Normalize if we have something
    if np.max(foreground_max_projection) > 0:
        foreground_max_projection = foreground_max_projection / np.max(foreground_max_projection)

    composite = background * (1 - alpha) + foreground * alpha
    composite = np.clip(composite, 0, 255).astype("uint8")

    # Adjust brightness outside foreground
    composite = composite + (background * (1 - alpha)) * (1 - foreground_max_projection)

    return composite.astype("uint8")
