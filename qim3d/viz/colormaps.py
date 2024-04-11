import colorsys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from qim3d.io.logger import log


def objects(
    nlabels,
    style="bright",
    first_color_background=True,
    last_color_background=False,
    background_color=(0.0, 0.0, 0.0),
    seed=19,
):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks

    Args:
        nlabels (int): Number of labels (size of colormap)
        style (str, optional): 'bright' for strong colors, 'soft' for pastel colors. Defaults to 'bright'.
        first_color_background (bool, optional): Option to use first color as background. Defaults to True.
        last_color_background (bool, optional): Option to use last color as background. Defaults to False.
        seed (int, optional): Seed for random number generator. Defaults to 19.

    Returns:
        cmap (matplotlib.colors.LinearSegmentedColormap): Colormap for matplotlib
    """
    # Check style
    if style not in ("bright", "soft"):
        raise ValueError(
            f'Please choose "bright" or "soft" for style in qim3dCmap not "{style}"'
        )

    # Translate strings to background color
    color_dict = {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0)}
    if not isinstance(background_color, tuple):
        try:
            background_color = color_dict[background_color]
        except KeyError:
            raise ValueError(
                f'Invalid color name "{background_color}". Please choose from {list(color_dict.keys())}.'
            )

    # Add one to nlabels to include the background color
    nlabels += 1

    # Create a new random generator, to locally set seed
    rng = np.random.default_rng(seed)

    # Generate color map for bright colors, based on hsv
    if style == "bright":
        randHSVcolors = [
            (
                rng.uniform(low=0.0, high=1),
                rng.uniform(low=0.4, high=1),
                rng.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if style == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                rng.uniform(low=low, high=high),
                rng.uniform(low=low, high=high),
                rng.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

    # Set first and last color to background
    if first_color_background:
        randRGBcolors[0] = background_color

    if last_color_background:
        randRGBcolors[-1] = background_color

    # Create colormap
    objects_cmap = LinearSegmentedColormap.from_list(
        "objects_cmap", randRGBcolors, N=nlabels
    )

    return objects_cmap
