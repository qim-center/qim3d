"""
This module provides a collection of colormaps useful for 3D visualization.
"""

import colorsys
from typing import Union, Tuple
import numpy as np
import math
from matplotlib.colors import LinearSegmentedColormap


def rearrange_colors(randRGBcolors_old, min_dist=0.5):
    # Create new list for re-arranged colors
    randRGBcolors_new = [randRGBcolors_old.pop(0)]

    while randRGBcolors_old:
        previous_color = randRGBcolors_new[-1]
        found = False

        # Find a next color that is at least min_dist away from previous color
        for color in randRGBcolors_old:
            if math.dist(color, previous_color) > min_dist:
                randRGBcolors_new.append(color)
                randRGBcolors_old.remove(color)
                found = True
                break

        # If no color was found, start over with the first color in the list
        if not found:
            randRGBcolors_new.append(randRGBcolors_old.pop(0))

    return randRGBcolors_new


def segmentation(
    num_labels: int,
    style: str = "bright",
    first_color_background: bool = True,
    last_color_background: bool = False,
    background_color: Union[Tuple[float, float, float], str] = (0.0, 0.0, 0.0),
    min_dist: int = 0.5,
    seed: int = 19,
) -> LinearSegmentedColormap:
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks

    Args:
        num_labels (int): Number of labels (size of colormap).
        style (str, optional): 'bright' for strong colors, 'soft' for pastel colors, 'earth' for yellow/green/blue colors, 'ocean' for blue/purple/pink colors. Defaults to 'bright'.
        first_color_background (bool, optional): If True, the first color is used as background. Defaults to True.
        last_color_background (bool, optional): If True, the last color is used as background. Defaults to False.
        background_color (tuple or str, optional): RGB tuple or string for background color. Can be "black" or "white". Defaults to (0.0, 0.0, 0.0).
        min_dist (int, optional): Minimum distance between neighboring colors. Defaults to 0.5.
        seed (int, optional): Seed for random number generator. Defaults to 19.

    Returns:
        color_map (matplotlib.colors.LinearSegmentedColormap): Colormap for matplotlib


    Example:
        ```python
        import qim3d

        cmap_bright = qim3d.viz.colormaps.segmentation(num_labels=100, style = 'bright', first_color_background=True, background_color="black", min_dist=0.7)
        cmap_soft = qim3d.viz.colormaps.segmentation(num_labels=100, style = 'soft', first_color_background=True, background_color="black", min_dist=0.2)
        cmap_earth = qim3d.viz.colormaps.segmentation(num_labels=100, style = 'earth', first_color_background=True, background_color="black", min_dist=0.8)
        cmap_ocean = qim3d.viz.colormaps.segmentation(num_labels=100, style = 'ocean', first_color_background=True, background_color="black", min_dist=0.9)

        display(cmap_bright)
        display(cmap_soft)
        display(cmap_earth)
        display(cmap_ocean)
        ```
        ![colormap objects](../../assets/screenshots/viz-colormaps-objects-all.png)

        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary = qim3d.filters.gaussian(vol, sigma = 2) < 60
        labeled_volume, num_labels = qim3d.segmentation.watershed(binary)

        color_map = qim3d.viz.colormaps.segmentation(num_labels, style = 'bright')
        qim3d.viz.slicer(labeled_volume, slice_axis = 1, color_map=color_map)
        ```
        ![colormap objects](../../assets/screenshots/viz-colormaps-objects.gif)

    Tip:
        It can be easily used when calling visualization functions as
        ```python
        qim3d.viz.slices_grid(segmented_volume, color_map = 'objects')
        ```
        which automatically detects number of unique classes 
        and creates the colormap object with defualt arguments.

    Tip:
        The `min_dist` parameter can be used to control the distance between neighboring colors.
        ![colormap objects mind_dist](../../assets/screenshots/viz-colormaps-min_dist.gif)
    """
    from skimage import color

    # Check style
    if style not in ("bright", "soft", "earth", "ocean"):
        raise ValueError(
            f'Please choose "bright", "soft", "earth" or "ocean" for style in qim3dCmap not "{style}"'
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

    # Add one to num_labels to include the background color
    num_labels += 1

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
            for i in range(num_labels)
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
            for i in range(num_labels)
        ]

    # Generate color map for earthy colors, based on LAB
    if style == "earth":
        randLABColors = [
            (
                rng.uniform(low=25, high=110),
                rng.uniform(low=-120, high=70),
                rng.uniform(low=-70, high=70),
            )
            for i in range(num_labels)
        ]

        # Convert LAB list to RGB
        randRGBcolors = []
        for LabColor in randLABColors:
            randRGBcolors.append(color.lab2rgb([[LabColor]])[0][0].tolist())

    # Generate color map for ocean colors, based on LAB
    if style == "ocean":
        randLABColors = [
            (
                rng.uniform(low=0, high=110),
                rng.uniform(low=-128, high=160),
                rng.uniform(low=-128, high=0),
            )
            for i in range(num_labels)
        ]

        # Convert LAB list to RGB
        randRGBcolors = []
        for LabColor in randLABColors:
            randRGBcolors.append(color.lab2rgb([[LabColor]])[0][0].tolist())

    # Re-arrange colors to have a minimum distance between neighboring colors
    randRGBcolors = rearrange_colors(randRGBcolors, min_dist)

    # Set first and last color to background
    if first_color_background:
        randRGBcolors[0] = background_color

    if last_color_background:
        randRGBcolors[-1] = background_color

    # Create colormap
    objects = LinearSegmentedColormap.from_list("objects", randRGBcolors, N=num_labels)

    return objects


