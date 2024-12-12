""" Provides a collection of visualisation functions for the Layers2d class."""
import io

import matplotlib.pyplot as plt
import numpy as np


from PIL import Image


def image_with_lines(image:np.ndarray, lines: list, line_thickness:float|int) -> Image:
    """
    Plots the image and plots the lines on top of it. Then extracts it as PIL.Image and in the same size as the input image was.
    Paramters:
    -----------
    image: Image on which we put the lines
    lines: list of 1D arrays to be plotted on top of the image
    line_thickness: how thick is the line supposed to be

    Returns:
    ----------
    image_with_lines: 
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap = 'gray')
    ax.axis('off')

    for line in lines:
        ax.plot(line, linewidth = line_thickness)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    buf.seek(0)
    return Image.open(buf).resize(size = image.squeeze().shape[::-1])

