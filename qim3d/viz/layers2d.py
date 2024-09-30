""" Provides a collection of visualisation functions for the Layers2d class."""
import io

import matplotlib.pyplot as plt
import numpy as np

from qim3d.processing import layers2d as l2d

from PIL import Image
def image_with_overlay(image:np.ndarray, overlay:np.ndarray, alpha:int|float|np.ndarray = 125) -> Image:
    #TODO : also accepts Image type
    # We want to accept as many different values as possible to make convenient for the user.
    """
    Takes image and puts a transparent segmentation mask on it.

    Parameters:
    -----------
    Image: Can be grayscale or colorful, accepts all kinds of shapes, color has to be the last axis
    Overlay: If has its own alpha channel, alpha argument is ignored. 
    Alpha:  Can be ansolute value as int, relative vlaue as float or an array so different parts can differ with the transparency.

    Returns:
    ---------
    Image: PIL.Image in the original size as the image array

    Raises:
    --------
    ValueError: If there is a missmatch of shapes or alpha has an invalid value.
    """
    def check_dtype(image:np.ndarray):
        if image.dtype != np.uint8:
            minimal = np.min(image)
            if minimal < 0:
                image = image + minimal
            maximum = np.max(image)
            if maximum > 255:
                image = (image/maximum)*255
            elif maximum <= 1:
                image = image*255  
            image = np.uint8(image)
        return image

    image = check_dtype(image)
    overlay = check_dtype(overlay)
    
    if image.shape[0] != overlay.shape[0] or image.shape[1] != overlay.shape[1]:
        raise ValueError(F"The first two dimensions of overlay image must match those of background image.\nYour background image: {image.shape}\nYour overlay image: {overlay.shape}")
    
    
    if image.ndim == 3:
        if image.shape[2] < 3:
            image = np.repeat(image[:,:,:1], 3, -1)
        elif image.shape[2] > 4:
            image = image[:,:,:4]

    elif image.ndim == 2:
        image = np.repeat(image[..., None], 3, -1)

    else:
        raise ValueError(F"Background image must have 2 or 3 dimensions. Yours have {image.ndim}")
    
    
    
    if isinstance(alpha, (float, int)):
        if alpha<0:
            raise ValueError(F"Alpha can not be negative. You passed {alpha}")
        elif alpha<=1:
            alpha = int(255*alpha)
        elif alpha> 255:
            alpha = 255
        else:
            alpha = int(alpha)

    elif isinstance(alpha, np.ndarray):
        if alpha.ndim == 3:
            alpha = alpha[..., :1] # Making sure it is only one layer
        elif alpha.ndim == 2:
            alpha = alpha[..., None] # Making sure it has 3 dimensions
        else:
            raise ValueError(F"If alpha is numpy array, it must have 2 or 3 dimensions. Your have {alpha.ndim}")
        
        # We have not checked ndims of overlay
        try:
            if alpha.shape[0] != overlay.shape[0] or alpha.shape[1] != overlay.shape[1]:
                raise ValueError(F"The first two dimensions of alpha must match those of overlay image.\nYour alpha: {alpha.shape}\nYour overlay: {overlay.shape}")
        except IndexError:
            raise ValueError(F"Overlay image must have 2 or 3 dimensions. Yours have {overlay.ndim}")
        

    if overlay.ndim == 3:
        if overlay.shape[2] < 3:
            overlay = np.repeat(overlay[..., :1], 4, -1)
            if alpha is None:
                raise ValueError("Alpha can not be None if overlay image doesn't have alpha channel")
            overlay[..., 3] = alpha
        elif overlay.shape[2] == 3:
            if isinstance(alpha, int):
                overlay = np.concatenate((overlay, np.full((overlay.shape[0], overlay.shape[1], 1,), alpha, dtype = np.uint8)), axis = -1)
            elif isinstance(alpha, np.ndarray):
                overlay = np.concatenate((overlay, alpha), axis = -1)

        elif overlay.shape[2]>4:
            raise ValueError(F"Overlay image can not have more than 4 channels. Yours have {overlay.shape[2]}")

    elif overlay.ndim == 2:
        overlay = np.repeat(overlay[..., None], 4, axis = -1)
        overlay[..., 3] = alpha
    else:
        raise ValueError(F"Overlay image must have 2 or 3 dimensions. Yours have {overlay.ndim}")
    
    background = Image.fromarray(image)
    overlay = Image.fromarray(overlay)
    background.paste(overlay, mask = overlay)
    return background


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

