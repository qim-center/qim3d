"""Wrapper for the local thickness function from the localthickness package including visualization functions."""

import localthickness as lt
import numpy as np
from typing import Optional
from skimage.filters import threshold_otsu
from qim3d.io.logger import log
from qim3d.viz import local_thickness as viz_local_thickness


def local_thickness(
    image: np.ndarray,
    scale: float = 1,
    mask: Optional[np.ndarray] = None,
    visualize=False,
    **viz_kwargs
) -> np.ndarray:
    """Wrapper for the local thickness function from the [local thickness package](https://github.com/vedranaa/local-thickness)

    Args:
        image (np.ndarray): 2D or 3D NumPy array representing the image/volume.
            If binary, it will be passed directly to the local thickness function.
            If grayscale, it will be binarized using Otsu's method.
        scale (float, optional): Downscaling factor, e.g. 0.5 for halving each dim of the image.
            Default is 1.
        mask (np.ndarray, optional): binary mask of the same size of the image defining parts of the
            image to be included in the computation of the local thickness. Default is None.
        visualize (bool, optional): Whether to visualize the local thickness. Default is False.
        **viz_kwargs: Additional keyword arguments passed to `qim3d.viz.local_thickness`. Only used if `visualize=True`.

    Returns:
        local_thickness (np.ndarray): 2D or 3D NumPy array representing the local thickness of the input image/volume.
    
    Example:
        ```python
        import qim3d

        fly = qim3d.examples.fly_150x256x256 # 3D volume
        lt_fly = qim3d.processing.local_thickness(fly, visualize=True, axis=0)
        ```
        ![local thickness 3d](assets/screenshots/local_thickness_3d.gif)

        ```python
        import qim3d

        blobs = qim3d.examples.blobs_256x256 # 2D image
        lt_blobs = qim3d.processing.local_thickness(blobs, visualize=True)
        ```
        ![local thickness 2d](assets/screenshots/local_thickness_2d.png)

    !!! quote "Reference"
        Dahl, V. A., & Dahl, A. B. (2023, June). Fast Local Thickness. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
        <https://doi.org/10.1109/cvprw59228.2023.00456>

        ```bibtex
        @inproceedings{Dahl_2023, title={Fast Local Thickness},
        url={http://dx.doi.org/10.1109/CVPRW59228.2023.00456},
        DOI={10.1109/cvprw59228.2023.00456},
        booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
        publisher={IEEE},
        author={Dahl, Vedrana Andersen and Dahl, Anders Bjorholm},
        year={2023},
        month=jun }

        ```
    """

    # Check if input is binary
    if np.unique(image).size > 2:
        # If not, binarize it using Otsu's method, log the threshold and compute the local thickness
        threshold = threshold_otsu(image=image)
        log.warning(
            "Input image is not binary. It will be binarized using Otsu's method with threshold: {}".format(
                threshold
            )
        )
        local_thickness = lt.local_thickness(image > threshold, scale=scale, mask=mask)
    else:
        # If it is binary, compute the local thickness directly
        local_thickness = lt.local_thickness(image, scale=scale, mask=mask)

    # Visualize the local thickness if requested
    if visualize:
        display(viz_local_thickness(image, local_thickness, **viz_kwargs))

    return local_thickness
