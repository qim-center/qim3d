"""Wrapper for the local thickness function from the localthickness package including visualization functions."""

import numpy as np
from typing import Optional
from qim3d.utils import log
import qim3d
from IPython.display import display

def local_thickness(
    image: np.ndarray,
    scale: float = 1,
    mask: Optional[np.ndarray] = None,
    visualize: bool = False,
    **viz_kwargs
) -> np.ndarray:
    """Wrapper for the local thickness function from the [local thickness package](https://github.com/vedranaa/local-thickness)

    The "Fast Local Thickness" by Vedrana Andersen Dahl and Anders Bjorholm Dahl from the Technical University of Denmark is a efficient algorithm for computing local thickness in 2D and 3D images.
    Their method significantly reduces computation time compared to traditional algorithms by utilizing iterative dilation with small structuring elements, rather than the large ones typically used.
    This approach allows the local thickness to be determined much faster, making it feasible for high-resolution volumetric data that are common in contemporary 3D microscopy.

    Testing against conventional methods and other Python-based tools like PoreSpy shows that the new algorithm is both accurate and faster, offering significant improvements in processing time for large datasets.


    Args:
        image (np.ndarray): 2D or 3D NumPy array representing the image/volume.
            If binary, it will be passed directly to the local thickness function.
            If grayscale, it will be binarized using Otsu's method.
        scale (float, optional): Downscaling factor, e.g. 0.5 for halving each dim of the image.
            Default is 1.
        mask (np.ndarray or None, optional): Binary mask of the same size of the image defining parts of the
            image to be included in the computation of the local thickness. Default is None.
        visualize (bool, optional): Whether to visualize the local thickness. Default is False.
        **viz_kwargs (Any): Additional keyword arguments passed to `qim3d.viz.local_thickness`. Only used if `visualize=True`.

    Returns:
        local_thickness (np.ndarray): 2D or 3D NumPy array representing the local thickness of the input image/volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fly_150x256x256
        lt_vol = qim3d.processing.local_thickness(vol, visualize=True, axis=0)
        ```
        ![local thickness 3d](../../assets/screenshots/local_thickness_3d.gif)

        ```python
        import qim3d

        # Generate synthetic collection of blobs
        vol, labels = qim3d.generate.noise_object_collection(num_objects=15)

        # Extract one slice to show that localthickness works on 2D slices too
        slice = vol[:,:,50]
        lt_blobs = qim3d.processing.local_thickness(slice, visualize=True)

        ```
        ![local thickness 2d](../../assets/screenshots/local_thickness_2d.png)

    !!! info "Runtime and memory usage of the local thickness method for different volume sizes"
        ![local thickness estimate time and mem](../../assets/screenshots/Local_thickness_time_mem_estimation.png)

        Performance computed on Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz.

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
    import localthickness as lt
    from skimage.filters import threshold_otsu

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
        display(qim3d.viz.local_thickness(image, local_thickness, **viz_kwargs))

    return local_thickness
