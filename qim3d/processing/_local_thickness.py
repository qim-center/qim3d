"""Wrapper for the local thickness function from the localthickness package including visualization functions."""

import numpy as np
from IPython.display import display

import qim3d
from qim3d.utils import log


def local_thickness(
    image: np.ndarray,
    scale: float = 1,
    mask: np.ndarray | None = None,
    visualize: bool = False,
    **viz_kwargs,
) -> np.ndarray:
    """
    Computes the local thickness map for a 2D or 3D image.

    Local Thickness is a morphometric measure defined as the diameter of the largest sphere that fits entirely within the object boundary and contains the point. Intuitively, it represents the "width" of the structure at any given voxel. It is widely used to analyze pore sizes, trabecular bone thickness, or vessel diameters.

    This function wraps the [localthickness](https://github.com/vedranaa/local-thickness) package, which implements the "Fast Local Thickness" algorithm. Unlike traditional methods that use computationally expensive large structuring elements, this algorithm uses iterative dilation with small kernels, making it significantly faster and feasible for high-resolution 3D microscopy data.

    **Note:** This function requires a **binary** image. If a grayscale image is provided, it will be automatically binarized using Otsu's thresholding method.

    Args:
        image (np.ndarray): The input 2D or 3D array.
            * **Binary:** Processed directly.
            * **Grayscale:** Automatically binarized using Otsu's method (a warning will be logged).
        scale (float, optional): A downscaling factor to speed up computation. For example, `0.5` downsamples the image by half in each dimension before processing. Defaults to 1 (no scaling).
        mask (np.ndarray, optional): A binary mask of the same shape as `image`. Local thickness will only be computed for regions where the mask is `True`. Defaults to `None`.
        visualize (bool, optional): If `True`, immediately displays a visualization of the thickness map using `qim3d.viz.local_thickness`. Defaults to `False`.
        **viz_kwargs (Any): Additional keyword arguments passed to the visualization function.

    Returns:
        local_thickness (np.ndarray):
            A NumPy array of the same shape as the input, where each pixel/voxel value represents the local thickness at that point.

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
        vol, labels = qim3d.generate.volume_collection(num_volumes=15)

        # Extract one slice to show that local thickness works on 2D slices too
        slice = vol[:,:,50]
        lt_blobs = qim3d.processing.local_thickness(slice, visualize=True)

        ```
        ![local thickness 2d](../../assets/screenshots/local_thickness_2d.png)

    !!! info "Runtime and memory usage"
        ![local thickness estimate time and mem](../../assets/screenshots/Local_thickness_time_mem_estimation.png)
        Performance computed on Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz.

    !!! quote "Reference"
        Dahl, V. A., & Dahl, A. B. (2023, June). Fast Local Thickness. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
        <https://doi.org/10.1109/cvprw59228.2023.00456>

        ```bibtex
        @inproceedings{Dahl_2023, title={Fast Local Thickness},
        url={[http://dx.doi.org/10.1109/CVPRW59228.2023.00456](http://dx.doi.org/10.1109/CVPRW59228.2023.00456)},
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
            f"Input image is not binary. It will be binarized using Otsu's method with threshold: {threshold}"
        )
        local_thickness = lt.local_thickness(image > threshold, scale=scale, mask=mask)
    else:
        # If it is binary, compute the local thickness directly
        local_thickness = lt.local_thickness(image, scale=scale, mask=mask)

    # Visualize the local thickness if requested
    if visualize:
        display(qim3d.viz.local_thickness(image, local_thickness, **viz_kwargs))

    return local_thickness
