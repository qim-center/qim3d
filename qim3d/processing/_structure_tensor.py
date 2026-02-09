"""Wrapper for the structure tensor function from the structure_tensor package."""

import logging

import numpy as np
from IPython.display import display


def structure_tensor(
    volume: np.ndarray,
    sigma: float = 1.0,
    rho: float = 6.0,
    base_noise: bool = True,
    smallest: bool = True,
    visualize: bool = False,
    **viz_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Structure Tensor for 3D local orientation analysis.

    The structure tensor is a matrix-based representation of partial derivatives. It is used to analyze the local orientation of features (like fibers, veins, or layers) in a 3D volume. By examining the eigenvalues and eigenvectors of this tensor, one can determine the dominant direction of structures at every voxel.

    This implementation wraps the [structure-tensor package](https://github.com/Skielex/structure-tensor/), which offers fast, GPU-accelerated computation suitable for large datasets (e.g., micro-CT).

    Args:
        volume (np.ndarray): The input 3D volume.
        sigma (float, optional): The noise scale. This defines the standard deviation of the Gaussian kernel used to smooth the image *before* calculating gradients. Features smaller than this scale are suppressed. Defaults to 1.0.
        rho (float, optional): The integration scale. This defines the size of the neighborhood over which orientation information is averaged (integrated). Larger values yield smoother orientation fields but may blur distinct features. Defaults to 6.0.
        base_noise (bool, optional): If `True`, adds infinitesimal noise to the volume before processing. This prevents numerical instability (NaNs) in uniform regions where gradients are zero. Defaults to `True`.
        smallest (bool, optional): If `True`, returns only the eigenvector corresponding to the **smallest** eigenvalue. In gradient-based analysis, the smallest eigenvalue typically corresponds to the direction of least change (i.e., **along** the fiber/structure axis). If `False`, returns all eigenvectors. Defaults to `True`.
        visualize (bool, optional): If `True`, immediately displays a visualization of the vector field using `qim3d.viz.vectors`. Defaults to `False`.
        **viz_kwargs (Any): Additional keyword arguments passed to the visualization function (e.g., `n_vectors`, `opacity`).

    Returns:
        (val, vec): A tuple containing:
            * **val** (np.ndarray): An array of shape `(3, Z, Y, X)` containing the eigenvalues sorted in ascending order.
            * **vec** (np.ndarray):
                * If `smallest` is `True`: An array of shape `(3, Z, Y, X)` representing the vector components (z, y, x) of the primary orientation.
                * If `smallest` is `False`: An array of shape `(3, 3, Z, Y, X)` containing all three eigenvectors sorted by eigenvalue.

    Raises:
        ValueError: If the input `volume` is not 3D.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fibers_150x150x150
        val, vec = qim3d.processing.structure_tensor(vol, visualize = True, axis = 1)
        ```
        ![structure tensor](../../assets/screenshots/structure_tensor_visualization_fibers.gif)

    !!! info "Runtime and memory usage"
        ![structure tensor estimate time and mem](../../assets/screenshots/Structure_tensor_time_mem_estimation.png)
        Performance computed on Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz.

    !!! quote "Reference"
        Jeppesen, N., et al. "Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis." Composites Part A: Applied Science and Manufacturing 149 (2021): 106541.
        <https://doi.org/10.1016/j.compositesa.2021.106541>

        ```bibtex
        @article{JEPPESEN2021106541,
        title = {Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis},
        journal = {Composites Part A: Applied Science and Manufacturing},
        volume = {149},
        pages = {106541},
        year = {2021},
        issn = {1359-835X},
        doi = {[https://doi.org/10.1016/j.compositesa.2021.106541](https://doi.org/10.1016/j.compositesa.2021.106541)},
        url = {[https://www.sciencedirect.com/science/article/pii/S1359835X21002633](https://www.sciencedirect.com/science/article/pii/S1359835X21002633)},
        author = {N. Jeppesen and L.P. Mikkelsen and A.B. Dahl and A.N. Christensen and V.A. Dahl}
        }
        ```
    """
    previous_logging_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.CRITICAL)
    import structure_tensor as st

    logging.getLogger().setLevel(previous_logging_level)

    if volume.ndim != 3:
        msg = 'The input volume must be 3D'
        raise ValueError(msg)

    # Ensure volume is a float
    if volume.dtype != np.float32 and volume.dtype != np.float64:
        volume = volume.astype(np.float32)

    if base_noise:
        # Add small noise to the volume
        # FIXME: This is a temporary solution to avoid uniform regions with constant values
        # in the volume, which lead to numerical issues in the structure tensor computation
        vol_noisy = volume + np.random.default_rng(seed=0).uniform(
            0, 1e-10, size=volume.shape
        )

        # Compute the structure tensor (of volume with noise)
        s_vol = st.structure_tensor_3d(vol_noisy, sigma, rho)

    else:
        # Compute the structure tensor (of volume without noise)
        s_vol = st.structure_tensor_3d(volume, sigma, rho)

    # Compute the eigenvalues and eigenvectors of the structure tensor
    full = not smallest
    print(
        f'Computing eigenvalues and eigenvectors of the structure tensor, full = {full}'
    )
    val, vec = st.eig_special_3d(s_vol, full=full, eigenvalue_order='asc')

    if visualize:
        from qim3d.viz import vectors

        display(vectors(volume, vec, **viz_kwargs))

    return val, vec
