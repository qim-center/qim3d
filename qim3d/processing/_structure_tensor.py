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
    Wrapper for the 3D structure tensor implementation from the [structure_tensor package](https://github.com/Skielex/structure-tensor/).

    The structure tensor algorithm is a method for analyzing the orientation of fiber-like structures in 3D images.
    The core of the algorithm involves computing a 3-by-3 matrix at each point in a volume, capturing the local orientation. This matrix, known as the structure tensor, is derived from the gradients of the image intensities and integrates neighborhood information using Gaussian kernels.

    The implementation here used allows for fast and efficient computation using GPU-based processing, making it suitable for large datasets.
    This efficiency is particularly advantageous for high-resolution imaging techniques like X-ray computed microtomography (μCT).

    Args:
        volume (np.ndarray): 3D NumPy array representing the volume.
        sigma (float, optional): A noise scale, structures smaller than sigma will be removed by smoothing. Defaults to 1.0.
        rho (float, optional): An integration scale giving the size over the neighborhood in which the orientation is to be analysed. Defaults to 6.0.
        base_noise (bool, optional): A flag indicating whether to add a small noise to the volume. Default is True.
        smallest (bool, optional): A flag indicating that only the eigenvector corresponding to the smallest eigenvalue should be returned. Default is True.
        visualize (bool, optional): Whether to visualize the structure tensor. Default is False.
        **viz_kwargs (Any): Additional keyword arguments for passed to `qim3d.viz.vectors`. Only used if `visualize=True`.

    Raises:
        ValueError: If the input volume is not 3D.

    Returns:
    val: An array of shape `(3, *vol.shape)` containing the eigenvalues of the
        structure tensor in ascending order.

    vec: If `smallest` is `True`, an array of shape `(3, *vol.shape)`; otherwise
        an array of shape `(3, 3, *vol.shape)`. The array contains the eigenvectors
        in ascending order, where axis 0 indexes the three eigenvectors and
        axis 1 indexes the three vector components(x,y,z).

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fibers_150x150x150
        val, vec = qim3d.processing.structure_tensor(vol, visualize = True, axis = 1)
        ```
        ![structure tensor](../../assets/screenshots/structure_tensor_visualization_fibers.gif)


    !!! info "Runtime and memory usage of the structure tensor method for different volume sizes"
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
        doi = {https://doi.org/10.1016/j.compositesa.2021.106541},
        url = {https://www.sciencedirect.com/science/article/pii/S1359835X21002633},
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
