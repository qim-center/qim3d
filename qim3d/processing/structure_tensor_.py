"""Wrapper for the structure tensor function from the structure_tensor package"""

from typing import Tuple
import numpy as np
import structure_tensor as st 
from qim3d.viz.structure_tensor import vectors


def structure_tensor(
    vol: np.ndarray,
    sigma: float = 1.0,
    rho: float = 6.0,
    full: bool = False,
    visualize=False,
    **viz_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for the 3D structure tensor implementation from the [structure_tensor package](https://github.com/Skielex/structure-tensor/)

    Args:
        vol (np.ndarray): 3D NumPy array representing the volume.
        sigma (float, optional): A noise scale, structures smaller than sigma will be removed by smoothing.
        rho (float, optional): An integration scale giving the size over the neighborhood in which the orientation is to be analysed.
        full (bool, optional): A flag indicating that all three eigenvalues should be returned. Default is False.
        visualize (bool, optional): Whether to visualize the structure tensor. Default is False.
        **viz_kwargs: Additional keyword arguments for passed to `qim3d.viz.vectors`. Only used if `visualize=True`.

    Raises:
        ValueError: If the input volume is not 3D.

    Returns:
        val: An array with shape `(3, *vol.shape)` containing the eigenvalues of the structure tensor.
        vec: An array with shape `(3, *vol.shape)` if `full` is `False`, otherwise `(3, 3, *vol.shape)` containing eigenvectors.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.NT_128x128x128
        val, vec = qim3d.processing.structure_tensor(vol, visualize=True, axis=2)
        ```
        ![structure tensor](assets/screenshots/structure_tensor.gif)

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

    if vol.ndim != 3:
        raise ValueError("The input volume must be 3D")
    
    # Ensure volume is a float
    if vol.dtype != np.float32 and vol.dtype != np.float64:
        vol = vol.astype(np.float32)
        

    s_vol = st.structure_tensor_3d(vol, sigma, rho)
    val, vec = st.eig_special_3d(s_vol, full=full)

    if visualize:
        display(vectors(vol, vec, **viz_kwargs))

    return val, vec
