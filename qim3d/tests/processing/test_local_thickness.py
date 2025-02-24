import numpy as np
from skimage.draw import disk, ellipsoid

import qim3d


def test_local_thickness_2d():
    # Create a binary 2D image
    shape = (100, 100)
    img = np.zeros(shape, dtype=bool)
    rr1, cc1 = disk((65, 65), 30, shape=shape)
    rr2, cc2 = disk((25, 25), 20, shape=shape)
    img[rr1, cc1] = True
    img[rr2, cc2] = True

    lt_manual = np.zeros(shape)
    lt_manual[rr1, cc1] = 30
    lt_manual[rr2, cc2] = 20

    # Compute local thickness
    lt = qim3d.processing.local_thickness(img)

    assert np.allclose(lt, lt_manual, rtol=1e-1)


def test_local_thickness_3d():
    disk3d = ellipsoid(15, 15, 15)

    # Remove weird border pixels
    border_thickness = 2
    disk3d = disk3d[
        border_thickness:-border_thickness,
        border_thickness:-border_thickness,
        border_thickness:-border_thickness,
    ]
    disk3d = np.pad(disk3d, border_thickness, mode='constant')

    lt = qim3d.processing.local_thickness(disk3d)

    lt_manual = np.zeros(disk3d.shape)
    lt_manual[disk3d] = 15

    assert np.allclose(lt, lt_manual, rtol=1e-1)
