import numpy as np
import pytest

import qim3d


def test_wrong_ndim():
    img_2d = np.random.rand(50, 50)
    with pytest.raises(ValueError, match='The input volume must be 3D'):
        qim3d.processing.structure_tensor(img_2d, 1.5, 1.5)


def test_structure_tensor():
    volume = np.random.rand(50, 50, 50)
    val, vec = qim3d.processing.structure_tensor(volume, 1.5, 1.5)
    assert val.shape == (3, 50, 50, 50)
    assert vec.shape == (3, 50, 50, 50)
    assert np.all(val[0] <= val[1])
    assert np.all(val[1] <= val[2])
    assert np.all(val[0] <= val[2])


def test_structure_tensor_full():
    volume = np.random.rand(50, 50, 50)
    val, vec = qim3d.processing.structure_tensor(volume, 1.5, 1.5, full=True)
    assert val.shape == (3, 50, 50, 50)
    assert vec.shape == (3, 3, 50, 50, 50)
    assert np.all(val[0] <= val[1])
    assert np.all(val[1] <= val[2])
    assert np.all(val[0] <= val[2])
