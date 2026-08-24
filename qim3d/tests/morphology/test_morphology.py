import numpy as np
import pytest
import qim3d


def test_dilate():
    vol = np.random.rand(50, 50, 50)

    p = 20
    vol_padded = qim3d.operations.pad(vol, x_axis=p, y_axis=p, z_axis=p)
    assert vol_padded.shape == (vol.shape[0]+p*2, vol.shape[1]+p*2, vol.shape[2]+p*2)

    s = 5
    strel = np.ones((s,s,s))
    vol_dilated = qim3d.morphology.dilate(vol_padded, strel, method='ndi')


    vol_trimmed = qim3d.operations.trim(vol_dilated)
    assert vol_trimmed.shape[0] <= vol_dilated.shape[0] and vol_trimmed.shape[1] <= vol_dilated.shape[1] and vol_trimmed.shape[2] <= vol_dilated.shape[2]

    vol_pad2 = qim3d.operations.pad_to(vol_trimmed, (100,100,100))
    assert vol_pad2.shape == (100,100,100)

def test_erode():
    vol = np.random.rand(50, 50, 50)

    s = 5
    strel = np.ones((s,s,s))
    vol_eroded = qim3d.morphology.erode(vol, strel, method='ndi')
    assert vol.shape == vol_eroded.shape

def test_opening():
    vol = np.random.rand(50, 50, 50)

    s = 5
    strel = np.ones((s,s,s))
    vol_opened = qim3d.morphology.opening(vol, strel, method='ndi')

    assert vol_opened.shape == vol.shape

def test_closing():
    vol = np.random.rand(50, 50, 50)

    s = 5
    strel = np.ones((s,s,s))
    vol_opened = qim3d.morphology.closing(vol, strel, method='ndi')

    assert vol_opened.shape == vol.shape


def test_black_tophat():
    vol = np.random.rand(50, 50, 50)

    s = 5
    strel = np.ones((s,s,s))
    vol_opened = qim3d.morphology.black_tophat(vol, strel, method='ndi')

    assert vol_opened.shape == vol.shape

def test_white_tophat():
    vol = np.random.rand(50, 50, 50)

    s = 5
    strel = np.ones((s,s,s))
    vol_opened = qim3d.morphology.white_tophat(vol, strel, method='ndi')

    assert vol_opened.shape == vol.shape
