import os
import re
from pathlib import Path

import numpy as np
import pytest

import qim3d

# Load volume into memory
vol = qim3d.examples.bone_128x128x128

# Ceate memory map to blobs
volume_path = Path(qim3d.__file__).parents[0] / 'examples' / 'bone_128x128x128.tif'
vol_memmap = qim3d.io.load(volume_path, virtual_stack=True)


def test_load_shape():
    assert vol.shape == vol_memmap.shape == (128, 128, 128)


def test_load_type():
    assert isinstance(vol, np.ndarray)


def test_load_type_memmap():
    assert isinstance(vol_memmap, np.memmap)


def test_invalid_path():
    invalid_path = os.path.join('this', 'path', 'doesnt', 'exist.tif')

    with pytest.raises(FileNotFoundError):
        qim3d.io.load(invalid_path)


def test_did_you_mean():
    # Remove last two characters from the path
    path_misspelled = str(volume_path)[:-2]

    with pytest.raises(FileNotFoundError, match=re.escape(repr(str(volume_path)))):
        qim3d.io.load(path_misspelled)
