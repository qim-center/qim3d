import qim3d
import numpy as np
from pathlib import Path
import os
import pytest

# Load blobs volume into memory
vol = qim3d.examples.blobs_256x256

# Ceate memory map to blobs 
blobs_path = Path(qim3d.__file__).parents[0] / "img_examples" / "blobs_256x256.tif"
vol_memmap = qim3d.io.load(blobs_path,virtual_stack=True)

def test_load_shape():
    assert vol.shape == vol_memmap.shape == (256,256)
    
def test_load_type():
    assert isinstance(vol,np.ndarray)

def test_load_type_memmap():
    assert isinstance(vol_memmap,np.memmap)

def test_invalid_path():
    invalid_path = os.path.join('this','path','doesnt','exist.tif')

    with pytest.raises(ValueError,match='Invalid path'):
        qim3d.io.load(invalid_path)

def test_did_you_mean():
    # Remove last two characters from the path
    blobs_path_misspelled = str(blobs_path)[:-2]

    with pytest.raises(ValueError,match=f"Invalid path.\nDid you mean '{blobs_path}'?"):
        qim3d.io.load(blobs_path_misspelled)