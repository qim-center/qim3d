import numpy as np
import pytest

import qim3d
import qim3d.generate


def test_data_not_fit_raises():
    # A single custom volume larger than the collection → error
    large = np.zeros((10, 10, 10), dtype=np.uint8)
    with pytest.raises(
        ValueError, match=r"No custom volumes fit within collection size"
    ):
        qim3d.generate.volume_collection(
            n_volumes=1, collection_shape=(5, 5, 5), data=large
        )


def test_data_single_fit_multiple_placements():
    # Single small volume fits repeatedly
    vol = np.full((3, 3, 3), fill_value=42, dtype=np.uint8)
    coll, labels = qim3d.generate.volume_collection(
        n_volumes=2, collection_shape=(10, 10, 10), data=vol
    )

    # Collection and labels have correct shape
    assert coll.shape == (10, 10, 10)
    assert labels.shape == coll.shape

    # Intensities are either 0 or 42
    unique_int = set(np.unique(coll))
    assert unique_int == {0, 42}

    # Two distinct labels plus background
    unique_lbl = set(np.unique(labels))
    assert unique_lbl == {0, 1, 2}
