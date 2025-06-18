import numpy as np
import pytest

import qim3d


# Unit test for background() apply_to ValueError
def test_background_apply_to_error():
    background_shape = (64, 64, 64)
    msg = f"Supply both apply_method and apply_to when applying background to a volume."

    with pytest.raises(ValueError, match=msg):
        qim3d.generate.background(background_shape=background_shape, apply_method='add')


# Unit test for background() voxel intensities
def test_background_intensities():
    baseline_value = 10
    min_noise_value = 25
    max_noise_value = 50

    background = qim3d.generate.background(
        background_shape=(64, 64, 64),
        baseline_value=baseline_value,
        generate_method='add',
        min_noise_value=min_noise_value,
        max_noise_value=max_noise_value,
    )

    # Assertions
    assert np.min(background) >= baseline_value + min_noise_value
    assert np.max(background) <= baseline_value + max_noise_value
    assert np.mean(background) >= baseline_value
