import numpy as np
import pytest

import qim3d


# Unit test for noise_volume() shape mismatch ValueError
def test_noise_volume_shape_mismatch():
    vol = np.ones([128, 128, 128])
    noise_volume_shape = (64, 64, 64)

    msg = f'Shape of input volume {vol.shape} does not match noise_volume_shape {noise_volume_shape}.'

    with pytest.raises(ValueError, match=msg):
        qim3d.generate.noise_volume(noise_volume_shape=noise_volume_shape, apply_to=vol)


# Unit test for noise_volume() voxel intensities
def test_noise_volume_intensities():
    baseline_value = 10
    min_noise_value = 25
    max_noise_value = 50

    noise_volume = qim3d.generate.noise_volume(
        noise_volume_shape=(64, 64, 64),
        baseline_value=baseline_value,
        min_noise_value=min_noise_value,
        max_noise_value=max_noise_value,
    )

    # Assertions
    assert np.min(noise_volume) >= baseline_value + min_noise_value
    assert np.max(noise_volume) <= baseline_value + max_noise_value
    assert np.mean(noise_volume) >= baseline_value
