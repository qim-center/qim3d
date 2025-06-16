import numpy as np

import qim3d

def test_mean_std_intensity():
    # Generate a synthetic 3D object
    volume = qim3d.generate.volume()

    # Min and max values for the volume
    min_value = np.min(volume)
    max_value = np.max(volume)

    # Generate a mask for the bottom right quarter of the volume
    mask = np.zeros_like(volume, dtype=bool)
    mask[volume.shape[0]//2:, volume.shape[1]//2:, volume.shape[2]//2:] = True

    # Compute mean and standard deviation of intensity 
    mean_volume1, std_volume1 = qim3d.features.mean_std_intensity(volume)  # Without mask
    mean_volume2, std_volume2 = qim3d.features.mean_std_intensity(volume, mask=mask)  # With mask

    # Assertions
    assert isinstance(mean_volume1, float), "Mean intensity should be a float"
    assert isinstance(std_volume1, float), "Standard deviation should be a float"
    assert mean_volume1 >= 0 and std_volume1 >= 0, "Mean and standard deviation should be non-negative"
    assert mean_volume2 >= 0 and std_volume2 >= 0, "Mean and standard deviation should be non-negative"
    assert mean_volume1 >= min_value and mean_volume1 <= max_value, "Mean intensity should be within the volume's intensity range"
    assert mean_volume2 >= min_value and mean_volume2 <= max_value, "Mean intensity should be within the volume's intensity range"
    assert std_volume2 < std_volume1, "Standard deviation should be lower for masked volume"

def test_size():
    # Generate a synthetic 3D object
    volume = qim3d.generate.volume()

    # Extract mesh from the volume
    mesh = qim3d.mesh.from_volume(volume)

    # Generate a mask for the bottom right quarter of the volume
    mask = np.zeros_like(volume, dtype=bool)
    mask[volume.shape[0]//2:, volume.shape[1]//2:, volume.shape[2]//2:] = True

    # Compute size from both volume and mesh
    size_volume = qim3d.features.size(volume)
    size_volume_masked = qim3d.features.size(volume, mask=mask)
    size_mesh = qim3d.features.size(mesh)

    # Assertions
    assert isinstance(size_volume, float) and isinstance(size_mesh, float), "Size should be a float"
    assert size_volume > 0 and size_mesh > 0, "Size should be positive"
    assert size_volume == size_mesh, "Size from volume and mesh should be equal"
    assert size_volume_masked < size_volume, "Size with mask applied should be less than size without mask applied"