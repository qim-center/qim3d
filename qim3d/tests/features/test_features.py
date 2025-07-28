import math
import numpy as np

import qim3d

def test_area():
    # Generate synthetic object
    volume = qim3d.generate.volume()
    mesh = qim3d.mesh.from_volume(volume)

    # Generate a mask for the bottom right quarter of the volume
    mask = np.zeros_like(volume, dtype=bool)
    mask[volume.shape[0]//2:, volume.shape[1]//2:, volume.shape[2]//2:] = True

    # Compute area from both volume and mesh
    area_volume = qim3d.features.area(volume)
    area_volume_masked = qim3d.features.area(volume, mask=mask)
    area_mesh = qim3d.features.area(mesh)

    # Assertions
    assert isinstance(area_volume, float) and isinstance(area_mesh, float), "Area should be a float"
    assert area_volume > 0 and area_mesh > 0, "Area should be positive"
    assert math.isclose(area_volume, area_mesh, rel_tol=1e-9), "Area from volume and mesh should be equal"
    assert area_volume_masked < area_volume, "Area with mask applied should be less than area without mask applied"

def test_volume():
    # Generate synthetic object
    volume = qim3d.generate.volume()
    mesh = qim3d.mesh.from_volume(volume)

    # Generate a mask for the bottom right quarter of the volume
    mask = np.zeros_like(volume, dtype=bool)
    mask[volume.shape[0]//2:, volume.shape[1]//2:, volume.shape[2]//2:] = True

    # Compute volume from both volume and mesh
    volume_value = qim3d.features.volume(volume)
    volume_value_masked = qim3d.features.volume(volume, mask=mask)
    mesh_volume = qim3d.features.volume(mesh)

    # Assertions
    assert isinstance(volume_value, float) and isinstance(mesh_volume, float), "Volume should be a float"
    assert volume_value > 0 and mesh_volume > 0, "Volume should be positive"
    assert math.isclose(volume_value, mesh_volume, rel_tol=1e-9), "Volume from volume and mesh should be equal"
    assert volume_value_masked < volume_value, "Volume with mask applied should be less than volume without mask applied"

def test_sphericity():
    # Generate synthetic objects of different noise levels
    volume_low_noise = qim3d.generate.volume(noise_scale=0.01)
    volume_med_noise = qim3d.generate.volume(noise_scale=0.05)
    volume_high_noise = qim3d.generate.volume(noise_scale=0.1)

    # Compute sphericity
    sphericity_low = qim3d.features.sphericity(volume_low_noise)
    sphericity_med = qim3d.features.sphericity(volume_med_noise)
    sphericity_high = qim3d.features.sphericity(volume_high_noise)

    # Assertions
    assert isinstance(sphericity_low, float) and isinstance(sphericity_med, float) and isinstance(sphericity_high, float), "Sphericity should be a float"
    assert sphericity_low >= 0 and sphericity_med >= 0 and sphericity_high >= 0, "Sphericity should be non-negative"
    assert sphericity_low >= sphericity_med >= sphericity_high, "Sphericity should decrease with noise level"

def test_mean_std_intensity():
    # Generate synthetic object
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
    # Generate synthetic object
    volume = qim3d.generate.volume()
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
    assert math.isclose(size_volume, size_mesh, rel_tol=1e-9), "Size from volume and mesh should be equal"
    assert size_volume_masked < size_volume, "Size with mask applied should be less than size without mask applied"

def test_roughness():
    # Generate synthetic objects of different noise levels
    volume_low_noise = qim3d.generate.volume(noise_scale=0.01)
    volume_high_noise = qim3d.generate.volume(noise_scale=0.05)

    # Extract meshes from the volumes
    mesh_low_noise = qim3d.mesh.from_volume(volume_low_noise)
    mesh_high_noise = qim3d.mesh.from_volume(volume_high_noise)

    # Compute roughness for volumes and meshes
    roughness_volume_low = qim3d.features.roughness(volume_low_noise)
    roughness_volume_high = qim3d.features.roughness(volume_high_noise)

    roughness_mesh_low = qim3d.features.roughness(mesh_low_noise)
    roughness_mesh_high = qim3d.features.roughness(mesh_high_noise)

    # Assertions
    assert isinstance(roughness_volume_low, float) and isinstance(roughness_volume_high, float), "Roughness should be a float"
    assert isinstance(roughness_mesh_low, float) and isinstance(roughness_mesh_high, float), "Roughness should be a float"
    assert roughness_volume_low >= 0 and roughness_volume_high >= 0, "Roughness should be non-negative"
    assert roughness_mesh_low >= 0 and roughness_mesh_high >= 0, "Roughness should be non-negative"
    assert math.isclose(roughness_volume_low, roughness_mesh_low, rel_tol=1e-9), "Roughness from volume and mesh should be equal"
    assert math.isclose(roughness_volume_high, roughness_mesh_high, rel_tol=1e-9), "Roughness from volume and mesh should be equal"
    assert roughness_volume_high > roughness_volume_low, "Roughness should increase with noise level"