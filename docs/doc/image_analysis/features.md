# Feature extraction

The `qim3d` library provides a set of methods for feature extraction on volumetric data.

!!! note "General usage"
    - All features assume a single connected object in the input.
    - All feature functions accept either a 3D volume or a mesh as input.
    - If a volume is provided, it is typically binarized (using the provided threshold or Otsu's method by default) and converted to a mesh before feature extraction.
    - A mask can be provided to restrict feature extraction to a specific region of interest in the volume.
    - If a mesh is provided, the threshold and/or mask arguments are ignored.

!!! tip "Efficient feature extraction"
    Before extracting **multiple** features, convert your input volume to a mesh using `qim3d.mesh.from_volume` for best performance. This avoids repeated volume-to-mesh conversions under the hood during feature extraction.

::: qim3d.features
    options:
        members:
            - area
            - volume
            - size
            - sphericity
            - roughness
            - mean_std_intensity
