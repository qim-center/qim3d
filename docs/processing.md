# Processing data

Here, we provide functionalities designed specifically for 3D image analysis and processing. From filter pipelines to structure tensor computation and blob detection, `qim3d` equips you with the tools you need to extract meaningful insights from your data.

::: qim3d.processing
    options:
        members:
            - structure_tensor
            - local_thickness
            - get_lines
            - segment_layers

::: qim3d.mesh
    options:
        members:
            - from_volume

::: qim3d.detection
    options:
        members:
            - blobs

::: qim3d.operations
    options:
        members:
            - remove_background
            - fade_mask
            - overlay_rgb_images

::: qim3d.segmentation
    options:
      members:
        - watershed
        - get_3d_cc

::: qim3d.filters
    options:
        members:
            - gaussian
            - median
            - maximum
            - minimum
            - tophat
::: qim3d.filters.Pipeline
    options:
        members:
            - append

::: qim3d.features
    options:
        members:
            - area
            - volume
            - sphericity