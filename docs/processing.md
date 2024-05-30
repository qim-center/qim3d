# Processing data

Here, we provide functionalities designed specifically for 3D image analysis and processing. From filter pipelines to structure tensor computation and blob detection, `qim3d` equips you with the tools you need to extract meaningful insights from your data.

::: qim3d.processing
    options:
        members:
            - structure_tensor
            - local_thickness
            - get_3d_cc
            - Pipeline
            - Blob

::: qim3d.processing.operations
    options:
        members:
            - remove_background
            - watershed
            - edge_fade
            - fade_mask
