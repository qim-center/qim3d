# Data handling
Dealing with volumetric data can be done by `qim3d` for the most common image formats available. This includes loading, saving and file conversions.

Currently, it is possible to directly load `tiff`, `h5`, `nii`,`txm`, `vol` and common `PIL` formats using one single function.

Additionally, synthtetic volumetric data can be generated using the `generate` module.

::: qim3d.io
    options:
        members:
            - load            
            - save
            - Downloader
            - export_ome_zarr
            - import_ome_zarr
            - save_mesh
            - load_mesh

::: qim3d.mesh
    options:
        members:
            - from_volume

::: qim3d.generate
    options:
        members:
            - noise_object
            - noise_object_collection
