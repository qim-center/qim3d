# Data input and output
Dealing with volumetric data can be done by `qim3d` for the most common image formats available.

Currently, it is possible to directly load `tiff`, `h5`, `nii`,`txm`, `vol` and common `PIL` formats using one single function.


!!! Example
    ```python
    import qim3d

    # Get some data from examples
    vol = qim3d.examples.blobs_256x256x256

    # Save in a local file
    qim3d.io.save("blobs.tif", vol)

    # Load data from file
    loaded_vol = qim3d.io.load("blobs.tif")
    ```

::: qim3d.io.load
    options:
        members:
            - load

::: qim3d.io.save
    options:
        members:
            - save

::: qim3d.io.downloader
    options:
        members:
            - Downloader