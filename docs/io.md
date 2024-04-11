# Data input and output
Dealing with volumetric data can be done by `qim3d` for the most common image formats available.

Currently, it is possible to directly load `tiff`, `h5`, `nii`,`txm`, `vol` and common `PIL` formats using one single function.

::: qim3d.io
    options:
        members:
            - load
            - save
            - Downloader
            - ImgExamples