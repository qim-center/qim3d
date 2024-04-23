# Release History
[![PyPI version](https://badge.fury.io/py/qim3d.svg)](https://badge.fury.io/py/qim3d)
[![Downloads](https://static.pepy.tech/badge/qim3d)](https://pepy.tech/project/qim3d)


Below, you'll find details about the version history of `qim3d`.

As the library is still in its early development stages, **there may be breaking changes** before `v1.0` without prior deprecation warnings. Therefore, it's advisable to review the release history for more information if you encounter any issues.

And remember to keep your pip installation [up to date](/qim3d/#upgrade) so that you have the latest features!

### v0.3.4 (coming soon!)
- Documentation for `qim3d.viz.plot_cc`
- Fixed issue with Annotation tool and recent Gradio versions
- New colormap: `qim3d.viz.colormaps.qim`, showcasing the Qim colors!
- Object separation using `qim3d.processing.operations.watershed`
- Added option to pass `dim_order` to `vol/vgi` files
- The 'Data Explorer' GUI now can load image sequences also

### v0.3.3 (11/04/2024)
- Introduction of `qim3d.viz.slicer` (and also `qim3d.viz.orthogonal` ) 🎉
- Introduction of `qim3d.gui.annotation_tool` 🎉
- Introduction of `qim3d.processing.Blob` for blob detection 🎉
- Introduction of `qim3d.processing.local_thickness` 🎉
- Introduction of `qim3d.processing.structure_tensor` 🎉
- Support for loading DICOM files with `qim3d.io.load`
- Introduction of `qim3d.processing.get_3d_cc` for 3D connected components and `qim3d.viz.plot_cc` for associated visualization 🎉
- Introduction of `qim3d.viz.colormaps` for easy visualization of e.g. multi-label segmentation results 🎉
- Introduction of `qim3d.processing.operations.background_removal` 🎉
- Documentation refactoring
- Fixed bug preventing `Data Explorer` to show files

### v0.3.2 (23/02/2024)

This version focus on the increased usability of the `qim3d` library

- Online documentation available at [https://platform.qim.dk/qim3d](https://platform.qim.dk/qim3d)
- Virtual stacks also available for `txm` files
- Updated GUI launch pipeline
- New functionalities for `qim3d.viz.slices`
- Introduction of `qim3d.processing.filters` 🎉 
- Introduction of `qim3d.viz.vol` 🎉 

### v0.3.1 (01/02/2024)

Release expanding the IO functionalities

- Support for loading `vol` `nii` and `bigtiff` files
- Data loader now supports `virtual_stack`
- Save functionality for all file formats except `txm`

### v0.3.0 (23/01/2024)
- Introduction of qim3d CLI 🎉 
- Introduction of memory utils 🎉 
- Data Explorer GUI
- Save functionality for `tif` files

### v0.2.0 (18/09/2023)

Includes new develoments toward the usability of the library, as well as its integration with the QIM platform.

- Refactored code for the graphical interfaces
    - For the local thicknes GUI, now it is possible to pass and receive numpy arrays instead of using the upload functionality.
- Improved data loader
    - Now the extensions `tif`, `h5` and `txm` are supported.
- Added `qim3d.viz.slices` for easy slice visualization.
- U-net model creation
    - Model availabe from `qim3d.models.UNet`
    - Data augmentation class at `qim3d.utils.Augmentation`
    - Notebook with full pipeline at `docs/notebooks/Unet.ipynb`
- Image examples accessible from `qim3d.examples`


### v0.1.3 (17/05/2023)

First stable release.

- Simple Tiff data loader
- Graphical interfaces
    - Local thickness
    - 3D Isosurfaces
    - Data exploration tool