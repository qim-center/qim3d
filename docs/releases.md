---
icon: fontawesome/solid/bookmark
---

# Release History

Below, you'll find details about the version history of `qim3d`.

As the library is still in its early development stages, there may be breaking changes before `v1.0` without prior deprecation warnings. Therefore, it's advisable to review the release history for more information if you encounter any issues.


### v0.3.2 (23/02/2024)

This version focus on the increased usability of the `qim3d` library

- Online documentation available at [https://platform.qim.dk/qim3d](https://platform.qim.dk/qim3d)
- Virtual stacks also available for `txm` files
- Updated GUI launch pipeline
- New functionalities for `qim3d.vix.slices`
- Introduction of `qim3d.processing.filters` 🎉 
- Introduction of `qim3d.viz.k3d` 🎉 

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