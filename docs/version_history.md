# Release history
Here you can fin details about the version history of `qim3d`


## v0.2.0 (Sept 18, 2023)
```pip install qim3d==0.2.0```

Includes new develoments toward the usability of the library, as well as its integration with the QIM platform.

- Refactored code for the graphical interfaces
    - For the local thicknes GUI, now it is possible to pass and receive numpy arrays instead of using the upload functionality.
- Improved data loader
    - Now the extensions `tif`, `h5` and `txm` are supported.
- Added `qim3d.viz.slice_viz` for easy slice visualization.
- U-net model creation
    - Model availabe from `qim3d.models.UNet`
    - Data augmentation class at `qim3d.utils.Augmentation`
    - Notebook with full pipeline at `docs/notebooks/Unet.ipynb`
- Image examples accessible from `qim3d.examples`


## v0.1.3 (May 17, 2023)
```pip install qim3d==0.1.3```

First stable release.

- Simple Tiff data loader
- Graphical interfaces
    - Local thickness
    - 3D Isosurfaces
    - Data exploration tool