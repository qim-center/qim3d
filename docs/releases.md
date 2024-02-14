# Release history
Here you can fin details about the version history of `qim3d`

## v0.3.1 (February 1, 2024)

- Save functionality for all file formats

### Just for tests

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent non dolor non justo pharetra elementum porttitor at quam. Duis quam ligula, consequat vitae dolor non, facilisis tincidunt justo. Aliquam congue ex ac nibh tristique, et fringilla odio hendrerit. Cras sit amet dui mauris. Curabitur vitae nibh ut dui luctus cursus at id orci. Proin quam lacus, finibus in porttitor sed, ultrices vel ante. Phasellus ut rhoncus diam. Vestibulum vel ultricies orci, ut vehicula libero. Sed bibendum velit sed volutpat maximus. Maecenas non euismod ipsum. Donec eu tempor lorem. Donec lectus turpis, interdum eget commodo sed, euismod id sapien. Fusce malesuada tortor est.



## v0.3.0 (January 23, 2024)

- Introduction of qim3d CLI
- Data Explorer GUI

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent non dolor non justo pharetra elementum porttitor at quam. Duis quam ligula, consequat vitae dolor non, facilisis tincidunt justo. Aliquam congue ex ac nibh tristique, et fringilla odio hendrerit. Cras sit amet dui mauris. Curabitur vitae nibh ut dui luctus cursus at id orci. Proin quam lacus, finibus in porttitor sed, ultrices vel ante. Phasellus ut rhoncus diam. Vestibulum vel ultricies orci, ut vehicula libero. Sed bibendum velit sed volutpat maximus. Maecenas non euismod ipsum. Donec eu tempor lorem. Donec lectus turpis, interdum eget commodo sed, euismod id sapien. Fusce malesuada tortor est.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent non dolor non justo pharetra elementum porttitor at quam. Duis quam ligula, consequat vitae dolor non, facilisis tincidunt justo. Aliquam congue ex ac nibh tristique, et fringilla odio hendrerit. Cras sit amet dui mauris. Curabitur vitae nibh ut dui luctus cursus at id orci. Proin quam lacus, finibus in porttitor sed, ultrices vel ante. Phasellus ut rhoncus diam. Vestibulum vel ultricies orci, ut vehicula libero. Sed bibendum velit sed volutpat maximus. Maecenas non euismod ipsum. Donec eu tempor lorem. Donec lectus turpis, interdum eget commodo sed, euismod id sapien. Fusce malesuada tortor est.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent non dolor non justo pharetra elementum porttitor at quam. Duis quam ligula, consequat vitae dolor non, facilisis tincidunt justo. Aliquam congue ex ac nibh tristique, et fringilla odio hendrerit. Cras sit amet dui mauris. Curabitur vitae nibh ut dui luctus cursus at id orci. Proin quam lacus, finibus in porttitor sed, ultrices vel ante. Phasellus ut rhoncus diam. Vestibulum vel ultricies orci, ut vehicula libero. Sed bibendum velit sed volutpat maximus. Maecenas non euismod ipsum. Donec eu tempor lorem. Donec lectus turpis, interdum eget commodo sed, euismod id sapien. Fusce malesuada tortor est.


## v0.2.0 (Sept 18, 2023)

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

First stable release.

- Simple Tiff data loader
- Graphical interfaces
    - Local thickness
    - 3D Isosurfaces
    - Data exploration tool