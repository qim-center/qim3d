# Quantitative Imaging in 3D

<img src="docs/assets/qim3d-logo.png" alt="qim3d logo" style="width:384px">

[![PyPI version](https://badge.fury.io/py/qim3d.svg)](https://badge.fury.io/py/qim3d)
[![Downloads](https://static.pepy.tech/badge/qim3d)](https://pepy.tech/project/qim3d)


The `qim3d` (kɪm θriː diː) library is designed to make it easier to work with 3D imaging data in Python. It offers a range of features, including data loading and manipulation, image processing and filtering, visualization of 3D data, and analysis of imaging results.

You can easily load and process 3D image data from various file formats, apply filters and transformations to the data, visualize the results using interactive plots and 3D rendering, and perform quantitative analysis on the images.

Documentation available at https://platform.qim.dk/qim3d/

For more information on the QIM center visit https://qim.dk/

## Installation

We recommend using a conda environment:

```bash
conda create -n qim3d python=3.11
```

After the environment is created, activate it by running:
```bash
conda activate qim3d
```

And then installation is easy using pip:
```bash
pip install qim3d
```

Remember that the environment needs to be activated each time you use `qim3d`!

For more detailed instructions and troubleshooting, please refer to the [documentation](https://platform.qim.dk/qim3d/#installation).

## Examples

### Interactive volume slicer

```python
import qim3d

vol = qim3d.examples.bone_128x128x128
qim3d.viz.slicer(vol)
```
![viz slicer](docs/assets/screenshots/viz-slicer.gif)

### Line profile

```python
import qim3d

vol = qim3d.examples.bone_128x128x128
qim3d.viz.line_profile(vol)
```
![line profile](docs/assets/screenshots/viz-line_profile.gif)

### Threshold exploration
```python
import qim3d

# Load a sample volume
vol = qim3d.examples.bone_128x128x128

# Visualize interactive thresholding
qim3d.viz.threshold(vol)
```
![threshold exploration](docs/assets/screenshots/interactive_thresholding.gif)



### Synthetic data generation

```python
import qim3d

# Generate synthetic collection of volumes
num_volumes = 15
volume_collection, labels = qim3d.generate.volume_collection(num_volumes = num_volumes)

# Visualize the collection
qim3d.viz.volumetric(volume_collection)
```
![synthetic collection](docs/assets/screenshots/synthetic_collection_default_rotation.gif )

### Structure tensor analysis

```python
import qim3d

vol = qim3d.examples.NT_128x128x128
val, vec = qim3d.processing.structure_tensor(vol, visualize = True, axis = 2)
```

![structure tensor](docs/assets/screenshots/structure_tensor_visualization.gif)

## Support

The development of the `qim3d` is supported by the Infrastructure for Quantitative AI-based Tomography **QUAITOM** which is supported by a Novo Nordisk Foundation Data Science Programme grant (Grant number NNF21OC0069766).

<img src="https://novonordiskfonden.dk//app/uploads/NNF-INT_logo_tagline_blue_RGB_solid.png" alt="NNF" style="width:256px">
