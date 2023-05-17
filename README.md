# QIM (Quantitative Imaging Library)

QIM is a Python library for quantitative imaging analysis. It provides functionality for loading and saving image data, as well as tools for visualization and analysis.

## Features

- Load image data from TIFF and HDF5 files.
- GUI components for interactive data exploration.
- Tools for performing quantitative analysis on image data.

## Installation

QIM can be installed using pip:

```shell
pip install qim
```

## Usage

### Loading Data
To load image data from a file, use the qim.io.load() function:

```python
import qim

# Load a TIFF file
data = qim.io.load("path/to/file.tif")

# Load a TIFF file as a virtual stack
data = qim.io.load("path/to/file.tif", virtual_stack=True)
```

### GUI Components
QIM provides GUI components for interactive data exploration. The qim.gui module contains various classes for visualization and analysis:
```python
import qim

app = qim.gui.iso3d.Interface()
app.launch()
```

# Contributing
Contributions to QIM are welcome! If you find a bug, have a feature request, or would like to contribute code, please open an issue or submit a pull request.

# License
This project is licensed under the MIT License.