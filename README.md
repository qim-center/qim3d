# QIM3D (Quantitative Imaging in 3D)

QIM is a Python library for 3D quantitative imaging analysis. It provides functionality for handling data, as well as tools for visualization and analysis.

&nbsp;
## Installation

Install using pip:

```bash
pip install qim3d
```

&nbsp;
# Usage

&nbsp;
## Loading Data
To load image data from a file, use `qim.io.load()`

```python
import qim3d

# Load a TIFF file
vol = qim3d.io.load("path/to/file.tif")

# Load a TIFF file as a virtual stack
vol = qim3d.io.load("path/to/file.tif", virtual_stack=True)
```

&nbsp;
## GUI Components
QIM provides GUI components for interactive data exploration. The `qim3d.gui` module contains various classes for visualization and analysis:

```python
import qim3d

app = qim3d.gui.iso3d.Interface()
app.launch()
```

Graphical interfaces currently available:
- Data exploration tool (`qim3d.gui.data_exploration`)
- 3D visualization with isosurfaces (`qim3d.gui.iso3d`)
- Local thickness (`qim3d.gui.local_thickness`)


&nbsp;
# Contributing
Contributions to QIM are welcome! If you find a bug, have a feature request, or would like to contribute code, please open an issue or submit a pull request.

&nbsp;
# License
This project is licensed under the MIT License.