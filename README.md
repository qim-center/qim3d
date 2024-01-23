# QIM3D (Quantitative Imaging in 3D)

`qim3D` is a Python library for quantitative imaging analysis in 3D. It provides functionality for handling data, as well as tools for visualization and analysis.

This library contains the tools and functionalities of the QIM platform, accessible at https://qim.dk/platform

## Installation

Install the latest stable version by using pip:

```bash
pip install qim3d
```

Or clone this repository for the most recent version.


# Usage
Some basic funtionalites are descibred here. The full documentation is still under development.

## Loading Data
To load image data from a file, use `qim.io.load()`

```python
import qim3d

# Load a file
vol = qim3d.io.load("path/to/file.tif")

# Load a file as a virtual stack
vol = qim3d.io.load("path/to/file.tif", virtual_stack=True)
```

## Visualize data
You can easily check slices from your volume using `slice_viz`

```python
import qim3d

img = qim3d.examples.fly_150x256x256

# By default shows the middle slice
qim3d.viz.slice_viz(img)

# Or we can specifly positions
qim3d.viz.slice_viz(img, position=[0,32,128])

# Parameters for size and colormap are also possible
qim3d.viz.slice_viz(img, img_width=6, img_height=6, cmap="inferno")

```


## GUI Components
The library also provides GUI components for interactive data analysis and exploration. 
The `qim3d.gui` module contains various classes for visualization and analysis:

```python
import qim3d

app = qim3d.gui.iso3d.Interface()
app.launch()
```

GUIs can also be launched using the Qim3D CLI:
```
$ qim3d gui --data-explorer
```

# Contributing
Contributions to `qim3d` are welcome! If you find a bug, have a feature request, or would like to contribute code, please open an issue or submit a pull request.

# License
This project is licensed under the MIT License.