# <img src="assets/qim3d-logo.png" width="300px">


`qim3d` is a Python library for quantitative imaging analysis in 3D. It provides functionality for handling data, as well as tools for visualization and analysis.


## Installation

Install the latest stable version by using pip:

```
pip install qim3d
```


## Getting started

Some basic funtionalites are descibred here. The full documentation is still under development.

### Loading Data
To load image data from a file, use `qim.io.load()`

```python
import qim3d

# Load a file
vol = qim3d.io.load("path/to/file.tif")

# Load a file as a virtual stack
vol = qim3d.io.load("path/to/file.tif", virtual_stack=True)
```

### Visualize data
You can easily check slices from your volume using `slices`

```python
import qim3d

img = qim3d.examples.fly_150x256x256

# By default shows the middle slice
qim3d.viz.slices(img)

# Or we can specifly positions
qim3d.viz.slices(img, position=[0,32,128])

# Parameters for size and colormap are also possible
qim3d.viz.slices(img, img_width=6, img_height=6, cmap="inferno")

```


### GUI Components
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

## Contributing
Contributions to `qim3d` are welcome! If you find a bug, have a feature request, or would like to contribute code, please open an issue or submit a pull request.

You can find us at Gitlab:
https://lab.compute.dtu.dk/QIM/tools/qim3d

## License
This project is licensed under the MIT License.