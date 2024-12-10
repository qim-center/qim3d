"""qim3d: A Python package for 3D image processing and visualization.

The qim3d library is designed to make it easier to work with 3D imaging data in Python. 
It offers a range of features, including data loading and manipulation,
image processing and filtering, visualization of 3D data, and analysis of imaging results.

Documentation available at https://platform.qim.dk/qim3d/

"""

__version__ = "0.4.5"


import importlib as _importlib


class _LazyLoader:
    """Lazy loader to load submodules only when they are accessed"""

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def _load(self):
        if self.module is None:
            self.module = _importlib.import_module(self.module_name)
        return self.module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)


# List of submodules
_submodules = [
    "examples",
    "generate",
    "gui",
    "io",
    "models",
    "processing",
    "tests",
    "utils",
    "viz",
    "cli",
]

# Creating lazy loaders for each submodule
for submodule in _submodules:
    globals()[submodule] = _LazyLoader(f"qim3d.{submodule}")
