import os
import re
from setuptools import find_packages, setup

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read the version from the __init__.py file
def read_version():
    with open(os.path.join("qim3d", "__init__.py"), "r", encoding="utf-8") as f:
        version_file = f.read()
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="qim3d",
    version=read_version(),
    author="Felipe Delestro",
    author_email="fima@dtu.dk",
    description="QIM tools and user interfaces for volumetric imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://platform.qim.dk/qim3d",
    packages=find_packages(),
    include_package_data=True,
    entry_points = {
        'console_scripts': [
            'qim3d=qim3d.cli:main'
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: User Interfaces",
    ],
    python_requires=">=3.10",
    install_requires=[
        "gradio==4.44",
        "h5py>=3.9.0",
        "localthickness>=0.1.2",
        "matplotlib>=3.8.0",
        "pydicom==2.4.4",
        "numpy>=1.26.0",
        "outputformat>=0.1.3",
        "Pillow>=10.0.1",
        "plotly>=5.14.1",
        "scipy>=1.11.2",
        "seaborn>=0.12.2",
        "setuptools>=68.0.0",
        "tifffile==2023.8.12",
        "imagecodecs==2023.7.10",
        "tqdm>=4.65.0",
        "nibabel>=5.2.0",
        "ipywidgets>=8.1.2",
        "dask>=2023.6.0",
        "k3d>=2.16.1",
        "olefile>=0.46",
        "psutil>=5.9.0",
        "structure-tensor>=0.2.1",
        "noise>=1.2.2",
        "zarr>=2.18.2",
        "ome_zarr>=0.9.0",
        "dask-image>=2024.5.3",
        "scikit-image>=0.24.0",
        "trimesh>=4.4.9"
    ],
    extras_require={
    "deep-learning": [
        "albumentations>=1.3.1",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "torchinfo>=1.8.0",
        "monai>=1.2.0",
    ]
}
)
