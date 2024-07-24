import os

from setuptools import find_packages, setup

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="qim3d",
    version="0.3.9",
    author="Felipe Delestro",
    author_email="fima@dtu.dk",
    description="QIM tools and user interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://platform.qim.dk/qim3d",
    packages=find_packages(),
    include_package_data=True,
    entry_points = {
        'console_scripts': [
            'qim3d=qim3d.utils.cli:main'
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
        "albumentations>=1.3.1",
        "gradio>=4.27.0",
        "h5py>=3.9.0",
        "localthickness>=0.1.2",
        "matplotlib>=3.8.0",
        "pydicom>=2.4.4",
        "monai>=1.2.0",
        "numpy>=1.26.0",
        "outputformat>=0.1.3",
        "Pillow>=10.0.1",
        "plotly>=5.14.1",
        "scipy>=1.11.2",
        "seaborn>=0.12.2",
        "setuptools>=68.0.0",
        "tifffile>=2023.4.12",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "torchinfo>=1.8.0",
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
    ],
)
